"""
Graph Optimization Service for Google Ads campaigns.

This service implements graph theory algorithms for optimizing various aspects
of Google Ads campaigns, such as keyword relationships, campaign structure,
and bidding strategies.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Set
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import time
import community as community_louvain
from pathlib import Path

# Correct relative import for BaseService
from ..base_service import BaseService

logger = logging.getLogger(__name__)


class GraphOptimizationService(BaseService):
    """
    Service for applying graph theory algorithms to Google Ads optimization.

    This service provides methods for building and analyzing graphs representing
    various aspects of Google Ads campaigns, such as keyword relationships,
    campaign structure, and conversion paths.
    """

    def __init__(
        self,
        ads_api=None,
        optimizer=None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the graph optimization service.

        Args:
            ads_api: Google Ads API client instance
            optimizer: AI optimizer instance
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(ads_api, optimizer, config, logger)

        # Validate and load required configurations
        self._validate_config()

        # Dictionary to store cached graphs
        self.graphs = {}

        # Create output directories
        os.makedirs("data/graph_analysis", exist_ok=True)
        os.makedirs("reports/graph_analysis", exist_ok=True)

        self.logger.info("Graph Optimization Service initialized")

    def _validate_config(self) -> None:
        """Validate the configuration for the graph optimization service."""
        # Set default values for missing configurations
        if "graph_optimization.min_weight_threshold" not in self.config:
            self.config["graph_optimization.min_weight_threshold"] = 0.1
            self.logger.warning(
                "Missing config: graph_optimization.min_weight_threshold, using default: 0.1"
            )

        if "graph_optimization.max_graph_size" not in self.config:
            self.config["graph_optimization.max_graph_size"] = 1000
            self.logger.warning(
                "Missing config: graph_optimization.max_graph_size, using default: 1000"
            )

    def build_keyword_relationship_graph(
        self, days: int = 30, campaign_id: Optional[str] = None, min_impressions: int = 100
    ) -> Dict[str, Any]:
        """
        Build a graph representing relationships between keywords based on performance data.

        Args:
            days: Number of days of data to analyze
            campaign_id: Optional campaign ID to filter data
            min_impressions: Minimum impressions for a keyword to be included

        Returns:
            Dictionary with graph analysis results
        """
        start_time = datetime.now()
        self.logger.info(f"Building keyword relationship graph for the last {days} days")

        try:
            # Fetch keyword performance data
            if not self.ads_api:
                raise ValueError("Google Ads API client is not initialized")

            # Get keyword performance data
            keyword_data = self.ads_api.get_keyword_performance(days=days, campaign_id=campaign_id)

            if not keyword_data:
                return {"status": "error", "message": "No keyword data available"}

            # Filter keywords with sufficient impressions
            keywords = [kw for kw in keyword_data if kw.get("impressions", 0) >= min_impressions]

            if not keywords:
                return {
                    "status": "error",
                    "message": f"No keywords with at least {min_impressions} impressions found",
                }

            # Create a graph
            G = nx.Graph()

            # Add nodes (keywords)
            for kw in keywords:
                G.add_node(
                    kw["keyword_text"],
                    impressions=kw.get("impressions", 0),
                    clicks=kw.get("clicks", 0),
                    conversions=kw.get("conversions", 0),
                    cost=kw.get("cost", 0),
                    avg_cpc=kw.get("average_cpc", 0),
                    campaign_id=kw.get("campaign_id", ""),
                    campaign_name=kw.get("campaign_name", ""),
                    ad_group_id=kw.get("ad_group_id", ""),
                    ad_group_name=kw.get("ad_group_name", ""),
                    match_type=kw.get("match_type", ""),
                )

            # Add edges based on similarity
            min_weight = self.config.get("graph_optimization.min_weight_threshold", 0.1)
            edges_added = self._create_keyword_edges(G, keywords, min_weight)

            # If we have too few edges, lower the threshold and try again
            if edges_added < len(keywords) / 2:
                self.logger.info(
                    f"Only {edges_added} edges created. Lowering threshold and retrying."
                )
                min_weight = min_weight / 2
                edges_added = self._create_keyword_edges(G, keywords, min_weight)

            # Store the graph
            graph_id = f"keyword_relationship_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.graphs[graph_id] = G

            # Analyze the graph
            analysis = self._analyze_graph(G, graph_id)

            # Save visualization
            viz_path = self._save_graph_visualization(G, graph_id)

            # Save the graph data
            self._save_graph_data(G, graph_id)

            result = {
                "status": "success",
                "graph_id": graph_id,
                "keywords_analyzed": len(keywords),
                "nodes": len(G.nodes()),
                "edges": len(G.edges()),
                "analysis": analysis,
                "visualization_path": viz_path,
            }

            self._track_execution(start_time, True)
            return result

        except Exception as e:
            self.logger.error(f"Error building keyword relationship graph: {str(e)}")
            self._track_execution(start_time, False)
            return {"status": "error", "message": str(e)}

    def _create_keyword_edges(
        self, G: nx.Graph, keywords: List[Dict[str, Any]], min_weight: float
    ) -> int:
        """
        Create edges between keywords based on semantic and performance similarity.

        Args:
            G: NetworkX graph
            keywords: List of keyword dictionaries
            min_weight: Minimum weight threshold for adding an edge

        Returns:
            Number of edges added
        """
        edges_added = 0
        keyword_texts = [kw["keyword_text"] for kw in keywords]

        # Create a mapping from keyword text to index
        keyword_to_idx = {kw["keyword_text"]: i for i, kw in enumerate(keywords)}

        # Calculate similarities between all pairs of keywords
        for i, kw1 in enumerate(keywords):
            for j in range(i + 1, len(keywords)):
                kw2 = keywords[j]

                # Skip keywords in different campaigns if desired
                if kw1.get("campaign_id") != kw2.get("campaign_id"):
                    # Uncomment if you want to skip cross-campaign edges
                    # continue
                    pass

                # Calculate semantic similarity (word overlap)
                words1 = set(kw1["keyword_text"].lower().split())
                words2 = set(kw2["keyword_text"].lower().split())

                if not words1 or not words2:
                    continue

                word_overlap = len(words1.intersection(words2)) / max(len(words1), len(words2))

                # Calculate performance similarity
                perf_sim = self._calculate_performance_similarity(kw1, kw2)

                # Combine similarities (weighted average)
                combined_weight = 0.7 * word_overlap + 0.3 * perf_sim

                # Add edge if weight is above threshold
                if combined_weight >= min_weight:
                    G.add_edge(
                        kw1["keyword_text"],
                        kw2["keyword_text"],
                        weight=combined_weight,
                        semantic_similarity=word_overlap,
                        performance_similarity=perf_sim,
                    )
                    edges_added += 1

        return edges_added

    def _calculate_performance_similarity(self, kw1: Dict[str, Any], kw2: Dict[str, Any]) -> float:
        """
        Calculate similarity between keywords based on performance metrics.

        Args:
            kw1: First keyword dictionary
            kw2: Second keyword dictionary

        Returns:
            Similarity score (0-1)
        """
        # Calculate CTR for both keywords
        ctr1 = kw1.get("clicks", 0) / max(1, kw1.get("impressions", 1))
        ctr2 = kw2.get("clicks", 0) / max(1, kw2.get("impressions", 1))

        # Calculate conversion rate for both keywords
        conv_rate1 = kw1.get("conversions", 0) / max(1, kw1.get("clicks", 1))
        conv_rate2 = kw2.get("conversions", 0) / max(1, kw2.get("clicks", 1))

        # Calculate cost per conversion
        cpc1 = (
            kw1.get("cost", 0) / max(1, kw1.get("conversions", 1))
            if kw1.get("conversions", 0) > 0
            else float("inf")
        )
        cpc2 = (
            kw2.get("cost", 0) / max(1, kw2.get("conversions", 1))
            if kw2.get("conversions", 0) > 0
            else float("inf")
        )

        # Calculate normalized differences
        ctr_diff = 1 - min(1, abs(ctr1 - ctr2) / max(ctr1, ctr2)) if max(ctr1, ctr2) > 0 else 0
        conv_diff = (
            1 - min(1, abs(conv_rate1 - conv_rate2) / max(conv_rate1, conv_rate2))
            if max(conv_rate1, conv_rate2) > 0
            else 0
        )

        # For CPC, handle infinite values
        if cpc1 == float("inf") and cpc2 == float("inf"):
            cpc_diff = 1  # Both have no conversions, so CPC similarity is high
        elif cpc1 == float("inf") or cpc2 == float("inf"):
            cpc_diff = 0  # One has conversions, one doesn't, so CPC similarity is low
        else:
            cpc_diff = 1 - min(1, abs(cpc1 - cpc2) / max(cpc1, cpc2)) if max(cpc1, cpc2) > 0 else 0

        # Combine metrics (weighted average)
        return 0.4 * ctr_diff + 0.4 * conv_diff + 0.2 * cpc_diff

    def _analyze_graph(self, G: nx.Graph, graph_id: str) -> Dict[str, Any]:
        """
        Analyze a graph to extract insights.

        Args:
            G: NetworkX graph
            graph_id: Identifier for the graph

        Returns:
            Dictionary with analysis results
        """
        analysis = {}

        # Basic metrics
        analysis["node_count"] = len(G.nodes())
        analysis["edge_count"] = len(G.edges())
        analysis["density"] = nx.density(G)
        analysis["average_clustering"] = nx.average_clustering(G)

        # Connected components
        connected_components = list(nx.connected_components(G))
        analysis["connected_components"] = len(connected_components)

        # Largest connected component
        if connected_components:
            largest_cc = max(connected_components, key=len)
            analysis["largest_component_size"] = len(largest_cc)
            analysis["largest_component_percentage"] = len(largest_cc) / len(G.nodes())

        # Centrality metrics
        # Degree centrality
        degree_centrality = nx.degree_centrality(G)
        analysis["top_degree_centrality"] = self._get_top_nodes(degree_centrality, 10)

        # PageRank
        try:
            pagerank = nx.pagerank(G, weight="weight")
            analysis["top_pagerank"] = self._get_top_nodes(pagerank, 10)
        except:
            analysis["top_pagerank"] = []

        # Betweenness centrality for smaller graphs (can be slow for large graphs)
        if len(G.nodes()) <= 500:
            try:
                betweenness = nx.betweenness_centrality(
                    G, weight="weight", k=min(100, len(G.nodes()))
                )
                analysis["top_betweenness_centrality"] = self._get_top_nodes(betweenness, 10)
            except:
                analysis["top_betweenness_centrality"] = []

        # Community detection
        try:
            # Use Louvain method for community detection
            partition = community_louvain.best_partition(G, weight="weight")
            communities = {}
            for node, community_id in partition.items():
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(node)

            # Sort communities by size
            sorted_communities = sorted(communities.values(), key=len, reverse=True)

            # Add top communities to analysis
            top_communities = []
            for i, community in enumerate(sorted_communities[:5]):  # Top 5 communities
                if len(community) > 2:  # Only include communities with more than 2 nodes
                    top_communities.append(
                        {
                            "id": i,
                            "size": len(community),
                            "keywords": community[:10],  # Show first 10 keywords in each community
                        }
                    )

            analysis["communities"] = top_communities
            analysis["community_count"] = len(communities)
            analysis["modularity"] = self._calculate_modularity(G, partition)

        except Exception as e:
            self.logger.warning(f"Error in community detection: {str(e)}")
            analysis["communities"] = []
            analysis["community_count"] = 0
            analysis["modularity"] = 0

        return analysis

    def _get_top_nodes(
        self, centrality_dict: Dict[str, float], n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get the top N nodes by centrality.

        Args:
            centrality_dict: Dictionary mapping node IDs to centrality values
            n: Number of top nodes to return

        Returns:
            List of dictionaries with node information
        """
        sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)

        return [{"node": node, "value": round(value, 4)} for node, value in sorted_nodes[:n]]

    def _calculate_modularity(self, G: nx.Graph, partition: Dict[str, int]) -> float:
        """
        Calculate the modularity of a partition.

        Args:
            G: NetworkX graph
            partition: Dictionary mapping nodes to community IDs

        Returns:
            Modularity value
        """
        try:
            return community_louvain.modularity(partition, G, weight="weight")
        except:
            return 0.0

    def _save_graph_visualization(self, G: nx.Graph, graph_id: str) -> str:
        """
        Save a visualization of the graph.

        Args:
            G: NetworkX graph
            graph_id: Identifier for the graph

        Returns:
            Path to the saved visualization
        """
        try:
            plt.figure(figsize=(12, 12))

            # For large graphs, use a simpler layout
            if len(G.nodes()) > 200:
                pos = nx.spring_layout(G, k=0.5, iterations=50)
                node_size = 10
                with_labels = False
            else:
                pos = nx.spring_layout(G, k=0.5, iterations=100)
                node_size = 100
                with_labels = True

            # Get edge weights for width
            edge_weights = [G[u][v].get("weight", 0.1) * 3 for u, v in G.edges()]

            # Draw the network
            nx.draw_networkx_nodes(G, pos, node_size=node_size, alpha=0.7)
            nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.4)

            if with_labels:
                nx.draw_networkx_labels(G, pos, font_size=8, alpha=0.7)

            plt.title(f"Keyword Relationship Graph: {graph_id}")
            plt.axis("off")

            # Save the figure
            file_path = f"reports/graph_analysis/{graph_id}.png"
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close()

            return file_path
        except Exception as e:
            self.logger.error(f"Error saving graph visualization: {str(e)}")
            return "Visualization failed"

    def _save_graph_data(self, G: nx.Graph, graph_id: str) -> None:
        """
        Save graph data to a file.

        Args:
            G: NetworkX graph
            graph_id: Identifier for the graph
        """
        try:
            # Convert graph to JSON-serializable format
            graph_data = {
                "nodes": [{"id": node, **G.nodes[node]} for node in G.nodes()],
                "edges": [{"source": u, "target": v, **G[u][v]} for u, v in G.edges()],
            }

            # Save to data directory
            self.save_data(graph_data, f"{graph_id}.json", "data/graph_analysis")
        except Exception as e:
            self.logger.error(f"Error saving graph data: {str(e)}")

    def build_campaign_structure_graph(
        self, days: int = 30, account_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build a graph representing the structure of campaigns, ad groups, and keywords.

        Args:
            days: Number of days of data to analyze
            account_id: Optional account ID to filter data

        Returns:
            Dictionary with graph analysis results
        """
        start_time = datetime.now()
        self.logger.info(f"Building campaign structure graph for the last {days} days")

        try:
            # Fetch campaign and keyword data
            if not self.ads_api:
                raise ValueError("Google Ads API client is not initialized")

            # Get campaign data
            campaign_data = self.ads_api.get_campaign_performance(days=days)

            if not campaign_data:
                return {"status": "error", "message": "No campaign data available"}

            # Get keyword data
            keyword_data = self.ads_api.get_keyword_performance(days=days)

            if not keyword_data:
                return {"status": "error", "message": "No keyword data available"}

            # Create a directed graph (hierarchy)
            G = nx.DiGraph()

            # Add campaign nodes
            for campaign in campaign_data:
                G.add_node(
                    f"campaign:{campaign['id']}",
                    type="campaign",
                    name=campaign.get("name", ""),
                    impressions=campaign.get("impressions", 0),
                    clicks=campaign.get("clicks", 0),
                    conversions=campaign.get("conversions", 0),
                    cost=campaign.get("cost", 0),
                )

            # Group keywords by ad group
            ad_groups = {}
            for kw in keyword_data:
                ad_group_id = kw.get("ad_group_id", "")
                if not ad_group_id:
                    continue

                if ad_group_id not in ad_groups:
                    ad_groups[ad_group_id] = {
                        "id": ad_group_id,
                        "name": kw.get("ad_group_name", ""),
                        "campaign_id": kw.get("campaign_id", ""),
                        "keywords": [],
                        "impressions": 0,
                        "clicks": 0,
                        "conversions": 0,
                        "cost": 0,
                    }

                # Add the keyword to this ad group
                ad_groups[ad_group_id]["keywords"].append(kw)

                # Aggregate metrics
                ad_groups[ad_group_id]["impressions"] += kw.get("impressions", 0)
                ad_groups[ad_group_id]["clicks"] += kw.get("clicks", 0)
                ad_groups[ad_group_id]["conversions"] += kw.get("conversions", 0)
                ad_groups[ad_group_id]["cost"] += kw.get("cost", 0)

            # Add ad group nodes and edges to campaigns
            for ad_group_id, ad_group in ad_groups.items():
                campaign_id = ad_group["campaign_id"]

                # Add the ad group node
                G.add_node(
                    f"adgroup:{ad_group_id}",
                    type="adgroup",
                    name=ad_group["name"],
                    impressions=ad_group["impressions"],
                    clicks=ad_group["clicks"],
                    conversions=ad_group["conversions"],
                    cost=ad_group["cost"],
                )

                # Add edge from campaign to ad group
                G.add_edge(
                    f"campaign:{campaign_id}",
                    f"adgroup:{ad_group_id}",
                    weight=ad_group["impressions"],
                )

                # Add keyword nodes and edges to ad groups
                for kw in ad_group["keywords"]:
                    kw_id = kw.get("criterion_id", "")
                    if not kw_id:
                        continue

                    # Add the keyword node
                    G.add_node(
                        f"keyword:{kw_id}",
                        type="keyword",
                        text=kw.get("keyword_text", ""),
                        match_type=kw.get("match_type", ""),
                        impressions=kw.get("impressions", 0),
                        clicks=kw.get("clicks", 0),
                        conversions=kw.get("conversions", 0),
                        cost=kw.get("cost", 0),
                    )

                    # Add edge from ad group to keyword
                    G.add_edge(
                        f"adgroup:{ad_group_id}",
                        f"keyword:{kw_id}",
                        weight=kw.get("impressions", 0),
                    )

            # Store the graph
            graph_id = f"campaign_structure_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.graphs[graph_id] = G

            # Analyze the graph
            analysis = self._analyze_structure_graph(G)

            # Save visualization
            viz_path = self._save_structure_visualization(G, graph_id)

            # Save the graph data
            self._save_graph_data(G, graph_id)

            result = {
                "status": "success",
                "graph_id": graph_id,
                "campaigns": len([n for n in G.nodes() if G.nodes[n]["type"] == "campaign"]),
                "ad_groups": len([n for n in G.nodes() if G.nodes[n]["type"] == "adgroup"]),
                "keywords": len([n for n in G.nodes() if G.nodes[n]["type"] == "keyword"]),
                "analysis": analysis,
                "visualization_path": viz_path,
            }

            self._track_execution(start_time, True)
            return result

        except Exception as e:
            self.logger.error(f"Error building campaign structure graph: {str(e)}")
            self._track_execution(start_time, False)
            return {"status": "error", "message": str(e)}

    def _analyze_structure_graph(self, G: nx.DiGraph) -> Dict[str, Any]:
        """
        Analyze a campaign structure graph to extract insights.

        Args:
            G: NetworkX directed graph

        Returns:
            Dictionary with analysis results
        """
        analysis = {}

        # Basic metrics
        analysis["node_count"] = len(G.nodes())
        analysis["edge_count"] = len(G.edges())

        # Count by type
        node_types = {G.nodes[n]["type"] for n in G.nodes()}
        type_counts = {
            t: len([n for n in G.nodes() if G.nodes[n]["type"] == t]) for t in node_types
        }
        analysis["type_counts"] = type_counts

        # Calculate average children per node type
        if "campaign" in type_counts and "adgroup" in type_counts:
            analysis["avg_adgroups_per_campaign"] = type_counts["adgroup"] / max(
                1, type_counts["campaign"]
            )

        if "adgroup" in type_counts and "keyword" in type_counts:
            analysis["avg_keywords_per_adgroup"] = type_counts["keyword"] / max(
                1, type_counts["adgroup"]
            )

        # Find campaigns with most ad groups
        campaign_adgroup_counts = {}
        for node in G.nodes():
            if G.nodes[node]["type"] == "campaign":
                children = list(G.successors(node))
                campaign_adgroup_counts[node] = len(children)

        # Get top campaigns by ad group count
        top_campaigns = sorted(campaign_adgroup_counts.items(), key=lambda x: x[1], reverse=True)
        analysis["top_campaigns_by_adgroup_count"] = [
            {
                "campaign_id": node.split(":")[-1],
                "name": G.nodes[node]["name"],
                "adgroup_count": count,
            }
            for node, count in top_campaigns[:5]
        ]

        # Find ad groups with most keywords
        adgroup_keyword_counts = {}
        for node in G.nodes():
            if G.nodes[node]["type"] == "adgroup":
                children = list(G.successors(node))
                adgroup_keyword_counts[node] = len(children)

        # Get top ad groups by keyword count
        top_adgroups = sorted(adgroup_keyword_counts.items(), key=lambda x: x[1], reverse=True)
        analysis["top_adgroups_by_keyword_count"] = [
            {
                "adgroup_id": node.split(":")[-1],
                "name": G.nodes[node]["name"],
                "keyword_count": count,
            }
            for node, count in top_adgroups[:5]
        ]

        # Find campaigns with highest performance metrics
        campaign_metrics = {}
        for node in G.nodes():
            if G.nodes[node]["type"] == "campaign":
                campaign_metrics[node] = {
                    "impressions": G.nodes[node]["impressions"],
                    "clicks": G.nodes[node]["clicks"],
                    "conversions": G.nodes[node]["conversions"],
                    "cost": G.nodes[node]["cost"],
                }

        # Get top campaigns by conversions
        top_campaigns_conv = sorted(
            campaign_metrics.items(), key=lambda x: x[1]["conversions"], reverse=True
        )
        analysis["top_campaigns_by_conversions"] = [
            {
                "campaign_id": node.split(":")[-1],
                "name": G.nodes[node]["name"],
                "conversions": metrics["conversions"],
            }
            for node, metrics in top_campaigns_conv[:5]
        ]

        return analysis

    def _save_structure_visualization(self, G: nx.DiGraph, graph_id: str) -> str:
        """
        Save a visualization of the campaign structure graph.

        Args:
            G: NetworkX directed graph
            graph_id: Identifier for the graph

        Returns:
            Path to the saved visualization
        """
        try:
            plt.figure(figsize=(15, 10))

            # Use hierarchical layout
            pos = (
                nx.nx_agraph.graphviz_layout(G, prog="dot")
                if len(G.nodes()) < 1000
                else nx.spring_layout(G)
            )

            # Differentiate node types by color
            campaign_nodes = [n for n in G.nodes() if G.nodes[n]["type"] == "campaign"]
            adgroup_nodes = [n for n in G.nodes() if G.nodes[n]["type"] == "adgroup"]
            keyword_nodes = [n for n in G.nodes() if G.nodes[n]["type"] == "keyword"]

            # Draw nodes by type
            nx.draw_networkx_nodes(
                G, pos, nodelist=campaign_nodes, node_color="tab:blue", node_size=300, alpha=0.8
            )
            nx.draw_networkx_nodes(
                G, pos, nodelist=adgroup_nodes, node_color="tab:green", node_size=150, alpha=0.6
            )

            # Draw keyword nodes only if there aren't too many
            if len(keyword_nodes) <= 500:
                nx.draw_networkx_nodes(
                    G, pos, nodelist=keyword_nodes, node_color="tab:red", node_size=50, alpha=0.4
                )

            # Draw edges
            nx.draw_networkx_edges(G, pos, arrows=True, alpha=0.2)

            # Draw labels only for campaigns and ad groups
            labels = {n: G.nodes[n]["name"] for n in G.nodes() if G.nodes[n]["type"] != "keyword"}
            if len(labels) <= 100:
                nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

            plt.title(f"Campaign Structure Graph: {graph_id}")
            plt.axis("off")

            # Save the figure
            file_path = f"reports/graph_analysis/{graph_id}.png"
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close()

            return file_path
        except Exception as e:
            self.logger.error(f"Error saving structure visualization: {str(e)}")
            return "Visualization failed"

    def identify_keyword_clusters(
        self,
        graph_id: Optional[str] = None,
        min_cluster_size: int = 3,
        similarity_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Identify clusters of related keywords from a keyword relationship graph.

        Args:
            graph_id: Identifier for a previously built graph, or None to build a new one
            min_cluster_size: Minimum size for a cluster to be included
            similarity_threshold: Minimum similarity score for keywords to be in the same cluster

        Returns:
            Dictionary with keyword cluster results
        """
        start_time = datetime.now()
        self.logger.info(f"Identifying keyword clusters")

        try:
            # Get the graph
            if graph_id and graph_id in self.graphs:
                G = self.graphs[graph_id]
                self.logger.info(f"Using existing graph: {graph_id}")
            else:
                # Build a new graph
                result = self.build_keyword_relationship_graph()
                if result["status"] != "success":
                    return result

                graph_id = result["graph_id"]
                G = self.graphs[graph_id]
                self.logger.info(f"Built new graph: {graph_id}")

            # Ensure we have a keyword relationship graph
            if not isinstance(G, nx.Graph):
                return {"status": "error", "message": "Invalid graph type for cluster analysis"}

            # Filter edges by weight
            H = G.copy()
            for u, v, data in G.edges(data=True):
                if data.get("weight", 0) < similarity_threshold:
                    H.remove_edge(u, v)

            # Find connected components (clusters)
            clusters = list(nx.connected_components(H))

            # Filter by size and sort by size
            valid_clusters = [c for c in clusters if len(c) >= min_cluster_size]
            valid_clusters.sort(key=len, reverse=True)

            # Prepare results
            cluster_results = []
            for i, cluster in enumerate(valid_clusters):
                # Get keywords and their attributes
                keywords = [
                    {
                        "text": kw,
                        "impressions": G.nodes[kw].get("impressions", 0),
                        "clicks": G.nodes[kw].get("clicks", 0),
                        "conversions": G.nodes[kw].get("conversions", 0),
                        "cost": G.nodes[kw].get("cost", 0),
                        "centrality": nx.degree_centrality(G.subgraph(cluster))[kw],
                    }
                    for kw in cluster
                ]

                # Sort by centrality
                keywords.sort(key=lambda x: x["centrality"], reverse=True)

                # Get most central keyword as potential theme
                central_keyword = keywords[0]["text"] if keywords else ""

                # Calculate common words
                common_words = self._find_common_words(cluster)

                # Create cluster dictionary
                cluster_dict = {
                    "id": i,
                    "size": len(cluster),
                    "central_keyword": central_keyword,
                    "common_words": common_words[:5],
                    "keywords": keywords,
                    "avg_similarity": self._calculate_avg_cluster_similarity(G, cluster),
                }

                cluster_results.append(cluster_dict)

            result = {
                "status": "success",
                "graph_id": graph_id,
                "cluster_count": len(cluster_results),
                "clusters": cluster_results,
                "params": {
                    "min_cluster_size": min_cluster_size,
                    "similarity_threshold": similarity_threshold,
                },
            }

            self._track_execution(start_time, True)
            return result

        except Exception as e:
            self.logger.error(f"Error identifying keyword clusters: {str(e)}")
            self._track_execution(start_time, False)
            return {"status": "error", "message": str(e)}

    def _find_common_words(self, keywords: Set[str]) -> List[str]:
        """
        Find common words that appear in multiple keywords.

        Args:
            keywords: Set of keyword strings

        Returns:
            List of common words sorted by frequency
        """
        if not keywords:
            return []

        # Extract all words
        all_words = []
        for kw in keywords:
            words = kw.lower().split()
            all_words.extend(words)

        # Count word frequencies
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        return [word for word, count in sorted_words if count > 1]

    def _calculate_avg_cluster_similarity(self, G: nx.Graph, cluster: Set[str]) -> float:
        """
        Calculate the average similarity between all pairs of keywords in a cluster.

        Args:
            G: NetworkX graph
            cluster: Set of keyword strings

        Returns:
            Average similarity score
        """
        if len(cluster) < 2:
            return 0.0

        # Create a subgraph for the cluster
        subgraph = G.subgraph(cluster)

        # Calculate the average weight of all edges
        total_weight = sum(data.get("weight", 0) for u, v, data in subgraph.edges(data=True))

        # If there are no edges, return 0
        if subgraph.number_of_edges() == 0:
            return 0.0

        return total_weight / subgraph.number_of_edges()

    def optimize_adgroup_structure(
        self,
        cluster_min_size: int = 3,
        similarity_threshold: float = 0.5,
        max_recommendations: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate recommendations for optimizing ad group structure based on keyword clusters.

        Args:
            cluster_min_size: Minimum size for a keyword cluster
            similarity_threshold: Minimum similarity threshold for cluster formation
            max_recommendations: Maximum number of recommendations to generate

        Returns:
            Dictionary with ad group structure recommendations
        """
        start_time = datetime.now()
        self.logger.info("Generating ad group structure optimization recommendations")

        try:
            # First, identify keyword clusters
            cluster_result = self.identify_keyword_clusters(
                min_cluster_size=cluster_min_size, similarity_threshold=similarity_threshold
            )

            if cluster_result["status"] != "success":
                return cluster_result

            # Get existing ad group structure (from most recent graph)
            graph_id = cluster_result["graph_id"]
            G = self.graphs[graph_id]

            # Analyze current ad group distribution across clusters
            adgroup_distributions = {}
            keyword_adgroups = {}
            for node, data in G.nodes(data=True):
                # Skip non-keywords
                if "ad_group_id" not in data:
                    continue

                ad_group_id = data["ad_group_id"]
                ad_group_name = data["ad_group_name"]

                keyword_adgroups[node] = ad_group_id

                # Initialize ad group entry if not exists
                if ad_group_id not in adgroup_distributions:
                    adgroup_distributions[ad_group_id] = {
                        "name": ad_group_name,
                        "keywords": [],
                        "campaigns": set(),
                        "clusters": {},
                    }

                # Add keyword to ad group
                adgroup_distributions[ad_group_id]["keywords"].append(node)

                # Add campaign to ad group's campaigns
                if "campaign_id" in data:
                    adgroup_distributions[ad_group_id]["campaigns"].add(data["campaign_id"])

            # For each cluster, analyze which ad groups its keywords belong to
            recommendations = []

            for cluster in cluster_result["clusters"]:
                cluster_keywords = {kw["text"] for kw in cluster["keywords"]}

                # Count ad groups for this cluster
                cluster_adgroups = {}
                for kw in cluster_keywords:
                    if kw in keyword_adgroups:
                        ag_id = keyword_adgroups[kw]
                        cluster_adgroups[ag_id] = cluster_adgroups.get(ag_id, 0) + 1

                        # Update ad group's cluster distribution
                        if ag_id in adgroup_distributions:
                            adgroup_distributions[ag_id]["clusters"][cluster["id"]] = (
                                cluster_adgroups[ag_id]
                            )

                # If keywords are spread across multiple ad groups, consider consolidation
                if len(cluster_adgroups) > 1:
                    # Check if there's a dominant ad group (>50% of keywords)
                    max_ag_id = (
                        max(cluster_adgroups, key=cluster_adgroups.get)
                        if cluster_adgroups
                        else None
                    )
                    max_count = cluster_adgroups.get(max_ag_id, 0)

                    if max_count > 0 and max_count < 0.7 * len(cluster_keywords):
                        # Suggest consolidation
                        suggestion = {
                            "type": "consolidate",
                            "cluster_id": cluster["id"],
                            "cluster_size": len(cluster_keywords),
                            "common_words": cluster["common_words"],
                            "central_keyword": cluster["central_keyword"],
                            "current_distribution": [
                                {
                                    "ad_group_id": ag_id,
                                    "ad_group_name": (
                                        adgroup_distributions[ag_id]["name"]
                                        if ag_id in adgroup_distributions
                                        else "Unknown"
                                    ),
                                    "keyword_count": count,
                                    "percentage": round(count / len(cluster_keywords) * 100, 1),
                                }
                                for ag_id, count in sorted(
                                    cluster_adgroups.items(), key=lambda x: x[1], reverse=True
                                )
                            ],
                            "recommendation": f"Consolidate {len(cluster_keywords)} related keywords into a single ad group",
                            "suggested_name": self._generate_adgroup_name(cluster),
                            "impact": "medium" if len(cluster_keywords) > 10 else "low",
                        }

                        recommendations.append(suggestion)

            # Look for large ad groups that contain multiple unrelated clusters
            for ag_id, ag_data in adgroup_distributions.items():
                if len(ag_data["keywords"]) > 20 and len(ag_data["clusters"]) > 2:
                    # Check if clusters are significantly different
                    significant_clusters = [
                        (cluster_id, count)
                        for cluster_id, count in ag_data["clusters"].items()
                        if count >= 5
                    ]

                    if len(significant_clusters) >= 2:
                        # Get cluster details
                        cluster_details = []
                        for cluster_id, count in significant_clusters:
                            for cluster in cluster_result["clusters"]:
                                if cluster["id"] == cluster_id:
                                    cluster_details.append(
                                        {
                                            "cluster_id": cluster_id,
                                            "common_words": cluster["common_words"],
                                            "central_keyword": cluster["central_keyword"],
                                            "keyword_count": count,
                                            "percentage": round(
                                                count / len(ag_data["keywords"]) * 100, 1
                                            ),
                                            "suggested_name": self._generate_adgroup_name(cluster),
                                        }
                                    )
                                    break

                        if cluster_details:
                            # Suggest splitting
                            suggestion = {
                                "type": "split",
                                "ad_group_id": ag_id,
                                "ad_group_name": ag_data["name"],
                                "total_keywords": len(ag_data["keywords"]),
                                "clusters": cluster_details,
                                "recommendation": f"Split ad group '{ag_data['name']}' into {len(cluster_details)} more focused ad groups",
                                "impact": "high" if len(ag_data["keywords"]) > 50 else "medium",
                            }

                            recommendations.append(suggestion)

            # Sort by impact and limit
            impact_values = {"high": 3, "medium": 2, "low": 1}
            recommendations.sort(key=lambda x: impact_values.get(x["impact"], 0), reverse=True)
            recommendations = recommendations[:max_recommendations]

            result = {
                "status": "success",
                "recommendation_count": len(recommendations),
                "ad_group_count": len(adgroup_distributions),
                "total_keywords": sum(len(ag["keywords"]) for ag in adgroup_distributions.values()),
                "recommendations": recommendations,
                "graph_id": graph_id,
            }

            self._track_execution(start_time, True)
            return result

        except Exception as e:
            self.logger.error(f"Error optimizing ad group structure: {str(e)}")
            self._track_execution(start_time, False)
            return {"status": "error", "message": str(e)}

    def _generate_adgroup_name(self, cluster: Dict[str, Any]) -> str:
        """
        Generate a suggested ad group name based on cluster data.

        Args:
            cluster: Cluster dictionary

        Returns:
            Suggested ad group name
        """
        # Use the most central keyword if available
        if cluster.get("central_keyword"):
            return f"AG - {cluster['central_keyword'][:30]}"

        # Otherwise use common words
        if cluster.get("common_words"):
            return f"AG - {' '.join(cluster['common_words'][:3])}"

        # Fallback
        return f"Ad Group {cluster.get('id', '')}"

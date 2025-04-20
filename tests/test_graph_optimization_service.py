"""
Tests for the Graph Optimization Service.

This module contains tests for the GraphOptimizationService, which
provides graph theory algorithms for optimizing Google Ads campaigns.
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import json
import tempfile
import shutil
import networkx as nx

from services.graph_optimization_service import GraphOptimizationService


class TestGraphOptimizationService(unittest.TestCase):
    """Test cases for the Graph Optimization Service."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create mock dependencies
        self.mock_ads_api = MagicMock()
        self.mock_optimizer = MagicMock()

        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.temp_dir, "data/graph_analysis"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "reports/graph_analysis"), exist_ok=True)

        # Configuration for testing
        self.config = {
            "graph_optimization.min_weight_threshold": 0.2,
            "graph_optimization.max_graph_size": 500,
        }

        # Create service instance
        self.service = GraphOptimizationService(
            ads_api=self.mock_ads_api, optimizer=self.mock_optimizer, config=self.config
        )

        # Replace save and load methods for testing
        self.service.save_data = MagicMock(return_value=True)

        # Create mock keyword data
        self.mock_keyword_data = [
            {
                "keyword_text": "running shoes",
                "impressions": 1000,
                "clicks": 100,
                "conversions": 10,
                "cost": 150.0,
                "average_cpc": 1.5,
                "campaign_id": "123",
                "campaign_name": "Running Campaign",
                "ad_group_id": "456",
                "ad_group_name": "Shoes Ad Group",
                "match_type": "EXACT",
            },
            {
                "keyword_text": "running sneakers",
                "impressions": 800,
                "clicks": 80,
                "conversions": 8,
                "cost": 120.0,
                "average_cpc": 1.5,
                "campaign_id": "123",
                "campaign_name": "Running Campaign",
                "ad_group_id": "456",
                "ad_group_name": "Shoes Ad Group",
                "match_type": "PHRASE",
            },
            {
                "keyword_text": "jogging shoes",
                "impressions": 600,
                "clicks": 60,
                "conversions": 6,
                "cost": 90.0,
                "average_cpc": 1.5,
                "campaign_id": "123",
                "campaign_name": "Running Campaign",
                "ad_group_id": "456",
                "ad_group_name": "Shoes Ad Group",
                "match_type": "BROAD",
            },
            {
                "keyword_text": "athletic footwear",
                "impressions": 400,
                "clicks": 40,
                "conversions": 4,
                "cost": 60.0,
                "average_cpc": 1.5,
                "campaign_id": "123",
                "campaign_name": "Running Campaign",
                "ad_group_id": "789",
                "ad_group_name": "Athletic Ad Group",
                "match_type": "PHRASE",
            },
            {
                "keyword_text": "trail running shoes",
                "impressions": 200,
                "clicks": 20,
                "conversions": 2,
                "cost": 30.0,
                "average_cpc": 1.5,
                "campaign_id": "123",
                "campaign_name": "Running Campaign",
                "ad_group_id": "456",
                "ad_group_name": "Shoes Ad Group",
                "match_type": "EXACT",
            },
        ]

        # Mock the get_keyword_performance method
        self.mock_ads_api.get_keyword_performance.return_value = self.mock_keyword_data

        # Create a sample graph for testing
        self.sample_graph = nx.Graph()
        self.sample_graph.add_node("running shoes", impressions=1000)
        self.sample_graph.add_node("running sneakers", impressions=800)
        self.sample_graph.add_node("jogging shoes", impressions=600)
        self.sample_graph.add_edge("running shoes", "running sneakers", weight=0.8)
        self.sample_graph.add_edge("running shoes", "jogging shoes", weight=0.7)
        self.sample_graph.add_edge("running sneakers", "jogging shoes", weight=0.6)

    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test service initialization."""
        self.assertIsNotNone(self.service)
        self.assertEqual(self.service.ads_api, self.mock_ads_api)
        self.assertEqual(self.service.optimizer, self.mock_optimizer)
        self.assertEqual(self.service.config["graph_optimization.min_weight_threshold"], 0.2)
        self.assertEqual(self.service.config["graph_optimization.max_graph_size"], 500)

    def test_validate_config(self):
        """Test configuration validation."""
        # Test with empty config
        service = GraphOptimizationService(config={})
        self.assertIn("graph_optimization.min_weight_threshold", service.config)
        self.assertIn("graph_optimization.max_graph_size", service.config)

        # Test with partial config
        service = GraphOptimizationService(config={"graph_optimization.min_weight_threshold": 0.3})
        self.assertEqual(service.config["graph_optimization.min_weight_threshold"], 0.3)
        self.assertIn("graph_optimization.max_graph_size", service.config)

    def test_build_keyword_relationship_graph(self):
        """Test building keyword relationship graph."""
        # Call the method
        result = self.service.build_keyword_relationship_graph(days=30, min_impressions=100)

        # Check that the result has the expected structure
        self.assertEqual(result["status"], "success")
        self.assertIn("graph_id", result)
        self.assertIn("keywords_analyzed", result)
        self.assertIn("nodes", result)
        self.assertIn("edges", result)
        self.assertIn("analysis", result)
        self.assertIn("visualization_path", result)

        # Verify that the mock API was called
        self.mock_ads_api.get_keyword_performance.assert_called_once_with(days=30, campaign_id=None)

        # Verify that graph was stored
        self.assertIn(result["graph_id"], self.service.graphs)

        # Verify that save_data was called
        self.service.save_data.assert_called_once()

    def test_build_keyword_relationship_graph_no_data(self):
        """Test building keyword relationship graph with no data."""
        # Set mock API to return empty data
        self.mock_ads_api.get_keyword_performance.return_value = []

        # Call the method
        result = self.service.build_keyword_relationship_graph()

        # Check that the result indicates error
        self.assertEqual(result["status"], "error")
        self.assertIn("message", result)

    def test_create_keyword_edges(self):
        """Test creating edges between keywords."""
        # Create an empty graph
        G = nx.Graph()

        # Add nodes to the graph
        for kw in self.mock_keyword_data:
            G.add_node(kw["keyword_text"], **{k: v for k, v in kw.items() if k != "keyword_text"})

        # Call the method
        edges_added = self.service._create_keyword_edges(G, self.mock_keyword_data, 0.2)

        # Check that edges were added
        self.assertGreater(edges_added, 0)
        self.assertGreater(len(G.edges()), 0)

        # Check that edges have expected attributes
        for u, v, data in G.edges(data=True):
            self.assertIn("weight", data)
            self.assertIn("semantic_similarity", data)
            self.assertIn("performance_similarity", data)

    def test_calculate_performance_similarity(self):
        """Test calculating performance similarity between keywords."""
        # Get two keywords
        kw1 = self.mock_keyword_data[0]
        kw2 = self.mock_keyword_data[1]

        # Call the method
        similarity = self.service._calculate_performance_similarity(kw1, kw2)

        # Check that the result is a float between 0 and 1
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)

    def test_analyze_graph(self):
        """Test analyzing a graph."""
        # Call the method
        analysis = self.service._analyze_graph(self.sample_graph, "test_graph")

        # Check that the analysis has the expected structure
        self.assertIn("node_count", analysis)
        self.assertIn("edge_count", analysis)
        self.assertIn("density", analysis)
        self.assertIn("average_clustering", analysis)
        self.assertIn("connected_components", analysis)
        self.assertIn("largest_component_size", analysis)
        self.assertIn("top_degree_centrality", analysis)
        self.assertIn("top_pagerank", analysis)

        # Verify values
        self.assertEqual(analysis["node_count"], 3)
        self.assertEqual(analysis["edge_count"], 3)

    def test_get_top_nodes(self):
        """Test getting top nodes by centrality."""
        # Create a centrality dictionary
        centrality = {"node1": 0.8, "node2": 0.6, "node3": 0.4, "node4": 0.2}

        # Call the method
        top_nodes = self.service._get_top_nodes(centrality, 2)

        # Check that the result has the expected structure
        self.assertEqual(len(top_nodes), 2)
        self.assertEqual(top_nodes[0]["node"], "node1")
        self.assertEqual(top_nodes[1]["node"], "node2")

    @patch("networkx.spring_layout")
    @patch("matplotlib.pyplot.savefig")
    def test_save_graph_visualization(self, mock_savefig, mock_spring_layout):
        """Test saving graph visualization."""
        # Mock the layout function
        mock_spring_layout.return_value = {
            "running shoes": (0, 0),
            "running sneakers": (1, 0),
            "jogging shoes": (0.5, 1),
        }

        # Call the method
        viz_path = self.service._save_graph_visualization(self.sample_graph, "test_graph")

        # Check that the result is a string
        self.assertIsInstance(viz_path, str)

        # Verify that savefig was called
        mock_savefig.assert_called_once()

    def test_save_graph_data(self):
        """Test saving graph data."""
        # Call the method
        self.service._save_graph_data(self.sample_graph, "test_graph")

        # Verify that save_data was called
        self.service.save_data.assert_called_once()

    def test_identify_keyword_clusters(self):
        """Test identifying keyword clusters."""
        # Add a graph to the service
        self.service.graphs["test_graph"] = self.sample_graph

        # Call the method
        result = self.service.identify_keyword_clusters(graph_id="test_graph")

        # Check that the result has the expected structure
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["graph_id"], "test_graph")
        self.assertIn("cluster_count", result)
        self.assertIn("clusters", result)
        self.assertIn("params", result)

    def test_find_common_words(self):
        """Test finding common words in keywords."""
        # Create a set of keywords
        keywords = {"running shoes", "running sneakers", "jogging shoes"}

        # Call the method
        common_words = self.service._find_common_words(keywords)

        # Check that the result contains expected words
        self.assertIn("shoes", common_words)
        self.assertIn("running", common_words)

    def test_optimize_adgroup_structure(self):
        """Test optimizing ad group structure."""
        # Mock identify_keyword_clusters to return a predefined result
        self.service.identify_keyword_clusters = MagicMock(
            return_value={
                "status": "success",
                "graph_id": "test_graph",
                "cluster_count": 2,
                "clusters": [
                    {
                        "id": 0,
                        "size": 3,
                        "central_keyword": "running shoes",
                        "common_words": ["running", "shoes"],
                        "keywords": [
                            {"text": "running shoes", "impressions": 1000},
                            {"text": "running sneakers", "impressions": 800},
                            {"text": "jogging shoes", "impressions": 600},
                        ],
                        "avg_similarity": 0.7,
                    },
                    {
                        "id": 1,
                        "size": 2,
                        "central_keyword": "athletic footwear",
                        "common_words": ["athletic", "footwear"],
                        "keywords": [
                            {"text": "athletic footwear", "impressions": 400},
                            {"text": "sport footwear", "impressions": 300},
                        ],
                        "avg_similarity": 0.6,
                    },
                ],
                "params": {"min_cluster_size": 2, "similarity_threshold": 0.5},
            }
        )

        # Add a graph to the service with node attributes
        G = nx.Graph()
        G.add_node(
            "running shoes", ad_group_id="456", ad_group_name="Shoes Ad Group", campaign_id="123"
        )
        G.add_node(
            "running sneakers", ad_group_id="456", ad_group_name="Shoes Ad Group", campaign_id="123"
        )
        G.add_node(
            "jogging shoes", ad_group_id="456", ad_group_name="Shoes Ad Group", campaign_id="123"
        )
        G.add_node(
            "athletic footwear",
            ad_group_id="789",
            ad_group_name="Athletic Ad Group",
            campaign_id="123",
        )
        G.add_node(
            "sport footwear",
            ad_group_id="789",
            ad_group_name="Athletic Ad Group",
            campaign_id="123",
        )
        self.service.graphs["test_graph"] = G

        # Call the method
        result = self.service.optimize_adgroup_structure()

        # Check that the result has the expected structure
        self.assertEqual(result["status"], "success")
        self.assertIn("recommendation_count", result)
        self.assertIn("ad_group_count", result)
        self.assertIn("total_keywords", result)
        self.assertIn("recommendations", result)
        self.assertIn("graph_id", result)

    def test_generate_adgroup_name(self):
        """Test generating ad group name from cluster."""
        # Create a sample cluster
        cluster = {
            "central_keyword": "running shoes",
            "common_words": ["running", "shoes", "athletic"],
        }

        # Call the method
        name = self.service._generate_adgroup_name(cluster)

        # Check that the result contains the central keyword
        self.assertIn("running shoes", name)

        # Test with no central keyword
        cluster = {"common_words": ["running", "shoes", "athletic"]}

        # Call the method
        name = self.service._generate_adgroup_name(cluster)

        # Check that the result contains common words
        self.assertIn("running", name)
        self.assertIn("shoes", name)


if __name__ == "__main__":
    unittest.main()

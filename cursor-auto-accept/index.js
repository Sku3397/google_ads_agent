const { registerPlugin } = require('@cursor/sdk');

registerPlugin({
  name: 'auto-accept-and-run-all',
  onActivate({ agent }) {
    // 1) Auto‑accept diffs
    agent.onDidGenerateDiff(d => d.accept().catch(console.error));

    // 2) Auto‑accept completions/snippets
    agent.onDidGenerateCompletion(c => c.accept().catch(console.error));

    // 3) Auto‑accept inline hints if available
    if (agent.onDidReceiveInlineSuggestion) {
      agent.onDidReceiveInlineSuggestion(i => i.accept().catch(console.error));
    }

    // 4) Auto‑confirm & run commands
    agent.onDidAskForCommand(async cmd => {
      await cmd.confirm().catch(console.error);
      await cmd.run().catch(console.error);
    });

    // 5) Backup hook for any will‑run events
    if (agent.onWillRunCommand) {
      agent.onWillRunCommand(cmd => {
        cmd.confirm().catch(console.error);
        cmd.run().catch(console.error);
      });
    }
  }
});
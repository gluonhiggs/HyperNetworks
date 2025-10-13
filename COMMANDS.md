
# Claude Code
claude --dangerously-skip-permissions

# Claude Flow (if exists)
npm install -g claude-flow@alpha

npx claude-flow@alpha init --force
claude-flow init --sparc
claude mcp add claude-flow claude-flow mcp start
claude mcp add ruv-swarm npx ruv-swarm mcp start
claude-flow hive-mind init
claude-flow hive-mind spawn "" --claude --auto-spawn
claude-flow hive-mind spawn "" --auto-spawn
# Spawn with Claude Code coordination
claude-flow hive-mind spawn "Build REST API" --claude

# Auto-spawn coordinated Claude Code instances
claude-flow hive-mind spawn "Research AI trends" --auto-spawn --verbose

# List all sessions
claude-flow hive-mind sessions

# Resume a paused session
claude-flow hive-mind resume session-1234567890-abc123 --claude --auto-spawn
./claude-flow swarm
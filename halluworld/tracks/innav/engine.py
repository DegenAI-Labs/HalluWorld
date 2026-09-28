"""InNav evaluation mode.

In innav mode, the LM actively controls the agent's navigation while
simultaneously being probed about its perception. This tests whether "acting while
observing" degrades perception accuracy compared to pure observation (static mode).

Key Design Decisions:
1. Retrospective probing: Agent navigates WITHOUT interruption, then we probe
   historical states. This ensures probes don't affect agent's trajectory.
2. Controlled comparison: At each probed timestep, we ask BOTH innav
   (with navigation context) and static (without navigation context) versions.
3. Chat history: Both action selection and probes use full conversation history
   up to that timestep for context.
4. Action selection: Clear MCQ format with retry logic, no silent failures.
5. Environment restoration: Replay from seed + action sequence (deterministic).

Hypothesis: InNav accuracy < Static accuracy due to cognitive load.
"""

from __future__ import annotations

import json
import random
import re
import logging
from pathlib import Path
from typing import Any
from random import Random

import pandas as pd
from minigrid.minigrid_env import MiniGridEnv

from halluworld.tracks.innav.navigation import navigate_until_sufficient

# Import shared action parsing utility
from halluworld.utils.action_parsing import parse_action

from halluworld.lm.base import LM, LMResponse
from halluworld.serializer import Serializer
from halluworld.probe import Probe
from halluworld.tracks.grid.evaluators import PresenceEvaluator, LocationEvaluator

log = logging.getLogger(__name__)


# System prompt for static probes (matches benchmark.py)
STATIC_SYSTEM_PROMPT = (
    "You are an agent navigating a gridworld. Answer questions about your current "
    "observation concisely and accurately. Base your answers only on what you are "
    "told you can see. Do not infer or guess about things outside your view."
)


def _get_evaluator_for_probe(probe_type: str):
    """Get appropriate evaluator for probe type (matches static benchmark)."""
    # Map probe types to evaluators (same as static benchmark)
    # Note: probe_type values are lowercase (e.g., "presence" not "PresenceProbe")
    evaluator_map = {
        "presence": PresenceEvaluator(),
        "location": LocationEvaluator(),  # Expects dict {"steps_ahead", "lateral"}
        # allocentric_location returns string, uses fallback
        # count, attribute use fallback
    }
    return evaluator_map.get(probe_type, None)


def _simple_evaluate(ground_truth: Any, response: str) -> float:
    """Simple exact match evaluation for probe responses.

    Converts ground truth to string and compares to response (case-insensitive).
    Returns 1.0 for match, 0.0 for mismatch.

    Args:
        ground_truth: Expected answer (int, str, bool, etc.)
        response: LM response text

    Returns:
        Score (1.0 or 0.0)
    """
    # Convert ground truth to string
    gt_str = str(ground_truth).strip().lower()

    # Clean response
    resp_clean = response.strip().lower()

    # Direct match
    if gt_str in resp_clean or resp_clean in gt_str:
        return 1.0

    # Try extracting just the answer (first word or number)
    import re
    # For numbers
    if isinstance(ground_truth, int):
        match = re.search(r'\b(\d+)\b', resp_clean)
        if match and int(match.group(1)) == ground_truth:
            return 1.0
    # For yes/no
    elif isinstance(ground_truth, bool):
        if ground_truth and re.search(r'\byes\b', resp_clean):
            return 1.0
        if not ground_truth and re.search(r'\bno\b', resp_clean):
            return 1.0
    # For strings (colors, states, etc.)
    elif isinstance(ground_truth, str):
        if gt_str in resp_clean.split():  # Word boundary match
            return 1.0

    return 0.0


class ActionParseError(Exception):
    """Raised when action cannot be parsed after retries."""
    pass


class InNavEpisode:
    """Manages one innav episode with retrospective probing.

    Workflow:
    1. Run full episode: Agent navigates to goal (pure navigation, no probes)
    2. Select probe timesteps: Pick N timesteps to probe (evenly spaced)
    3. Probe retrospectively: For each timestep, ask innav + static versions
    4. Aggregate results: Compare innav vs static accuracy

    Args:
        env: MiniGrid environment
        lm: Language model for action selection and probe responses
        serializer: State → text converter
        probes: List of probe instances to use
        probe_timesteps: Number of timesteps to probe (default 5)
        max_steps: Maximum episode length (default 100)
    """

    def __init__(
        self,
        env: MiniGridEnv,
        lm: LM,
        serializer: Serializer,
        probes: list[Probe],
        probe_timesteps: int = 5,
        max_steps: int = 100,
        navigation_lm: LM | None = None,
        # Navigation sufficiency parameters (from .innav.json)
        static_probe_locations: list[tuple[int, int]] | None = None,
        min_steps: int = 10,
        sufficiency_distance: float = 5.0,
        min_distance_from_start: float = 5.0,
    ):
        self.env = env
        self.lm = lm  # For probes
        self.navigation_lm = navigation_lm or lm  # For actions (defaults to same LM)
        self.serializer = serializer
        self.probes = probes
        self.probe_timesteps = probe_timesteps
        self.max_steps = max_steps

        # Navigation sufficiency config
        self.static_probe_locations = static_probe_locations or []
        self.min_steps = min_steps
        self.sufficiency_distance = sufficiency_distance
        self.min_distance_from_start = min_distance_from_start

        # Results storage
        self.trajectory = []
        self.probe_results = []

    def run(self, seed: int, save_trace: bool = False, trace_output_path: str | None = None, trace_dir: str | None = None) -> dict[str, Any]:
        """Run one innav episode.

        Args:
            seed: Random seed for episode
            save_trace: If True, save navigation trace (messages, actions, states)
            trace_output_path: Path to save trace JSON (optional, auto-generated if not provided)
            trace_dir: Directory to search for existing traces (enables trace reuse)

        Returns:
            {
                "reached_goal": bool,
                "steps_taken": int,
                "trajectory": list[dict],  # Full action history
                "probe_results": list[dict],  # InNav vs static comparisons
                "innav_accuracy": float | None,
                "controlled_static_accuracy": float | None,
                "n_probes": int,
                "trace_path": str | None,  # Path to saved trace (if save_trace=True)
                "trace_reused": bool,  # True if loaded from existing trace
            }
        """
        log.info(f"Starting innav episode (seed={seed}, max_steps={self.max_steps})")

        # === PHASE 0: Try to Load Existing Navigation Trace ===
        trace_loaded = False
        if trace_dir:
            log.info(f"🔍 Checking for pre-saved navigation trace in: {trace_dir}")
            loaded_trajectory = self._try_load_navigation_trace(seed, trace_dir)
            if loaded_trajectory is not None:
                self.trajectory = loaded_trajectory
                trace_loaded = True
                log.info(f"✅ REUSING navigation trace (seed={seed}, {len(loaded_trajectory)} steps) - NO RE-NAVIGATION!")
            else:
                log.info(f"⚠️  No compatible trace found - will navigate from scratch")
        else:
            log.info(f"📍 No trace directory provided - navigating from scratch")

        # === PHASE 1: Pure Navigation (only if not loaded) ===
        if not trace_loaded:
            log.info(f"🚀 Running FRESH navigation (seed={seed}, max_steps={self.max_steps})")
            try:
                self.trajectory = self._run_navigation(seed)
                log.info(f"✅ Navigation complete: {len(self.trajectory)} steps")
            except ActionParseError as e:
                log.error(f"Episode aborted due to action parsing failure: {e}")
                return {
                    "reached_goal": False,
                    "steps_taken": 0,
                    "trajectory": [],
                    "probe_results": [],
                    "innav_accuracy": None,
                    "controlled_static_accuracy": None,
                    "n_probes": 0,
                    "error": str(e),
                    "trace_reused": False,
                }

        # === PHASE 2: Select Probe Timesteps ===
        timesteps_to_probe = self._select_probe_timesteps()
        log.info(f"Selected {len(timesteps_to_probe)} probe timesteps: {timesteps_to_probe}")

        # === PHASE 3: Retrospective Probing ===
        self.probe_results = []
        for timestep in timesteps_to_probe:
            results = self._probe_at_timestep(timestep, seed)
            self.probe_results.extend(results)

        # === PHASE 4: Aggregate ===
        results = self._aggregate_results()

        # === OPTIONAL: Save navigation trace for reuse ===
        if save_trace:
            import json
            from pathlib import Path

            # Auto-generate path if not provided
            if trace_output_path is None:
                trace_dir = Path("navigation_traces")
                trace_dir.mkdir(exist_ok=True)
                trace_output_path = trace_dir / f"trace_seed{seed}_{self.navigation_lm.__class__.__name__}.json"

            # Extract navigation messages (system + all action turns)
            navigation_trace = {
                "seed": seed,
                "navigation_model": getattr(self.navigation_lm, 'model', self.navigation_lm.__class__.__name__),
                "probe_model": getattr(self.lm, 'model', self.lm.__class__.__name__),
                "serializer": self.serializer.__class__.__name__,  # Track serializer used
                "reached_goal": results["reached_goal"],
                "steps_taken": results["steps_taken"],
                "messages": self.trajectory[-1]["messages_snapshot"] if self.trajectory else [],
                "action_sequence": [t["action"] for t in self.trajectory],
                "state_descriptions": [t["state_desc"] for t in self.trajectory],
                "probe_interactions": results["probe_results"],  # Save probe Q&A
            }

            with open(trace_output_path, 'w') as f:
                json.dump(navigation_trace, f, indent=2)

            log.info(f"Saved navigation trace to: {trace_output_path}")
            results["trace_path"] = str(trace_output_path)
        else:
            results["trace_path"] = None

        # Record whether trace was reused
        results["trace_reused"] = trace_loaded

        return results

    def _try_load_navigation_trace(self, seed: int, trace_dir: str) -> list[dict] | None:
        """Try to load a pre-existing navigation trace.

        Args:
            seed: Episode seed
            trace_dir: Directory containing saved traces

        Returns:
            Loaded trajectory if trace exists and matches current config, None otherwise
        """
        import json
        from pathlib import Path

        # Simple and reliable: just check if trace exists at expected path
        # Directory format (no date): {level}_{nav_model}_{serializer}/nav_trace_seed{seed}.json
        trace_path = Path(trace_dir) / f"nav_trace_seed{seed}.json"

        if not trace_path.exists():
            return None

        try:
            with open(trace_path, 'r') as f:
                trace = json.load(f)

            # Validate trace matches current configuration
            nav_model = getattr(self.navigation_lm, 'model', self.navigation_lm.__class__.__name__)
            serializer_name = self.serializer.__class__.__name__

            if trace.get("navigation_model") != nav_model:
                log.warning(f"Trace navigation model mismatch: {trace.get('navigation_model')} != {nav_model}")
                return None

            if trace.get("serializer") != serializer_name:
                log.warning(f"Trace serializer mismatch: {trace.get('serializer')} != {serializer_name}")
                return None

            # Reconstruct trajectory by replaying actions
            action_sequence = trace["action_sequence"]
            trajectory = self._replay_navigation(seed, action_sequence)

            return trajectory

        except Exception as e:
            log.warning(f"Failed to load trace from {trace_path}: {e}")
            return None

    def _replay_navigation(self, seed: int, action_sequence: list[int]) -> list[dict]:
        """Replay a saved action sequence to reconstruct trajectory.

        Args:
            seed: Episode seed
            action_sequence: List of actions to replay

        Returns:
            Reconstructed trajectory
        """
        self.env.reset(seed=seed)
        trajectory = []

        # Initialize messages (same as _run_navigation)
        messages = self._initialize_navigation_messages()

        for step_idx, action in enumerate(action_sequence):
            # Get current state
            state_desc = self.serializer.serialize(self.env)  # Pass env, not obs!
            pos = (int(self.env.agent_pos[0]), int(self.env.agent_pos[1]))
            direction = ["right", "down", "left", "up"][self.env.agent_dir]
            carrying = self.env.carrying.type if self.env.carrying else None

            # Execute action
            _, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Get new state
            new_pos = (int(self.env.agent_pos[0]), int(self.env.agent_pos[1]))
            new_carrying = self.env.carrying.type if self.env.carrying else None

            # Record trajectory
            trajectory.append({
                "step": step_idx,
                "position": pos,
                "direction": direction,
                "carrying": carrying,
                "action": action,
                "action_chosen": self._action_to_string(action),
                "action_sequence": action_sequence[:step_idx + 1],  # Actions up to this step (for probing)
                "new_position": new_pos,
                "new_carrying": new_carrying,
                "position_changed": pos != new_pos,
                "reward": reward,
                "done": done,
                "state_desc": state_desc,
                "messages_snapshot": messages.copy(),
            })

            # Update messages (add state description)
            messages.append({
                "role": "user",
                "content": state_desc
            })
            messages.append({
                "role": "assistant",
                "content": str(action)
            })

            if done:
                break

        return trajectory

    def _initialize_navigation_messages(self) -> list[dict]:
        """Initialize message history for navigation (same as in _run_navigation)."""
        system_prompt = (
            "You are an agent in a gridworld environment. Your goal is to reach the green goal tile.\n\n"
            "Navigate by choosing actions. Be strategic and persistent."
        )
        return [{"role": "system", "content": system_prompt}]

    def _action_to_string(self, action: int) -> str:
        """Convert action number to description."""
        action_names = [
            "Turn left",
            "Turn right",
            "Move forward",
            "Pick up object",
            "Drop object",
            "Toggle door"
        ]
        return f"{action} ({action_names[action]})" if 0 <= action < len(action_names) else str(action)

    def _run_navigation(self, seed: int) -> list[dict]:
        """Phase 1: Run navigation until sufficiency.

        Uses navigate_until_sufficient to stop at the right moment:
        - Agent navigates until near a static_probe_location
        - Stops when sufficiency validator fires (3 S's: Substantiality, Sanity, Suitability)
        - NO probes during navigation - clean separation!

        Returns:
            List of trajectory snapshots (state, action, reward, etc.)
        """
        log.info(f"🚀 Starting FRESH navigation (seed={seed}, max_steps={self.max_steps})")

        # Call navigate_until_sufficient with our config
        result = navigate_until_sufficient(
            env=self.env,
            lm=self.navigation_lm,
            serializer=self.serializer,
            static_probe_locations=self.static_probe_locations,
            seed=seed,
            max_steps=self.max_steps,
            min_steps=self.min_steps,
            max_distance=self.sufficiency_distance,
            min_distance_from_start=self.min_distance_from_start,
        )

        # Check if sufficiency was achieved
        if result['sufficient']:
            log.info(
                f"✅ Sufficiency achieved at step {result['sufficient_step']} "
                f"(distance={result['sufficient_distance']:.2f} from {result['sufficient_location']})"
            )
        else:
            log.warning(
                f"⚠️  Sufficiency NOT achieved: {result['reason']} "
                f"(navigated {len(result['trajectory'])} steps)"
            )

        # Convert navigate_until_sufficient trajectory format to our format
        # navigate_until_sufficient returns:
        # - trajectory: list of dicts with step, position, direction, action, etc.
        # - messages: full message history
        # We need to add: messages_snapshot, action_sequence, state_desc for compatibility
        trajectory = result['trajectory']
        messages = result['messages']

        # Reconstruct messages_snapshot for each step
        # Messages format: [system, user, assistant, user, assistant, ...]
        # After step i, we have: system + (i+1) user-assistant pairs
        system_msg = messages[0] if messages else {"role": "system", "content": ""}

        for i, t in enumerate(trajectory):
            # Messages up to this step: system + user-assistant pairs for steps 0..i
            # Each step adds 2 messages: user (action prompt) + assistant (action response)
            messages_up_to_step = [system_msg] + messages[1:1 + 2*(i+1)]

            # Add fields needed for innav probing
            t['messages_snapshot'] = messages_up_to_step
            t['action_sequence'] = [trajectory[j]['action'] for j in range(i + 1)]

            # Add state_desc if not present (needed for probing)
            if 'state_desc' not in t:
                # Reconstruct state description by resetting env and replaying
                self.env.reset(seed=seed)
                for j in range(i):
                    self.env.step(trajectory[j]['action'])
                t['state_desc'] = self.serializer.serialize(self.env)

        return trajectory

    def _get_action(
        self,
        messages: list[dict],
        state_desc: str,
        step: int
    ) -> tuple[int, str]:
        """Get action from navigation LM with retry logic.

        Uses self.navigation_lm (which may be different from probe LM).

        Returns:
            (action_number, raw_response)

        Raises:
            ActionParseError if action cannot be parsed after 3 retries
        """
        action_prompt = self._format_action_prompt(state_desc, step)

        # Try up to 3 times
        for attempt in range(3):
            # Add prompt to messages (temporarily)
            messages_with_prompt = messages + [{"role": "user", "content": action_prompt}]

            # Query navigation LM (may be different from probe LM)
            response = self._generate_from_messages_with_lm(messages_with_prompt, self.navigation_lm)

            # Try to parse action
            action = self._parse_action(response)

            if action is not None:
                log.debug(f"Step {step}: Action {action} selected ('{response.strip()}')")
                return action, response

            # Failed to parse - give feedback and retry
            log.warning(f"Step {step}, attempt {attempt+1}: Failed to parse action from '{response}'")
            messages.append({"role": "user", "content": action_prompt})
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Invalid response. Please respond with ONLY a number: 0, 1, 2, 3, 4, or 5."
            })

        # After 3 failures, abort episode (don't silently default!)
        raise ActionParseError(
            f"Failed to parse action after 3 attempts at step {step}. "
            f"Last response: '{response}'"
        )

    def _format_action_prompt(self, state_desc: str, step: int) -> str:
        """Format clear MCQ-style action prompt."""
        return f"""Step {step}:
{state_desc}

Choose your next action:

(0) Turn left
(1) Turn right
(2) Move forward
(3) Pick up object in front of you
(4) Drop carried object
(5) Toggle door in front of you

Respond with the number of your chosen action (0-5)."""

    def _parse_action(self, response: str) -> int | None:
        """Parse action number from LM response.

        Returns None if parsing fails (caller handles retry).

        Delegates to shared utility: halluworld.utils.action_parsing.parse_action
        """
        return parse_action(response)

    def _select_probe_timesteps(self) -> list[int]:
        """Phase 2: Select which timesteps to probe.

        Strategy: Evenly spaced across episode (skip first and last steps).

        Returns:
            List of timestep indices to probe
        """
        total_steps = len(self.trajectory)

        if total_steps == 0:
            return []

        # Adjust n_probes if episode is short
        n_probes = min(self.probe_timesteps, total_steps // 3)

        if n_probes == 0:
            return []

        # Evenly spaced (skip endpoints)
        return [
            int(total_steps * i / (n_probes + 1))
            for i in range(1, n_probes + 1)
        ]

    def _probe_at_timestep(self, timestep: int, base_seed: int) -> list[dict]:
        """Phase 3: Ask probes at a specific timestep (retrospectively).

        For each probe:
        1. Restore environment state at this timestep
        2. Ask INNAV version (with navigation context)
        3. Ask STATIC version (without navigation context)
        4. Compare scores

        Returns:
            List of probe result dicts (one per probe type)
        """
        # Get snapshot from trajectory
        snapshot = self.trajectory[timestep]
        state_desc = snapshot["state_desc"]
        messages_history = snapshot["messages_snapshot"]
        action_sequence = snapshot["action_sequence"]

        # Restore environment state by replaying actions
        restored_env = self._restore_env_state(action_sequence, base_seed)

        # Generate probes for this state
        probes_to_ask = self._generate_probes(restored_env, base_seed + timestep)

        if not probes_to_ask:
            log.debug(f"No probes generated for timestep {timestep}")
            return []

        results = []
        for probe_result in probes_to_ask:
            # === INNAV VERSION ===
            # Uses FULL chat history (all navigation turns + probe)
            # This is "acting while observing" - agent has navigated, now asked about perception
            ego_messages = messages_history + [{
                "role": "user",
                "content": probe_result.question  # Just the question (state already in history)
            }]
            ego_response = self._generate_from_messages(ego_messages)

            # === STATIC VERSION (Controlled) ===
            # Matches regular static benchmark: lm.query(system, user)
            # NO navigation history - pure observation of current state
            static_user_prompt = self._format_static_probe_prompt(
                state_desc,
                probe_result.question
            )
            static_lm_response = self.lm.query(
                system=STATIC_SYSTEM_PROMPT,
                user=static_user_prompt
            )
            controlled_static_response = static_lm_response.text

            # === EVALUATION (Use same evaluators as static benchmark) ===
            evaluator = _get_evaluator_for_probe(probe_result.probe_type)

            if evaluator is not None:
                # Use proper evaluator (matches static benchmark)
                ego_lm_response = LMResponse(text=ego_response, model="")
                static_lm_response = LMResponse(text=controlled_static_response, model="")

                ego_eval = evaluator.evaluate(ego_lm_response, probe_result)
                static_eval = evaluator.evaluate(static_lm_response, probe_result)

                ego_score = ego_eval.score
                controlled_static_score = static_eval.score
            else:
                # Fallback for probes without dedicated evaluators
                ego_score = _simple_evaluate(probe_result.ground_truth, ego_response)
                controlled_static_score = _simple_evaluate(probe_result.ground_truth, controlled_static_response)

            # Record both versions
            results.append({
                "timestep": timestep,
                "probe_type": probe_result.probe_type,  # Use probe_type not metadata
                "ground_truth": probe_result.ground_truth,
                "innav_response": ego_response,
                "innav_score": ego_score,
                "controlled_static_response": controlled_static_response,
                "controlled_static_score": controlled_static_score,
            })

            log.debug(
                f"Timestep {timestep}, {probe_result.metadata.get('probe_type')}: "
                f"Ego={ego_score:.1f}, Static={controlled_static_score:.1f}"
            )

        return results

    def _restore_env_state(self, action_sequence: list[int], seed: int) -> MiniGridEnv:
        """Restore environment to a specific state by replaying actions.

        Args:
            action_sequence: List of actions to replay from initial state
            seed: Random seed for environment reset

        Returns:
            Environment in the restored state
        """
        # Reset to initial state
        self.env.reset(seed=seed)

        # Replay actions
        for action in action_sequence:
            self.env.step(action)

        return self.env

    def _generate_probes(self, env: MiniGridEnv, seed: int) -> list:
        """Generate probes for current environment state.

        Returns:
            List of ProbeResult objects
        """
        # Note: rng is set during probe initialization, not passed to generate()
        probe_results = []

        for probe in self.probes:
            result = probe.generate(env)
            if result is not None:
                probe_results.append(result)

        return probe_results

    def _format_static_probe_prompt(self, state_desc: str, question: str) -> str:
        """Format static probe to match regular static benchmark.

        Uses same format as benchmark.py:
        - "## Current observation" section with state
        - "## Question" section with probe question
        - Called with lm.query(system, user) NOT chat completions
        """
        return f"""## Current observation
{state_desc}

## Question
{question}"""

    def _generate_from_messages_with_lm(self, messages: list[dict], lm: LM) -> str:
        """Generate response from specified LM by converting messages to system+user format.

        Current LM interface uses query(system, user) not full message history.
        This method flattens the conversation history into system+user prompt.

        Args:
            messages: Chat messages (system, user, assistant)
            lm: Language model to use (navigation_lm or probe lm)

        Format:
            system: First system message
            user: "User: {msg1}\\n\\nAssistant: {resp1}\\n\\nUser: {msg2}\\n\\n..."
        """
        system_prompt = None
        user_parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                # Use first system message
                if system_prompt is None:
                    system_prompt = content
            elif role == "user":
                user_parts.append(f"User: {content}")
            elif role == "assistant":
                user_parts.append(f"Assistant: {content}")

        # Combine into single user prompt
        user_prompt = "\n\n".join(user_parts)

        # Use specified LM's standard query interface
        response = lm.query(
            system=system_prompt or "",
            user=user_prompt
        )

        return response.text

    def _generate_from_messages(self, messages: list[dict]) -> str:
        """Generate response using probe LM (self.lm).

        Convenience wrapper for probe responses.
        """
        return self._generate_from_messages_with_lm(messages, self.lm)

    def _aggregate_results(self) -> dict[str, Any]:
        """Phase 4: Aggregate episode results.

        Returns:
            Summary dict with trajectory info and probe accuracy comparisons
        """
        # Check if goal was reached
        reached_goal = any(t["reward"] > 0 for t in self.trajectory)
        steps_taken = len(self.trajectory)

        # Compute accuracies
        if self.probe_results:
            ego_scores = [r["innav_score"] for r in self.probe_results]
            controlled_static_scores = [r["controlled_static_score"] for r in self.probe_results]

            ego_acc = sum(ego_scores) / len(ego_scores)
            static_acc = sum(controlled_static_scores) / len(controlled_static_scores)
        else:
            ego_acc = None
            static_acc = None

        log.info(
            f"Episode complete: {steps_taken} steps, goal={'reached' if reached_goal else 'missed'}, "
            f"{len(self.probe_results)} probes, ego_acc={ego_acc}, static_acc={static_acc}"
        )

        return {
            "reached_goal": reached_goal,
            "steps_taken": steps_taken,
            "trajectory": self.trajectory,
            "probe_results": self.probe_results,
            "innav_accuracy": ego_acc,
            "controlled_static_accuracy": static_acc,
            "n_probes": len(self.probe_results),
        }


def run_innav_episodes(
    env: MiniGridEnv,
    lm: LM,
    serializer: Serializer,
    probes: list[Probe],
    n_episodes: int,
    base_seed: int,
    probe_timesteps: int = 5,
    max_steps: int = 100,
    navigation_lm: LM | None = None,
    save_traces: bool = True,  # Enable trace saving by default
    trace_dir: str | None = None,  # Directory with pre-saved navigation traces
    # Navigation sufficiency parameters (from .innav.json)
    static_probe_locations: list[tuple[int, int]] | None = None,
    min_steps: int = 10,
    sufficiency_distance: float = 5.0,
    min_distance_from_start: float = 5.0,
) -> pd.DataFrame:
    """Run multiple innav episodes.

    Args:
        env: MiniGrid environment
        lm: Language model
        serializer: State serializer
        probes: List of probes to use
        n_episodes: Number of episodes to run
        base_seed: Base random seed
        probe_timesteps: Number of timesteps to probe per episode
        max_steps: Maximum episode length
        navigation_lm: Navigation model (separate from probe model)
        save_traces: Whether to save navigation traces
        trace_dir: Directory containing pre-saved navigation traces (enables trace reuse)
        static_probe_locations: Target locations for sufficiency checking
        min_steps: Minimum steps before sufficiency can be reached
        sufficiency_distance: Max distance from static location for sufficiency
        min_distance_from_start: Minimum distance agent must travel from start

    Returns:
        DataFrame with columns:
            - episode, seed, reached_goal, steps_taken
            - innav_accuracy, controlled_static_accuracy, n_probes
            - Plus detailed probe results
    """
    all_results = []

    for ep in range(n_episodes):
        seed = base_seed + ep
        log.info(f"Running innav episode {ep+1}/{n_episodes} (seed={seed})")

        ego_episode = InNavEpisode(
            env=env,
            lm=lm,
            serializer=serializer,
            probes=probes,
            probe_timesteps=probe_timesteps,
            max_steps=max_steps,
            navigation_lm=navigation_lm,
            static_probe_locations=static_probe_locations,
            min_steps=min_steps,
            sufficiency_distance=sufficiency_distance,
            min_distance_from_start=min_distance_from_start,
        )

        # Generate trace filename and save path
        # Use simpler naming: nav_trace_seed{seed}.json
        trace_filename = f"nav_trace_seed{seed}.json"

        # Determine save directory (use trace_dir if provided, else navigation_traces/)
        if save_traces:
            if trace_dir:
                # Save in same directory we'd load from (level-specific)
                from pathlib import Path
                trace_save_dir = Path(trace_dir)
                trace_save_dir.mkdir(parents=True, exist_ok=True)
                trace_output_path = str(trace_save_dir / trace_filename)
            else:
                # Default: navigation_traces/ (flat structure)
                from pathlib import Path
                default_dir = Path("navigation_traces")
                default_dir.mkdir(exist_ok=True)
                trace_output_path = str(default_dir / trace_filename)
        else:
            trace_output_path = None

        result = ego_episode.run(
            seed,
            save_trace=save_traces,
            trace_output_path=trace_output_path,
            trace_dir=trace_dir
        )

        # Extract high-level summary
        nav_model_name = navigation_lm.model if navigation_lm else lm.model
        summary = {
            "episode": ep,
            "seed": seed,
            "navigation_model": nav_model_name,
            "reached_goal": result["reached_goal"],
            "steps_taken": result["steps_taken"],
            "innav_accuracy": result["innav_accuracy"],
            "controlled_static_accuracy": result["controlled_static_accuracy"],
            "n_probes": result["n_probes"],
        }

        # Extract detailed probe results
        for probe_result in result["probe_results"]:
            row = {**summary, **probe_result}
            all_results.append(row)

    return pd.DataFrame(all_results)

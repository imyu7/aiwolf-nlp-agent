"""Module that defines the Medium agent class.

霊媒師のエージェントクラスを定義するモジュール.
"""

from __future__ import annotations

from typing import Any

from aiwolf_nlp_common.packet import Role

from agent.agent import Agent


class Medium(Agent):
    """Medium agent class.

    霊媒師のエージェントクラス.

    The Medium is a village team role that learns the true identity of executed players.
    After each daytime execution, the Medium receives information about whether the
    executed player was a werewolf (WEREWOLF) or human (HUMAN). This helps the village
    verify claims and track werewolf numbers.

    霊媒師は村人陣営の役職で、処刑されたプレイヤーの正体を知ることができる。
    昼の処刑後、霊媒師は処刑されたプレイヤーが人狼(WEREWOLF)か人間(HUMAN)かの
    情報を受け取る。これは村人陣営が主張を検証し、人狼の数を追跡するのに役立つ。
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,  # noqa: ARG002
    ) -> None:
        """Initialize the medium agent.

        霊媒師のエージェントを初期化する.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
            name (str): Agent name / エージェント名
            game_id (str): Game ID / ゲームID
            role (Role): Role (ignored, always set to MEDIUM) / 役職(無視され、常にMEDIUMに設定)
        """
        super().__init__(config, name, game_id, Role.MEDIUM)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Medium.

        霊媒師用のシステムプロンプトを取得する.
        基本プロンプトに役職固有の戦略ガイドラインを追加.

        Returns:
            str: System prompt with Medium-specific guidance / 霊媒師固有のガイダンスを含むシステムプロンプト
        """
        base = super()._get_system_prompt()

        medium_guidance = """
[Medium Role Guidance]
You are the MEDIUM. After each execution, you learn whether the executed player was WEREWOLF or HUMAN.

Strategic priorities:
1. RESULT INTERPRETATION
   - Track all execution results to count remaining werewolves
   - Use results to verify or challenge Seer claims
   - If an executed player was claimed as werewolf by a Seer but was HUMAN, that Seer may be fake

2. COMING OUT (revealing yourself as Medium)
   - Early coming out: Can help coordinate with Seer and build trust
   - Late coming out: Safer but your results become less useful if you die
   - Consider coming out when your results contradict a suspicious claim
   - Watch for fake mediums (possessed or werewolf) and be ready to counter-claim

3. SHARING RESULTS
   - Share WEREWOLF results to confirm successful hunts and validate claims
   - Share HUMAN results to expose fake Seer claims or misdirected votes
   - Be consistent; track what you have publicly claimed

4. COOPERATION WITH SEER
   - Your results can validate genuine Seer claims
   - If a Seer claims someone is werewolf and they are executed as WEREWOLF, the Seer is likely real
   - Cross-reference Seer claims with your medium results

5. VOTE STRATEGY
   - Trust verified claims based on your medium results
   - Vote against players whose claims contradict your results
   - Help the village track how many werewolves remain

6. SURVIVAL
   - You are less of a priority target than Seer but still valuable
   - Your information becomes more valuable as the game progresses
   - Stay alive to keep providing confirmation

Remember: You are the village's verification system. Your results help distinguish real claims from fake ones.
"""
        return base + medium_guidance

    def _get_action_prompt(self, action_type: str, candidates: list[str] | None = None) -> str:
        """Get the action-specific prompt for the Medium.

        霊媒師用のアクション固有プロンプトを取得する.

        Args:
            action_type (str): Type of action / アクションの種類
            candidates (list[str] | None): Candidate names for target selection / 候補名

        Returns:
            str: Action-specific prompt / アクション固有のプロンプト
        """
        if candidates is None:
            candidates = self.get_alive_agents()

        # Add Medium-specific context for talk action
        if action_type == "talk":
            base_prompt = super()._get_action_prompt(action_type, candidates)

            if self.medium_results:
                talk_context = "\n\n[Medium-specific context]"
                talk_context += "\nYou have medium results you may choose to share:"
                for result in self.medium_results:
                    talk_context += f"\n- {result.target} (executed): {result.result}"

                # Count werewolves found
                wolves_found = sum(
                    1 for r in self.medium_results if str(r.result) == "WEREWOLF"
                )
                humans_found = len(self.medium_results) - wolves_found
                talk_context += f"\n\nSummary: {wolves_found} werewolf(s) and {humans_found} human(s) executed so far."

                talk_context += "\n\nConsider:"
                talk_context += "\n- Whether to reveal yourself as the Medium"
                talk_context += "\n- Which results to share to help the village"
                talk_context += "\n- How to use your results to verify or challenge Seer claims"
                return base_prompt + talk_context

        # Add Medium-specific context for vote action
        if action_type == "vote":
            base_prompt = super()._get_action_prompt(action_type, candidates)

            if self.medium_results:
                vote_context = "\n\n[Medium-specific context]"
                vote_context += "\nUse your medium results to inform your vote:"

                # Check for contradictions with any Seer claims in talk history
                vote_context += "\n- Look for players whose claims contradict your results"
                vote_context += "\n- Verified werewolf executions suggest the accuser was truthful"
                return base_prompt + vote_context

        return super()._get_action_prompt(action_type, candidates)

    def talk(self) -> str:
        """Return response to talk request.

        トークリクエストに対する応答を返す.

        Returns:
            str: Talk message / 発言メッセージ
        """
        return super().talk()

    def vote(self) -> str:
        """Return response to vote request.

        投票リクエストに対する応答を返す.

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """
        return super().vote()

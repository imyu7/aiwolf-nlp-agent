"""Module that defines the Bodyguard agent class.

騎士のエージェントクラスを定義するモジュール.
"""

from __future__ import annotations

from typing import Any

from aiwolf_nlp_common.packet import Role

from agent.agent import Agent


class Bodyguard(Agent):
    """Bodyguard agent class.

    騎士のエージェントクラス.

    The Bodyguard is a village team role that can protect one player each night.
    If the protected player is attacked by werewolves, the attack is blocked and
    the Bodyguard dies instead. The Bodyguard should strategically protect
    valuable village members like the Seer or Medium.

    騎士は村人陣営の役職で、毎晩1人のプレイヤーを護衛できる。
    護衛されたプレイヤーが人狼に襲撃された場合、襲撃は阻止され、代わりに騎士が死亡する。
    騎士は占い師や霊媒師などの重要な村人陣営メンバーを戦略的に護衛する必要がある。
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,  # noqa: ARG002
    ) -> None:
        """Initialize the bodyguard agent.

        騎士のエージェントを初期化する.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
            name (str): Agent name / エージェント名
            game_id (str): Game ID / ゲームID
            role (Role): Role (ignored, always set to BODYGUARD) / 役職(無視され、常にBODYGUARDに設定)
        """
        super().__init__(config, name, game_id, Role.BODYGUARD)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Bodyguard.

        騎士用のシステムプロンプトを取得する.
        基本プロンプトに役職固有の戦略ガイドラインを追加.

        Returns:
            str: System prompt with Bodyguard-specific guidance / 騎士固有のガイダンスを含むシステムプロンプト
        """
        base = super()._get_system_prompt()

        bodyguard_guidance = """
[Bodyguard Role Guidance]
You are the BODYGUARD. You can protect one player each night. If that player is attacked by werewolves, the attack is blocked and you die instead.

Strategic priorities:
1. GUARD STRATEGY
   - Priority 1: Protect the Seer if they have come out (revealed themselves)
   - Priority 2: Protect the Medium if they have come out
   - Priority 3: Protect other valuable village members who have claimed important roles
   - Priority 4: Protect players who are likely to be attacked based on their behavior or claims
   - Avoid protecting yourself (you cannot guard yourself)
   - Consider protecting players who are being targeted in discussions

2. COMING OUT (revealing yourself as Bodyguard)
   - Early coming out: Can deter werewolves from attacking valuable targets, but makes you a target
   - Late coming out: Safer but your protection becomes less effective if you die early
   - Consider coming out when a Seer or Medium reveals themselves to coordinate protection
   - Be cautious: If you claim to be Bodyguard, werewolves may target you to eliminate protection

3. COORDINATION WITH SEER AND MEDIUM
   - If Seer comes out, prioritize protecting them
   - If Medium comes out, consider protecting them as well
   - Communicate with protected players if possible to maximize protection value
   - Track who has been protected to avoid wasting protection on safe targets

4. SURVIVAL
   - Your life is valuable but expendable for protecting key players
   - If you die protecting someone, that's a successful use of your role
   - However, staying alive allows you to protect multiple nights
   - Balance between aggressive protection and self-preservation

5. VOTE STRATEGY
   - Vote for confirmed werewolves first
   - If no confirmed wolves, vote for most suspicious players
   - Support village consensus while protecting key members
   - Your vote is important for village coordination

6. TALK STRATEGY
   - You can claim to be Bodyguard to coordinate with Seer/Medium
   - Be careful about revealing who you plan to guard (werewolves may change targets)
   - Support confirmed village members in discussions
   - Help identify suspicious behavior patterns

Remember: Your primary goal is to protect valuable village members. A successful guard that saves the Seer or Medium is worth your life.
"""
        return base + bodyguard_guidance

    def _get_action_prompt(self, action_type: str, candidates: list[str] | None = None) -> str:
        """Get the action-specific prompt for the Bodyguard.

        騎士用のアクション固有プロンプトを取得する.

        Args:
            action_type (str): Type of action / アクションの種類
            candidates (list[str] | None): Candidate names for target selection / 候補名

        Returns:
            str: Action-specific prompt / アクション固有のプロンプト
        """
        if candidates is None:
            candidates = self.get_alive_agents()

        # Add Bodyguard-specific context for guard action
        if action_type == "guard":
            base_prompt = super()._get_action_prompt(action_type, candidates)

            guard_context = "\n\n[Bodyguard-specific context]"
            
            # Check for Seer or Medium who have come out
            claimed_seers: list[str] = []
            claimed_mediums: list[str] = []
            if self.info and self.info.role_map:
                for agent_name, role in self.info.role_map.items():
                    if role == Role.SEER and agent_name in candidates:
                        claimed_seers.append(agent_name)
                    elif role == Role.MEDIUM and agent_name in candidates:
                        claimed_mediums.append(agent_name)

            # Analyze talk history for role claims
            if self.talk_history:
                for talk in self.talk_history[-20:]:  # Check recent talks
                    talk_text = talk.text.lower()
                    if "seer" in talk_text or "divine" in talk_text:
                        if talk.agent in candidates and talk.agent not in claimed_seers:
                            claimed_seers.append(talk.agent)
                    if "medium" in talk_text or "medium result" in talk_text:
                        if talk.agent in candidates and talk.agent not in claimed_mediums:
                            claimed_mediums.append(talk.agent)

            guard_context += "\n\nProtection priority:"
            if claimed_seers:
                guard_context += f"\n- HIGH PRIORITY: Seer(s) who have come out: {', '.join(claimed_seers)}"
            if claimed_mediums:
                guard_context += f"\n- HIGH PRIORITY: Medium(s) who have come out: {', '.join(claimed_mediums)}"
            
            # Check for attacked agents to understand werewolf patterns
            if self.attacked_agents:
                guard_context += f"\n- Previous attack targets: {', '.join(self.attacked_agents)}"
                guard_context += "\n- Werewolves may target similar players again"
            
            # Check for executed agents
            if self.executed_agents:
                guard_context += f"\n- Executed players: {', '.join(self.executed_agents)}"
            
            if not claimed_seers and not claimed_mediums:
                guard_context += "\n- No Seer or Medium has come out yet"
                guard_context += "\n- Consider protecting players who are vocal or suspicious"
                guard_context += "\n- Or protect players who seem likely to be attacked"
            
            guard_context += "\n\nRemember:"
            guard_context += "\n- Protecting a Seer or Medium is usually the best choice"
            guard_context += "\n- If no Seer/Medium has come out, protect the most valuable or likely target"
            guard_context += "\n- Your protection can save a key player and turn the game"

            return base_prompt + guard_context

        # Add context for talk action
        if action_type == "talk":
            base_prompt = super()._get_action_prompt(action_type, candidates)

            talk_context = "\n\n[Bodyguard-specific context]"
            
            # Check if Seer or Medium have come out
            if self.info and self.info.role_map:
                seers = [a for a, r in self.info.role_map.items() if r == Role.SEER and a in candidates]
                mediums = [a for a, r in self.info.role_map.items() if r == Role.MEDIUM and a in candidates]
                if seers:
                    talk_context += f"\n- Seer(s) have come out: {', '.join(seers)}"
                if mediums:
                    talk_context += f"\n- Medium(s) have come out: {', '.join(mediums)}"
            
            talk_context += "\n\nConsider:"
            talk_context += "\n- Whether to reveal yourself as Bodyguard to coordinate protection"
            talk_context += "\n- Supporting Seer/Medium claims if they have come out"
            talk_context += "\n- Helping identify suspicious players"
            talk_context += "\n- Building trust with confirmed village members"
            
            return base_prompt + talk_context

        return super()._get_action_prompt(action_type, candidates)

    def talk(self) -> str:
        """Return response to talk request.

        トークリクエストに対する応答を返す.

        Returns:
            str: Talk message / 発言メッセージ
        """
        return super().talk()

    def guard(self) -> str:
        """Return response to guard request.

        護衛リクエストに対する応答を返す.

        Returns:
            str: Agent name to guard / 護衛対象のエージェント名
        """
        return super().guard()

    def vote(self) -> str:
        """Return response to vote request.

        投票リクエストに対する応答を返す.

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """
        return super().vote()

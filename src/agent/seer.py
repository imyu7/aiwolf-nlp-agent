"""Module that defines the Seer agent class.

占い師のエージェントクラスを定義するモジュール.
"""

from __future__ import annotations

from typing import Any

from aiwolf_nlp_common.packet import Role

from agent.agent import Agent


class Seer(Agent):
    """Seer agent class.

    占い師のエージェントクラス.

    The Seer is a key role for the village team. Each night, the Seer can divine
    one player to learn whether they are a werewolf (WEREWOLF) or human (HUMAN).
    The Seer should strategically share or withhold this information to help
    the village team identify and eliminate werewolves.

    占い師は村人陣営の重要な役職。毎晩1人のプレイヤーを占い、
    そのプレイヤーが人狼(WEREWOLF)か人間(HUMAN)かを知ることができる。
    占い師は戦略的にこの情報を共有または秘匿し、村人陣営が人狼を
    特定・追放するのを助ける必要がある。
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,  # noqa: ARG002
    ) -> None:
        """Initialize the seer agent.

        占い師のエージェントを初期化する.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
            name (str): Agent name / エージェント名
            game_id (str): Game ID / ゲームID
            role (Role): Role (ignored, always set to SEER) / 役職(無視され、常にSEERに設定)
        """
        super().__init__(config, name, game_id, Role.SEER)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Seer.

        占い師用のシステムプロンプトを取得する.
        基本プロンプトに役職固有の戦略ガイドラインを追加.

        Returns:
            str: System prompt with Seer-specific guidance / 占い師固有のガイダンスを含むシステムプロンプト
        """
        base = super()._get_system_prompt()

        seer_guidance = """
[Seer Role Guidance]
You are the SEER. You have the critical ability to divine players each night to learn if they are WEREWOLF or HUMAN.

Strategic priorities:
1. DIVINE STRATEGY
   - Prioritize divining players who are most suspicious or vocal
   - Avoid divining players who are likely to be executed anyway
   - Consider divining quiet players who might be hiding their role
   - Track who you have already divined to avoid wasting actions

2. COMING OUT (revealing yourself as Seer)
   - Early coming out: Risky but can establish trust and coordinate votes
   - Late coming out: Safer but may lose credibility or die before sharing info
   - Consider coming out when you find a werewolf to build consensus for voting
   - Watch for fake seers (possessed or werewolf) and be ready to counter-claim

3. SHARING RESULTS
   - Share WEREWOLF results strategically to lead votes
   - HUMAN results can clear trusted allies but also reveal your investigation pattern
   - Be consistent with your claims; contradictions will expose you as fake

4. SURVIVAL
   - Once revealed, you are a prime target for werewolves
   - Build alliances with confirmed humans
   - If there is a bodyguard, they may protect you

5. VOTE STRATEGY
   - Vote for confirmed werewolves first
   - If no confirmed wolves, vote for most suspicious player
   - Coordinate with other villagers based on your intel

Remember: You are the village's most important information source. Use your power wisely.
"""
        return base + seer_guidance

    def _get_action_prompt(self, action_type: str, candidates: list[str] | None = None) -> str:
        """Get the action-specific prompt for the Seer.

        占い師用のアクション固有プロンプトを取得する.

        Args:
            action_type (str): Type of action / アクションの種類
            candidates (list[str] | None): Candidate names for target selection / 候補名

        Returns:
            str: Action-specific prompt / アクション固有のプロンプト
        """
        if candidates is None:
            candidates = self.get_alive_agents()

        # Add Seer-specific context for divine action
        if action_type == "divine":
            # Build info about already divined players
            already_divined = {r.target for r in self.divine_results}
            not_divined = [c for c in candidates if c not in already_divined]

            base_prompt = super()._get_action_prompt(action_type, candidates)

            divine_context = "\n\n[Seer-specific context]"
            if self.divine_results:
                divine_context += "\nYour past divination results:"
                for result in self.divine_results:
                    divine_context += f"\n- {result.target}: {result.result}"

            if not_divined:
                divine_context += f"\n\nPlayers you have NOT divined yet: {', '.join(not_divined)}"
                divine_context += "\nPrioritize divining players you haven't checked yet."
            else:
                divine_context += "\nYou have divined all remaining candidates."

            return base_prompt + divine_context

        # Add context for talk action
        if action_type == "talk":
            base_prompt = super()._get_action_prompt(action_type, candidates)

            if self.divine_results:
                talk_context = "\n\n[Seer-specific context]"
                talk_context += "\nYou have divination results you may choose to share:"
                for result in self.divine_results:
                    talk_context += f"\n- {result.target}: {result.result}"
                talk_context += "\n\nConsider:"
                talk_context += "\n- Whether to reveal yourself as the Seer"
                talk_context += "\n- Which results to share and when"
                talk_context += "\n- How to build credibility if challenged"
                return base_prompt + talk_context

        return super()._get_action_prompt(action_type, candidates)

    def talk(self) -> str:
        """Return response to talk request.

        トークリクエストに対する応答を返す.

        Returns:
            str: Talk message / 発言メッセージ
        """
        return super().talk()

    def divine(self) -> str:
        """Return response to divine request.

        占いリクエストに対する応答を返す.

        Returns:
            str: Agent name to divine / 占い対象のエージェント名
        """
        return super().divine()

    def vote(self) -> str:
        """Return response to vote request.

        投票リクエストに対する応答を返す.

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """
        return super().vote()

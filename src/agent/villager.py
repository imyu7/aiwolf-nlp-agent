"""Module that defines the Villager agent class.

村人のエージェントクラスを定義するモジュール.
"""

from __future__ import annotations

from typing import Any

from aiwolf_nlp_common.packet import Role

from agent.agent import Agent


class Villager(Agent):
    """Villager agent class.

    村人のエージェントクラス.

    The Villager is a basic village team role with no special abilities.
    Villagers must rely on observation, logic, and cooperation to identify
    and eliminate werewolves. They should support confirmed village members
    and help coordinate votes.

    村人は特殊能力を持たない基本的な村人陣営の役職。
    村人は観察、論理、協力に頼って人狼を特定し、排除する必要がある。
    確認された村人陣営メンバーを支援し、投票を調整する必要がある。
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,  # noqa: ARG002
    ) -> None:
        """Initialize the villager agent.

        村人のエージェントを初期化する.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
            name (str): Agent name / エージェント名
            game_id (str): Game ID / ゲームID
            role (Role): Role (ignored, always set to VILLAGER) / 役職(無視され、常にVILLAGERに設定)
        """
        super().__init__(config, name, game_id, Role.VILLAGER)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Villager.

        村人用のシステムプロンプトを取得する.
        基本プロンプトに役職固有の戦略ガイドラインを追加.

        Returns:
            str: System prompt with Villager-specific guidance / 村人固有のガイダンスを含むシステムプロンプト
        """
        base = super()._get_system_prompt()

        villager_guidance = """
[Villager Role Guidance]
You are a VILLAGER. You have no special abilities, but you are a crucial part of the village team.
Your role is to observe, analyze, and help the village identify and eliminate werewolves through voting.

Strategic priorities:
1. OBSERVATION AND ANALYSIS
   - Pay close attention to all public statements and voting patterns
   - Track who claims what roles and verify consistency
   - Identify contradictions in claims and behavior
   - Notice who is being quiet or avoiding certain topics
   - Analyze voting patterns to find suspicious alignments

2. SUPPORTING CONFIRMED VILLAGE MEMBERS
   - Trust and support the Seer if they have come out with results
   - Trust and support the Medium if they have come out with results
   - Coordinate with Bodyguard if they reveal themselves
   - Follow the lead of confirmed village members when they identify werewolves

3. IDENTIFYING SUSPICIOUS BEHAVIOR
   - Players who avoid answering direct questions
   - Players who change their stories or contradict themselves
   - Players who vote inconsistently with their stated beliefs
   - Players who seem to be coordinating secretly (possible werewolf team)
   - Players who are too eager to vote without evidence

4. TALK STRATEGY
   - Ask direct questions to gather information
   - Share your observations and suspicions clearly
   - Support confirmed village members in their claims
   - Challenge suspicious claims or contradictions
   - Help build consensus around voting targets
   - Avoid being too aggressive (you don't want to look suspicious yourself)

5. VOTE STRATEGY
   - Vote for confirmed werewolves first (if Seer/Medium have identified any)
   - Vote for the most suspicious player if no confirmed wolves
   - Coordinate with other villagers to ensure werewolves are eliminated
   - Consider voting patterns: if someone consistently votes with suspected werewolves, they may be one too
   - Don't waste votes on players who are likely to be executed anyway

6. SURVIVAL
   - Stay alive to continue contributing to village discussions
   - Avoid drawing unnecessary attention to yourself
   - Build trust through consistent, logical behavior
   - Don't make wild accusations without evidence

7. FAKE ROLE CLAIMS
   - Be cautious of players claiming to be Seer or Medium
   - Verify claims by checking if results make sense
   - If multiple players claim the same role, at least one is lying
   - Support the claim that seems most credible based on evidence

8. COOPERATION
   - Work with other villagers to build consensus
   - Share information you've gathered through observation
   - Help coordinate votes to eliminate werewolves
   - Support logical arguments and challenge illogical ones

Remember: As a Villager, you are the foundation of the village team. Your careful observation and logical thinking are essential for victory. Stay active, ask questions, and help coordinate the village's efforts.
"""
        return base + villager_guidance

    def _get_action_prompt(self, action_type: str, candidates: list[str] | None = None) -> str:
        """Get the action-specific prompt for the Villager.

        村人用のアクション固有プロンプトを取得する.

        Args:
            action_type (str): Type of action / アクションの種類
            candidates (list[str] | None): Candidate names for target selection / 候補名

        Returns:
            str: Action-specific prompt / アクション固有のプロンプト
        """
        if candidates is None:
            candidates = self.get_alive_agents()

        # Add Villager-specific context for vote action
        if action_type == "vote":
            base_prompt = super()._get_action_prompt(action_type, candidates)

            vote_context = "\n\n[Villager-specific context]"
            
            # Check for confirmed werewolves from Seer/Medium
            confirmed_wolves: list[str] = []
            if self.info and self.info.role_map:
                for agent_name, role in self.info.role_map.items():
                    if role == Role.WEREWOLF and agent_name in candidates:
                        confirmed_wolves.append(agent_name)

            # Check divine results for werewolves
            if self.divine_results:
                for result in self.divine_results:
                    if str(result.result) == "WEREWOLF" and result.target in candidates:
                        if result.target not in confirmed_wolves:
                            confirmed_wolves.append(result.target)

            # Check medium results for werewolves
            if self.medium_results:
                for result in self.medium_results:
                    if str(result.result) == "WEREWOLF" and result.target in candidates:
                        if result.target not in confirmed_wolves:
                            confirmed_wolves.append(result.target)

            vote_context += "\n\nVoting priority:"
            if confirmed_wolves:
                vote_context += f"\n- HIGH PRIORITY: Confirmed werewolves: {', '.join(confirmed_wolves)}"
                vote_context += "\n- Vote for one of these confirmed werewolves"
            else:
                vote_context += "\n- No confirmed werewolves yet"
                vote_context += "\n- Analyze talk history and voting patterns to identify suspicious players"
                
                # Analyze talk history for suspicious behavior
                if self.talk_history:
                    vote_context += "\n\nConsider these factors:"
                    vote_context += "\n- Who has been avoiding questions or giving vague answers?"
                    vote_context += "\n- Who has contradicted themselves or changed their story?"
                    vote_context += "\n- Who has been voting suspiciously or inconsistently?"
                    vote_context += "\n- Who seems to be coordinating with suspected werewolves?"

            # Check vote history for patterns
            if self.vote_history:
                vote_context += "\n\nPrevious vote patterns:"
                for day_votes in self.vote_history[-2:]:  # Last 2 days
                    vote_counts: dict[str, int] = {}
                    for vote in day_votes:
                        vote_counts[vote.target] = vote_counts.get(vote.target, 0) + 1
                    if vote_counts:
                        vote_context += f"\n- Day {day_votes[0].day if day_votes else '?'} votes: {vote_counts}"

            vote_context += "\n\nRemember:"
            vote_context += "\n- Vote for confirmed werewolves first"
            vote_context += "\n- If no confirmed wolves, vote for the most suspicious player"
            vote_context += "\n- Coordinate with other villagers to eliminate werewolves"

            return base_prompt + vote_context

        # Add context for talk action
        if action_type == "talk":
            base_prompt = super()._get_action_prompt(action_type, candidates)

            talk_context = "\n\n[Villager-specific context]"
            
            # Check for confirmed roles
            if self.info and self.info.role_map:
                seers = [a for a, r in self.info.role_map.items() if r == Role.SEER and a in candidates]
                mediums = [a for a, r in self.info.role_map.items() if r == Role.MEDIUM and a in candidates]
                wolves = [a for a, r in self.info.role_map.items() if r == Role.WEREWOLF and a in candidates]
                
                if seers:
                    talk_context += f"\n- Seer(s) have come out: {', '.join(seers)}"
                if mediums:
                    talk_context += f"\n- Medium(s) have come out: {', '.join(mediums)}"
                if wolves:
                    talk_context += f"\n- Confirmed werewolf(ves): {', '.join(wolves)}"
            
            # Check for divine/medium results
            if self.divine_results:
                talk_context += "\n- Divine results available:"
                for result in self.divine_results:
                    talk_context += f"\n  - {result.target}: {result.result}"
            
            if self.medium_results:
                talk_context += "\n- Medium results available:"
                for result in self.medium_results:
                    talk_context += f"\n  - {result.target}: {result.result}"
            
            talk_context += "\n\nConsider:"
            talk_context += "\n- Asking direct questions to gather information"
            talk_context += "\n- Sharing your observations about suspicious behavior"
            talk_context += "\n- Supporting confirmed village members (Seer/Medium)"
            talk_context += "\n- Challenging suspicious claims or contradictions"
            talk_context += "\n- Helping build consensus around voting targets"
            talk_context += "\n- Staying active but not drawing unnecessary attention"
            
            return base_prompt + talk_context

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

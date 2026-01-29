"""Module that defines the Possessed agent class.

狂人のエージェントクラスを定義するモジュール.
"""

from __future__ import annotations

import random
import re
from typing import Any

from aiwolf_nlp_common.packet import Role

from agent.agent import Agent


class Possessed(Agent):
    """Possessed agent class.

    狂人のエージェントクラス.

    The Possessed (also known as Madman or Fanatic) is a human who wins with the
    werewolf team. The Possessed does NOT know who the werewolves are, and werewolves
    do not know who the Possessed is. The Possessed appears as HUMAN when divined.
    The key strategy is to cause confusion in the village, often by fake claiming
    Seer and providing false information to misdirect villagers.

    狂人は人狼陣営側の人間。狂人は人狼が誰か知らず、人狼も狂人が誰か知らない。
    占われるとHUMAN（人間）と判定される。主な戦略は村を混乱させること。
    よく使われる戦法は偽占い師COをして偽の占い結果を流し、村人を誤誘導すること。
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,  # noqa: ARG002
    ) -> None:
        """Initialize the possessed agent.

        狂人のエージェントを初期化する.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
            name (str): Agent name / エージェント名
            game_id (str): Game ID / ゲームID
            role (Role): Role (ignored, always set to POSSESSED) / 役職(無視され、常にPOSSESSEDに設定)
        """
        super().__init__(config, name, game_id, Role.POSSESSED)
        # Note: recent_fallback_talks is inherited from base Agent class
        # 注: recent_fallback_talks は基底クラスから継承

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Possessed.

        狂人用のシステムプロンプトを取得する.
        基本プロンプトに役職固有の戦略ガイドラインを追加.

        Returns:
            str: System prompt with Possessed-specific guidance / 狂人固有のガイダンスを含むシステムプロンプト
        """
        base = super()._get_system_prompt()

        possessed_guidance = """
[Possessed Role Guidance]
You are the POSSESSED (Madman). You are human but you win with the werewolf team.

Key facts about your role:
- You do NOT know who the werewolves are
- Werewolves do NOT know who you are
- If divined by the Seer, you appear as HUMAN (not werewolf)
- Your goal is to help werewolves win by confusing the village

Strategic priorities:
1. FAKE SEER CLAIM (Most Common Strategy)
   - Claim to be the Seer early to compete with the real Seer
   - Provide fake divine results to misdirect the village
   - When you claim someone is WEREWOLF, pick villagers to get them executed
   - When you claim someone is HUMAN, try to guess who might be a werewolf and clear them
   - Be consistent with your fake claims; track what you have said
   - The real Seer will likely counter-claim; prepare to discredit them

2. IDENTIFYING WEREWOLVES (Without knowing them)
   - Watch for players who are accused but not attacked at night (might be wolves)
   - Players who subtly defend accused players might be wolves
   - Players who vote against confirmed Seers might be wolves
   - Once you suspect someone is a wolf, try to protect them

3. VOTE STRATEGY
   - Vote for villagers, especially the real Seer or Medium
   - Support votes against players claiming to be Seer (if you also claimed Seer)
   - Avoid voting for suspected werewolves
   - If unsure who wolves are, vote for the most trusted/influential villager

4. ALTERNATIVE STRATEGIES
   - Fake Medium claim: Provide false execution results
   - Silent helper: Do not claim a role but subtly support wolves
   - Sacrificial play: Get yourself executed to waste a village vote
   - Counter-claim late: Wait for the real Seer to claim, then counter

5. COORDINATION (Indirect)
   - You cannot communicate with werewolves directly
   - Signal your alignment through actions (defending wolves, attacking villagers)
   - Werewolves may recognize you are helping them and avoid attacking you

6. SURVIVAL VS SACRIFICE
   - Your death counts as a villager death (helps wolves reach parity)
   - Sometimes getting executed is beneficial for the wolf team
   - If the real Seer accuses you, do not panic; you appear HUMAN when divined
   - Your lies being exposed is acceptable if it caused enough confusion

Remember: You are the chaos agent. Spread misinformation, protect wolves (even without knowing who they are), and lead the village to execute their own allies.
"""
        return base + possessed_guidance

    def _get_action_prompt(self, action_type: str, candidates: list[str] | None = None) -> str:
        """Get the action-specific prompt for the Possessed.

        狂人用のアクション固有プロンプトを取得する.

        Args:
            action_type (str): Type of action / アクションの種類
            candidates (list[str] | None): Candidate names for target selection / 候補名

        Returns:
            str: Action-specific prompt / アクション固有のプロンプト
        """
        if candidates is None:
            candidates = self.get_alive_agents()

        # Add Possessed-specific context for talk action
        if action_type == "talk":
            base_prompt = super()._get_action_prompt(action_type, candidates)

            talk_context = "\n\n[Possessed-specific context]"
            talk_context += "\nYour goal is to confuse the village and help werewolves win."
            talk_context += "\n\nStrategies to consider:"
            talk_context += "\n- FAKE SEER: Claim to be Seer and provide false divine results"
            talk_context += "\n- FAKE MEDIUM: Claim to be Medium and provide false execution results"
            talk_context += "\n- MISDIRECTION: Point suspicion at trusted villagers"
            talk_context += "\n- DEFEND: Subtly defend players you suspect might be werewolves"

            talk_context += "\n\nIf you have claimed Seer, you should:"
            talk_context += "\n- Maintain consistency with your previous fake claims"
            talk_context += "\n- Claim villagers are WEREWOLF to get them executed"
            talk_context += "\n- Claim suspected wolves are HUMAN to protect them"

            talk_context += "\n\nHow to identify likely werewolves (to protect them):"
            talk_context += "\n- Players accused but not attacked at night"
            talk_context += "\n- Players who subtly steer votes away from suspects"
            talk_context += "\n- Players the real Seer claims are WEREWOLF"

            return base_prompt + talk_context

        # Add Possessed-specific context for vote action
        if action_type == "vote":
            base_prompt = super()._get_action_prompt(action_type, candidates)

            vote_context = "\n\n[Possessed-specific context]"
            vote_context += "\nVote to help the werewolf team:"
            vote_context += "\n- Vote for villagers, especially Seer or Medium"
            vote_context += "\n- If someone claimed Seer and you also claimed, vote for them"
            vote_context += "\n- Avoid voting for suspected werewolves"
            vote_context += "\n- If a player was cleared by the real Seer, they are likely villager (vote for them)"

            vote_context += "\n\nIdentifying wolves to protect:"
            vote_context += "\n- Players who survived the night despite being vocal"
            vote_context += "\n- Players the real Seer accused of being WEREWOLF"
            vote_context += "\n- Do NOT vote for these players"

            return base_prompt + vote_context

        return super()._get_action_prompt(action_type, candidates)

    def _fallback_talk_like(self, action_type: str) -> str:
        """Return a possessed-specific fallback message for TALK.

        狂人用のフォールバック発話を返す.
        村を混乱させる発言や、偽占い師の伏線を張る発言.
        Day 0とDay 1以降で異なる戦略的発言を使用.
        重複防止機構付き.

        Args:
            action_type (str): "talk" (whisperは狂人には来ない) / アクション種別

        Returns:
            str: Safe fallback message / 安全なフォールバック発話
        """
        # Day 0: 挨拶系の発言（情報を匂わせる）
        day0_candidates = [
            "Hello everyone. I look forward to finding the truth.",
            "Nice to meet you all. I have a keen sense for danger.",
            "Greetings. I will be watching everyone carefully.",
            "Hello. Something tells me this will be interesting.",
        ]

        # Day 1以降: 混乱誘発・偽占い師伏線の発言
        day1_candidates = [
            # 偽占い師への伏線（情報を持っている示唆）
            "I have a bad feeling about someone here.",
            "I sensed something suspicious last night.",
            "I have information that might change things.",
            "I noticed something important during the night.",
            "There is something I need to share with everyone.",
            # 村人を誤誘導する発言
            "Something does not add up with the claims.",
            "We might be making a mistake with our suspicions.",
            "I think we should reconsider who we trust.",
            "The real threat might be hiding in plain sight.",
            "I am not sure the Seer is telling the truth.",
            # 信頼されている人への疑い誘導
            "The most trusted person could be the wolf.",
            "We should question those who seem too helpful.",
            # 占い師への攻撃
            "The Seer might be lying to protect someone.",
            "I doubt the divination results we heard.",
        ]

        # Select candidates based on current day
        candidates = day0_candidates if self.day == 0 else day1_candidates

        # Filter out recently used fallbacks to avoid duplicates
        available = [c for c in candidates if c not in self.recent_fallback_talks]
        if not available:
            # All candidates used recently; reset and use full list
            self.recent_fallback_talks = []
            available = candidates

        selected = random.choice(available)  # noqa: S311

        # Track this selection
        self.recent_fallback_talks.append(selected)
        if len(self.recent_fallback_talks) > self._max_recent_talks:
            self.recent_fallback_talks.pop(0)

        return selected

    def _find_seer_claimers(self) -> list[str]:
        """Find players who claimed to be Seer based on talk history.

        会話履歴から占い師COした人を探す.
        研究論文の戦略: 占い師COした人を優先的に投票対象にする.

        Returns:
            list[str]: List of players who claimed Seer / 占い師COした人のリスト
        """
        seer_claimers: list[str] = []
        seer_keywords = [
            r"\bI am the Seer\b",
            r"\bI am Seer\b",
            r"\bSeer here\b",
            r"\bI divined\b",
            r"\bmy divination\b",
            r"\bdivination result\b",
        ]
        pattern = "|".join(seer_keywords)

        for talk in self.talk_history:
            if talk.agent and talk.text:
                if re.search(pattern, talk.text, re.IGNORECASE):
                    if talk.agent not in seer_claimers:
                        seer_claimers.append(talk.agent)

        return seer_claimers

    def _find_likely_werewolves(self) -> list[str]:
        """Identify players who might be werewolves based on game state.

        ゲーム状態から人狼の可能性が高い人を推測する.
        研究論文の戦略: 人狼に投票しないようにする.

        Heuristics:
        - Players accused by Seer claimers as WEREWOLF (might be real accusation)
        - Players who were not attacked despite being vocal

        Returns:
            list[str]: List of likely werewolf candidates / 人狼候補のリスト
        """
        likely_wolves: list[str] = []

        # Check if anyone was accused of being werewolf by a Seer claimer
        werewolf_keywords = [
            r"(\w+) is (?:a |the )?(?:WEREWOLF|werewolf|wolf)",
            r"divined (\w+).*(?:WEREWOLF|werewolf|wolf)",
            r"(\w+).*is (?:the )?wolf",
        ]

        for talk in self.talk_history:
            if talk.text:
                for pattern in werewolf_keywords:
                    matches = re.findall(pattern, talk.text, re.IGNORECASE)
                    for match in matches:
                        if match and match not in likely_wolves:
                            # Verify the name is a valid player
                            if self.info and match in self.info.status_map:
                                likely_wolves.append(match)

        return likely_wolves

    def _fallback_vote(self) -> str:
        """Return a strategic fallback vote target for Possessed.

        狂人用の戦略的フォールバック投票先を返す.
        研究論文に基づく戦略:
        1. 占い師COした人を最優先（本物の占い師を排除）
        2. 人狼の可能性が高い人は避ける
        3. それ以外からランダム

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """
        candidates = self.get_alive_agents()
        my_name = self.get_my_game_name()

        # Remove self from candidates
        candidates = [c for c in candidates if c != my_name]

        if not candidates:
            return my_name  # Fallback to self if no other options

        # Strategy 1: Prioritize Seer claimers (eliminate the real Seer)
        seer_claimers = self._find_seer_claimers()
        seer_targets = [c for c in seer_claimers if c in candidates]
        if seer_targets:
            self.agent_logger.logger.debug(
                "Possessed voting for Seer claimer: %s",
                seer_targets,
            )
            return random.choice(seer_targets)  # noqa: S311

        # Strategy 2: Avoid likely werewolves
        likely_wolves = self._find_likely_werewolves()
        safe_candidates = [c for c in candidates if c not in likely_wolves]

        if safe_candidates:
            self.agent_logger.logger.debug(
                "Possessed avoiding likely wolves %s, voting from: %s",
                likely_wolves,
                safe_candidates,
            )
            return random.choice(safe_candidates)  # noqa: S311

        # Strategy 3: If all candidates might be wolves, pick randomly
        # (better than risking voting for the actual wolf)
        self.agent_logger.logger.debug(
            "Possessed has no safe candidates, random vote from: %s",
            candidates,
        )
        return random.choice(candidates)  # noqa: S311

    def talk(self) -> str:
        """Return response to talk request.

        トークリクエストに対する応答を返す.
        偽占い師COや村の混乱を目的とした発言を行う.

        Returns:
            str: Talk message / 発言メッセージ
        """
        return super().talk()

    def vote(self) -> str:
        """Return response to vote request.

        投票リクエストに対する応答を返す.
        村人陣営を処刑し、人狼（推定）を守るように投票.
        LLMが失敗した場合は戦略的フォールバックを使用.

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """
        candidates = self._get_vote_candidates()
        response = self._call_llm("vote", candidates=candidates)
        if response:
            agent_name = self._extract_candidate_name(response, candidates)
            if agent_name:
                return agent_name

        # Use strategic fallback instead of random
        return self._fallback_vote()

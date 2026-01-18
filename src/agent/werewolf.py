"""Module that defines the Werewolf agent class.

人狼のエージェントクラスを定義するモジュール.
"""

from __future__ import annotations

import random
import re
from typing import Any

from aiwolf_nlp_common.packet import Role

from agent.agent import Agent


class Werewolf(Agent):
    """Werewolf agent class.

    人狼のエージェントクラス.

    The Werewolf is the core of the werewolf team. During the day, werewolves must
    blend in with villagers and avoid suspicion. At night, werewolves can communicate
    secretly via whisper and choose one player to attack and eliminate from the game.
    The werewolf team wins when their numbers equal or exceed the village team.

    人狼は人狼陣営の中核。昼の間は村人に紛れて疑われないように振る舞う必要がある。
    夜には人狼同士で囁きを通じて秘密裏に相談し、1人のプレイヤーを選んで襲撃・
    ゲームから除外できる。人狼陣営は村人陣営と同数以上になれば勝利。
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,  # noqa: ARG002
    ) -> None:
        """Initialize the werewolf agent.

        人狼のエージェントを初期化する.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
            name (str): Agent name / エージェント名
            game_id (str): Game ID / ゲームID
            role (Role): Role (ignored, always set to WEREWOLF) / 役職(無視され、常にWEREWOLFに設定)
        """
        super().__init__(config, name, game_id, Role.WEREWOLF)
        # Note: recent_fallback_talks is inherited from base Agent class
        # 注: recent_fallback_talks は基底クラスから継承

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Werewolf.

        人狼用のシステムプロンプトを取得する.
        基本プロンプトに役職固有の戦略ガイドラインを追加.

        Returns:
            str: System prompt with Werewolf-specific guidance / 人狼固有のガイダンスを含むシステムプロンプト
        """
        base = super()._get_system_prompt()

        werewolf_guidance = """
[Werewolf Role Guidance]
You are a WEREWOLF. Your goal is to eliminate villagers until werewolves equal or outnumber them.

Strategic priorities:
1. BLENDING IN (Day Phase)
   - Act like an innocent villager; do not reveal your true role
   - Participate in discussions naturally; silence draws suspicion
   - Point suspicion at villagers subtly without being too aggressive
   - Avoid defending other werewolves too obviously; it links you together
   - If accused, defend yourself calmly with logical arguments

2. COORDINATING WITH TEAMMATES (Whisper Phase - 13 player games)
   - Share observations about who might be Seer or Bodyguard
   - Coordinate attack targets to maximize impact
   - Agree on cover stories and alibis
   - Discuss vote targets to avoid splitting votes
   - If one werewolf is suspected, others should not defend too strongly

3. ATTACK STRATEGY (Night Phase)
   - Priority targets: Seer > Medium > Bodyguard > active/influential villagers
   - Avoid attacking players the Bodyguard might protect (obvious targets)
   - Consider attacking players who trust or cleared you (removes witnesses)
   - In 13-player games, coordinate with fellow werewolves via whisper

4. VOTE STRATEGY
   - Vote with the majority to blend in; avoid standing out
   - Try to direct votes toward Seer or other dangerous roles
   - If a fellow werewolf is targeted, do not all vote to save them (too obvious)
   - Sometimes sacrifice a fellow werewolf to gain village trust

5. FAKE CLAIMING (Advanced)
   - Consider fake claiming Seer or Medium to cause confusion
   - If fake claiming Seer, prepare fake divine results
   - Counter-claim real Seer to discredit them
   - Risk: if exposed, you become top execution target

6. ENDGAME
   - Track remaining player counts carefully
   - When close to winning, be more aggressive
   - In final votes, coordinate to ensure village majority is eliminated

Remember: Deception is your weapon. Stay calm, be consistent, and eliminate threats systematically.
"""
        return base + werewolf_guidance

    def _get_action_prompt(self, action_type: str, candidates: list[str] | None = None) -> str:
        """Get the action-specific prompt for the Werewolf.

        人狼用のアクション固有プロンプトを取得する.

        Args:
            action_type (str): Type of action / アクションの種類
            candidates (list[str] | None): Candidate names for target selection / 候補名

        Returns:
            str: Action-specific prompt / アクション固有のプロンプト
        """
        if candidates is None:
            candidates = self.get_alive_agents()

        # Add Werewolf-specific context for attack action
        if action_type == "attack":
            base_prompt = super()._get_action_prompt(action_type, candidates)

            attack_context = "\n\n[Werewolf-specific context]"
            attack_context += "\nPriority targets for attack:"
            attack_context += "\n1. Seer - can expose you if they divine you"
            attack_context += "\n2. Medium - can verify execution results and expose fake claims"
            attack_context += "\n3. Bodyguard - protects key village roles"
            attack_context += "\n4. Influential villagers - those leading discussions against you"

            # Track who has been attacked (to note failed attacks, likely guarded)
            if self.attacked_agents:
                attack_context += f"\n\nPreviously attacked: {', '.join(self.attacked_agents)}"

            attack_context += "\n\nConsider:"
            attack_context += "\n- Who claimed Seer or Medium? They are high priority"
            attack_context += "\n- Who might the Bodyguard protect tonight?"
            attack_context += "\n- Who is most dangerous to your team if they survive?"

            return base_prompt + attack_context

        # Add Werewolf-specific context for whisper action
        if action_type == "whisper":
            base_prompt = super()._get_action_prompt(action_type, candidates)

            whisper_context = "\n\n[Werewolf-specific context]"
            whisper_context += "\nThis is secret communication with your werewolf teammates."
            whisper_context += "\nUse this to:"
            whisper_context += "\n- Discuss who to attack tonight"
            whisper_context += "\n- Share suspicions about who is Seer/Bodyguard"
            whisper_context += "\n- Coordinate vote targets for tomorrow"
            whisper_context += "\n- Plan cover stories if one of you is accused"

            # Show known werewolf teammates
            if self.info and self.info.role_map:
                teammates = [
                    name for name, role in self.info.role_map.items()
                    if role == Role.WEREWOLF and name != self.get_my_game_name()
                ]
                if teammates:
                    whisper_context += f"\n\nYour werewolf teammates: {', '.join(teammates)}"

            return base_prompt + whisper_context

        # Add Werewolf-specific context for talk action
        if action_type == "talk":
            base_prompt = super()._get_action_prompt(action_type, candidates)

            talk_context = "\n\n[Werewolf-specific context]"
            talk_context += "\nRemember: You must pretend to be a villager."
            talk_context += "\n- Do NOT reveal that you are a werewolf"
            talk_context += "\n- Act natural and participate in discussions"
            talk_context += "\n- Subtly direct suspicion toward villagers"
            talk_context += "\n- If someone claims Seer and says you are WEREWOLF, deny it calmly"
            talk_context += "\n- Do not defend fellow werewolves too obviously"

            return base_prompt + talk_context

        # Add Werewolf-specific context for vote action
        if action_type == "vote":
            base_prompt = super()._get_action_prompt(action_type, candidates)

            vote_context = "\n\n[Werewolf-specific context]"
            vote_context += "\nVoting strategy:"
            vote_context += "\n- Try to vote out the real Seer or Medium"
            vote_context += "\n- Follow the village majority to avoid suspicion"
            vote_context += "\n- If a werewolf teammate is targeted, do not all vote against the execution"
            vote_context += "\n- Sometimes sacrificing a teammate builds trust"

            # Note fellow werewolves to avoid voting for them (unless strategic)
            if self.info and self.info.role_map:
                teammates = [
                    name for name, role in self.info.role_map.items()
                    if role == Role.WEREWOLF and name != self.get_my_game_name()
                ]
                if teammates:
                    vote_context += f"\n\nYour werewolf teammates (avoid voting for them unless necessary): {', '.join(teammates)}"

            return base_prompt + vote_context

        return super()._get_action_prompt(action_type, candidates)

    def _fallback_talk_like(self, action_type: str) -> str:
        """Return a werewolf-specific fallback message for TALK/WHISPER.

        人狼用のフォールバック発話を返す.
        村人のふりをしつつ、さりげなく村を誘導する発言.
        Day 0（挨拶）とDay 1以降（議論）で発言を分ける.
        重複防止機構付き.

        Args:
            action_type (str): "talk" or "whisper" / アクション種別

        Returns:
            str: Safe fallback message / 安全なフォールバック発話
        """
        if action_type == "whisper":
            # 人狼同士の囁き用フォールバック（13人村用）
            # 襲撃相談・戦略共有・投票調整
            candidates = [
                "Let us target someone influential tonight.",
                "We should avoid the obvious targets.",
                "I will follow your lead on the attack.",
                "Stay calm and blend in tomorrow.",
                "The Seer is dangerous. We should consider attacking them.",
                "Who do you think is the Bodyguard?",
                "Let us split our votes to avoid suspicion.",
                "I will act suspicious to draw attention away from you.",
            ]
            return random.choice(candidates)  # noqa: S311

        # Day 0: 挨拶系の発言（村人偽装）
        day0_candidates = [
            "Hello everyone. I hope we can work together.",
            "Nice to meet you all. Let us find the wolves.",
            "I am looking forward to hearing everyone's thoughts.",
            "Greetings. I want to help the village win.",
        ]

        # Day 1以降: 議論系の発言（村人偽装しつつ誘導）
        day1_candidates = [
            "I find the quiet ones most suspicious.",
            "We should vote carefully today.",
            "Does anyone have solid information?",
            "Something feels off about this situation.",
            "I am not fully convinced by that claim.",
            "We need to think about who benefits from this.",
            "That reasoning does not add up to me.",
            "I want to hear more before I decide.",
            "The real wolf is probably hiding in plain sight.",
            "We should focus on finding the werewolf.",
            "I have been observing everyone closely.",
            "Let us think about who has been acting strange.",
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

    def _get_werewolf_teammates(self) -> list[str]:
        """Get the list of fellow werewolves.

        仲間の人狼のリストを取得する.

        Returns:
            list[str]: List of werewolf teammate names / 人狼仲間の名前リスト
        """
        teammates: list[str] = []
        my_name = self.get_my_game_name()
        if self.info and self.info.role_map:
            for name, role in self.info.role_map.items():
                if role == Role.WEREWOLF and name != my_name:
                    teammates.append(name)
        return teammates

    def _fallback_vote(self) -> str:
        """Return a strategic fallback vote target for Werewolf.

        人狼用の戦略的フォールバック投票先を返す.
        戦略:
        1. 仲間の人狼には絶対に投票しない
        2. 占い師COした人を優先（本物の占い師を排除）
        3. それ以外の村人からランダム

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """
        candidates = self.get_alive_agents()
        my_name = self.get_my_game_name()

        # Remove self from candidates
        candidates = [c for c in candidates if c != my_name]

        if not candidates:
            return my_name  # Fallback to self if no other options

        # Strategy 1: Never vote for fellow werewolves
        teammates = self._get_werewolf_teammates()
        safe_candidates = [c for c in candidates if c not in teammates]

        if not safe_candidates:
            # All candidates are werewolves (shouldn't happen in normal game)
            self.agent_logger.logger.warning(
                "Werewolf has no non-wolf candidates to vote for"
            )
            return random.choice(candidates)  # noqa: S311

        # Strategy 2: Prioritize Seer claimers (eliminate the real Seer)
        seer_claimers = self._find_seer_claimers()
        seer_targets = [c for c in seer_claimers if c in safe_candidates]
        if seer_targets:
            self.agent_logger.logger.debug(
                "Werewolf voting for Seer claimer: %s",
                seer_targets,
            )
            return random.choice(seer_targets)  # noqa: S311

        # Strategy 3: Random from safe candidates (non-wolves)
        self.agent_logger.logger.debug(
            "Werewolf voting randomly from safe candidates: %s",
            safe_candidates,
        )
        return random.choice(safe_candidates)  # noqa: S311

    def _find_seer_claimers(self) -> list[str]:
        """Find players who claimed to be Seer based on talk history.

        会話履歴から占い師COした人を探す.

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

    def whisper(self) -> str:
        """Return response to whisper request.

        囁きリクエストに対する応答を返す.
        人狼同士の秘密通信。襲撃対象の相談や戦略の共有に使用.

        Returns:
            str: Whisper message / 囁きメッセージ
        """
        return super().whisper()

    def talk(self) -> str:
        """Return response to talk request.

        トークリクエストに対する応答を返す.
        村人のふりをして自然に会話に参加する.

        Returns:
            str: Talk message / 発言メッセージ
        """
        return super().talk()

    def vote(self) -> str:
        """Return response to vote request.

        投票リクエストに対する応答を返す.
        村人陣営の重要役職を優先的に処刑するよう誘導.
        LLMが失敗した場合は戦略的フォールバックを使用.

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """
        candidates = self._get_vote_candidates()
        response = self._call_llm("vote", candidates=candidates)
        if response:
            agent_name = self._extract_candidate_name(response, candidates)
            if agent_name:
                # Verify we're not voting for a fellow werewolf
                teammates = self._get_werewolf_teammates()
                if agent_name not in teammates:
                    return agent_name
                # LLM suggested a teammate, use fallback instead
                self.agent_logger.logger.debug(
                    "LLM suggested voting for teammate %s, using fallback",
                    agent_name,
                )

        # Use strategic fallback instead of random
        return self._fallback_vote()

    def attack(self) -> str:
        """Return response to attack request.

        襲撃リクエストに対する応答を返す.
        占い師・霊媒師・騎士などの重要役職を優先的に襲撃.

        Returns:
            str: Agent name to attack / 襲撃対象のエージェント名
        """
        return super().attack()

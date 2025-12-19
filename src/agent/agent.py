"""Module that defines the base class for agents.

エージェントの基底クラスを定義するモジュール.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from aiwolf_nlp_common.packet import (
    Info,
    Judge,
    Packet,
    Request,
    Role,
    Setting,
    Status,
    Talk,
    Vote,
)

from llm.base import LLMClient, LLMError
from llm.gemini import GeminiClient
from llm.prompt import PromptBuilder
from utils.agent_logger import AgentLogger
from utils.stoppable_thread import StoppableThread

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
T = TypeVar("T")


class Agent:
    """Base class for agents.

    エージェントの基底クラス.

    Notes:
        - `agent_name` is the *connection name* returned to the server on NAME request
          (e.g. "kanolab1").
        - In-game names used for talk, mentions, and target selection are provided by the
          server via `info.agent` and keys in `info.status_map` (e.g. "Minato" or "Agent[01]").
    """

    _RES_SKIP = "Skip"
    _RES_OVER = "Over"

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,
    ) -> None:
        """Initialize the agent.

        エージェントの初期化を行う.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
            name (str): Connection name returned on NAME request / 接続名
            game_id (str): Game ID / ゲームID
            role (Role): Role / 役職
        """
        self.config = config

        # Connection name used for NAME request.
        self.agent_name = name

        self.agent_logger = AgentLogger(config, name, game_id)
        self.request: Request | None = None
        self.info: Info | None = None
        self.setting: Setting | None = None

        # In-game identity (server-provided)
        self.game_name: str | None = None
        self.profile_text: str | None = None

        self.talk_history: list[Talk] = []
        self.whisper_history: list[Talk] = []
        self.role = role

        # Optional legacy random talk corpus (NOT used for fallback by default)
        self.comments: list[str] = []
        try:
            with Path.open(
                Path(str(self.config["path"]["random_talk"])),
                encoding="utf-8",
            ) as f:
                self.comments = f.read().splitlines()
        except OSError as e:
            # Keep the agent runnable even if the file is missing.
            self.agent_logger.logger.warning("Failed to load random talk file: %s", e)
            self.comments = []

        # Game state tracking / ゲーム状態追跡用
        self.day: int = 0
        self.divine_results: list[Judge] = []
        self.medium_results: list[Judge] = []
        self.executed_agents: list[str] = []
        self.attacked_agents: list[str] = []
        self.vote_history: list[list[Vote]] = []

        # LLM client initialization
        self.llm_client: LLMClient | None = None
        self.llm_enabled = self._init_llm_client()

    def _init_llm_client(self) -> bool:
        """Initialize the LLM client.

        LLMクライアントを初期化する.

        Returns:
            bool: True if LLM is enabled and initialized successfully / LLMが有効で正常に初期化された場合True
        """
        llm_config = self.config.get("llm", {})
        if not llm_config.get("enabled", False):
            self.agent_logger.logger.info("LLM is disabled in config")
            return False

        provider = llm_config.get("provider", "gemini")
        try:
            if provider == "gemini":
                self.llm_client = GeminiClient(self.config)
                self.agent_logger.logger.info("Gemini LLM client initialized")
                return True
            self.agent_logger.logger.warning("Unknown LLM provider: %s", provider)
            return False
        except LLMError as e:
            self.agent_logger.logger.warning("Failed to initialize LLM client: %s", e)
            return False

    def _get_system_prompt(self) -> str:
        """Get the system prompt for LLM.

        LLM用のシステムプロンプトを取得する.
        継承先でオーバーライドして役職固有のプロンプトを追加可能.

        Returns:
            str: System prompt / システムプロンプト
        """
        base = PromptBuilder.get_base_system_prompt()
        if self.profile_text:
            return f"{base}\n\nCharacter profile (must be reflected in your speaking style):\n{self.profile_text}\n"
        return base

    def _get_action_prompt(self, action_type: str, candidates: list[str] | None = None) -> str:
        """Get the action-specific prompt for LLM.

        LLM用のアクション固有プロンプトを取得する.
        継承先でオーバーライドして役職固有の戦略を追加可能.

        Args:
            action_type (str): Type of action / アクションの種類
            candidates (list[str] | None): Candidate names for target selection / 候補名

        Returns:
            str: Action-specific prompt / アクション固有のプロンプト
        """
        if candidates is None:
            candidates = self.get_alive_agents()
        return PromptBuilder.get_action_prompt(action_type, candidates)

    def _build_game_context(self) -> str:
        """Build game context for LLM.

        LLM用のゲームコンテキストを構築する.
        過去の占い/霊媒結果、処刑者/襲撃者の累積情報も含める.

        Returns:
            str: Game context / ゲームコンテキスト
        """
        game_name = self.get_my_game_name()
        return PromptBuilder.build_game_context(
            info=self.info,
            talk_history=self.talk_history,
            whisper_history=self.whisper_history,
            role=self.role,
            game_name=game_name,
            profile=self.profile_text,
            divine_results=self.divine_results,
            medium_results=self.medium_results,
            executed_agents=self.executed_agents,
            attacked_agents=self.attacked_agents,
        )

    def _call_llm(self, action_type: str, *, candidates: list[str] | None = None) -> str | None:
        """Call LLM to get response for the given action.

        指定されたアクションに対するLLMの応答を取得する.

        Args:
            action_type (str): Type of action / アクションの種類
            candidates (list[str] | None): Candidate names for target selection / 候補名

        Returns:
            str | None: LLM response or None if failed / LLMの応答、失敗時はNone
        """
        if not self.llm_enabled or not self.llm_client:
            return None

        try:
            system_prompt = self._get_system_prompt()
            game_context = self._build_game_context()
            action_prompt = self._get_action_prompt(action_type, candidates)

            user_prompt = f"""[Game state]\n{game_context}\n\n[Task]\n{action_prompt}"""
            response = self.llm_client.generate(system_prompt, user_prompt)
            self.agent_logger.logger.debug("LLM response for %s: %s", action_type, response)
            return response

        except LLMError as e:
            self.agent_logger.logger.warning("LLM call failed for %s: %s", action_type, e)
            return None

    def _extract_candidate_name(self, response: str, candidates: list[str]) -> str | None:
        """Extract one candidate name from an LLM response.

        LLMの応答から候補名（完全一致するターゲット名）を抽出する.

        Args:
            response (str): LLM response / LLMの応答
            candidates (list[str]): Valid candidate names / 有効な候補名

        Returns:
            str | None: Extracted candidate name or None / 抽出された候補名、なければNone
        """
        if not candidates:
            return None

        cleaned = response.strip().replace("\n", " ").replace("\r", " ").strip()
        cleaned = cleaned.strip("\"'")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Direct exact match first.
        if cleaned in candidates:
            return cleaned

        # Allow a leading @mention.
        if cleaned.startswith("@") and cleaned[1:] in candidates:
            return cleaned[1:]

        # Search candidates in text with basic boundary protection for ASCII names.
        for name in sorted(candidates, key=len, reverse=True):
            if name in cleaned:
                # If the name is pure ASCII letters/numbers, avoid matching inside a larger token.
                if name.isascii() and name.replace("[", "").replace("]", "").isalnum():
                    pattern = rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])"
                    if re.search(pattern, cleaned):
                        return name
                    continue
                return name

        return None

    def _sanitize_free_text(self, text: str, *, action_type: str) -> str:
        """Sanitize TALK/WHISPER free text output to comply with INLG rules.

        TALK/WHISPER の出力を規約に合わせてサニタイズする.

        This performs:
        - single-line normalization
        - comma removal
        - lightweight removal of common formatting prefixes

        Args:
            text (str): Raw text / 元テキスト
            action_type (str): "talk" or "whisper" / アクション種別

        Returns:
            str: Sanitized text / サニタイズ後のテキスト
        """
        # Normalize whitespace to a single line.
        cleaned = text.replace("\r", " ").replace("\n", " ").strip()
        cleaned = cleaned.strip("\"'")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Remove common prefixes the model may add.
        cleaned = re.sub(r"^(talk|whisper)\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)

        # Remove commas (half-width and common variants).
        cleaned = cleaned.replace(",", "")
        cleaned = cleaned.replace("，", "")
        cleaned = cleaned.replace("、", "")

        # Allow control responses as-is.
        if cleaned in {self._RES_SKIP, self._RES_OVER}:
            return cleaned

        # Avoid emitting AIWolf-style protocol tokens in natural language phases.
        if re.search(r"\b(VOTE|DIVINE|GUARD|ATTACK|COMINGOUT|ESTIMATE|AGREE|DISAGREE)\b", cleaned):
            fallback = self._fallback_talk_like(action_type)
            return fallback.strip()

        # Best-effort English enforcement: if the text contains Japanese characters
        # (excluding known in-game names), fall back to a safe English sentence.
        if self.info:
            masked = cleaned
            for name in self.info.status_map.keys():
                masked = masked.replace(f"@{name}", "")
                masked = masked.replace(name, "")
            if re.search(r"[\u3000-\u303f\u3040-\u30ff\u3400-\u9fff]", masked):
                fallback = self._fallback_talk_like(action_type)
                return fallback.strip()

        return cleaned

    def _fallback_talk_like(self, action_type: str) -> str:
        """Return a safe fallback message for TALK/WHISPER.

        TALK/WHISPER 用の安全なフォールバック発話を返す.

        Args:
            action_type (str): "talk" or "whisper" / アクション種別

        Returns:
            str: Safe fallback message / 安全なフォールバック発話
        """
        if action_type == "whisper":
            candidates = [
                "We should keep calm and watch reactions.",
                "Let us coordinate our votes and keep a consistent story.",
                "I will follow the flow and avoid drawing attention.",
            ]
        else:
            candidates = [
                "Hello everyone.",
                "I want to hear your reasoning.",
                "Let us discuss who seems suspicious.",
                "I am not sure yet but I will share my thoughts soon.",
            ]
        # Avoid commas by construction.
        return random.choice(candidates)  # noqa: S311

    def _random_choice(self, candidates: list[str]) -> str:
        """Pick one element from candidates.

        候補から1つ選ぶ.

        Args:
            candidates (list[str]): Candidate list / 候補リスト

        Returns:
            str: One selected candidate / 選ばれた候補
        """
        if not candidates:
            return ""
        return random.choice(candidates)  # noqa: S311

    @staticmethod
    def timeout(func: Callable[P, T]) -> Callable[P, T]:
        """Decorator to set action timeout.

        アクションタイムアウトを設定するデコレータ.

        Args:
            func (Callable[P, T]): Function to be decorated / デコレート対象の関数

        Returns:
            Callable[P, T]: Function with timeout functionality / タイムアウト機能を追加した関数
        """

        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            res: T | Exception = Exception("No result")

            def execute_with_timeout() -> None:
                nonlocal res
                try:
                    res = func(*args, **kwargs)
                except Exception as e:  # noqa: BLE001
                    res = e

            thread = StoppableThread(target=execute_with_timeout)
            thread.start()
            self = args[0] if args else None
            if not isinstance(self, Agent):
                raise TypeError(self, " is not an Agent instance")
            timeout_value = (self.setting.timeout.action if hasattr(self, "setting") and self.setting else 0) // 1000
            if timeout_value > 0:
                thread.join(timeout=timeout_value)
                if thread.is_alive():
                    self.agent_logger.logger.warning(
                        "アクションがタイムアウトしました: %s",
                        self.request,
                    )
                    if bool(self.config["agent"]["kill_on_timeout"]):
                        thread.stop()
                        self.agent_logger.logger.warning(
                            "アクションを強制終了しました: %s",
                            self.request,
                        )
            else:
                thread.join()
            if isinstance(res, Exception):  # type: ignore[arg-type]
                raise res
            return res

        return _wrapper

    def set_packet(self, packet: Packet) -> None:
        """Set packet information.

        パケット情報をセットする.

        Args:
            packet (Packet): Received packet / 受信したパケット
        """
        self.request = packet.request

        if packet.info:
            self.info = packet.info
            self.game_name = packet.info.agent

            # Store profile (only sent on INITIALIZE)
            profile = getattr(packet.info, "profile", None)
            if self.request == Request.INITIALIZE and isinstance(profile, str) and profile.strip():
                self.profile_text = profile

        if packet.setting:
            self.setting = packet.setting

        if packet.talk_history:
            self.talk_history.extend(packet.talk_history)

        if packet.whisper_history:
            self.whisper_history.extend(packet.whisper_history)

        if self.request == Request.INITIALIZE:
            self.talk_history = []
            self.whisper_history = []

        self.agent_logger.logger.debug(packet)

    def get_my_game_name(self) -> str:
        """Get this agent's in-game name.

        自分のゲーム内名を取得する.

        Returns:
            str: In-game name (best-effort) / ゲーム内名（推定を含む）
        """
        if self.info and getattr(self.info, "agent", None):
            return str(self.info.agent)
        if self.game_name:
            return self.game_name
        # Fallback to connection name, but this may differ from in-game name.
        return self.agent_name

    def get_alive_agents(self) -> list[str]:
        """Get the list of alive agents.

        生存しているエージェントのリストを取得する.

        Returns:
            list[str]: List of alive agent in-game names / 生存エージェント名のリスト
        """
        if not self.info:
            return []
        return [k for k, v in self.info.status_map.items() if v == Status.ALIVE]

    def _get_vote_candidates(self) -> list[str]:
        """Get valid vote candidates.

        有効な投票候補を取得する.

        Returns:
            list[str]: Candidate names / 候補名
        """
        candidates = self.get_alive_agents()
        me = self.get_my_game_name()
        if not self.setting:
            # Safe default: exclude self unless explicitly allowed by server setting.
            return [c for c in candidates if c != me]

        allow_self_vote = getattr(getattr(self.setting, "vote", None), "allow_self_vote", False)
        if allow_self_vote is False:
            candidates = [c for c in candidates if c != me]
        return candidates

    def _get_divine_candidates(self) -> list[str]:
        """Get valid divine candidates.

        有効な占い候補を取得する.

        Returns:
            list[str]: Candidate names / 候補名
        """
        me = self.get_my_game_name()
        return [c for c in self.get_alive_agents() if c != me]

    def _get_guard_candidates(self) -> list[str]:
        """Get valid guard candidates.

        有効な護衛候補を取得する.

        Returns:
            list[str]: Candidate names / 候補名
        """
        me = self.get_my_game_name()
        return [c for c in self.get_alive_agents() if c != me]

    def _get_attack_candidates(self) -> list[str]:
        """Get valid attack candidates.

        有効な襲撃候補を取得する.

        Returns:
            list[str]: Candidate names / 候補名
        """
        alive = self.get_alive_agents()
        me = self.get_my_game_name()

        # Exclude known werewolves from candidates.
        wolves: set[str] = set()
        if self.info and self.info.role_map:
            for agent_name, role in self.info.role_map.items():
                if role == Role.WEREWOLF:
                    wolves.add(agent_name)

        return [a for a in alive if a != me and a not in wolves]

    def name(self) -> str:
        """Return response to name request.

        名前リクエストに対する応答を返す.

        Returns:
            str: Connection name / 接続名
        """
        return self.agent_name

    def initialize(self) -> None:
        """Perform initialization for game start request.

        ゲーム開始リクエストに対する初期化処理を行う.
        ゲーム状態（日付、各種結果履歴）をリセットし、設定情報をログ出力する.
        """
        # Reset game state / ゲーム状態をリセット
        self.day = 0
        self.divine_results = []
        self.medium_results = []
        self.executed_agents = []
        self.attacked_agents = []
        self.vote_history = []

        # Log game settings / ゲーム設定をログ出力
        if self.setting:
            self.agent_logger.logger.info(
                "Game initialized: %d players, role=%s",
                self.setting.agent_count,
                self.role.name if self.role else "Unknown",
            )
        if self.info:
            self.agent_logger.logger.info(
                "In-game name: %s, Role map visible: %s",
                self.info.agent,
                bool(self.info.role_map),
            )
            if getattr(self.info, "profile", None):
                self.agent_logger.logger.info("Profile received")

    def daily_initialize(self) -> None:
        """Perform processing for daily initialization request.

        昼開始リクエストに対する処理を行う.
        占い/霊媒結果の保存、処刑者/襲撃者の記録、生存者情報のログ出力を行う.
        """
        if not self.info:
            return

        # Update day count / 日付を更新
        self.day = self.info.day

        self.agent_logger.logger.info("=== Day %d started ===", self.day)

        # Store divine result (for seer) / 占い結果を保存（占い師用）
        if self.info.divine_result:
            self.divine_results.append(self.info.divine_result)
            self.agent_logger.logger.info(
                "Divine result: %s is %s",
                self.info.divine_result.target,
                self.info.divine_result.result.name,
            )

        # Store medium result (for medium) / 霊媒結果を保存（霊媒師用）
        if self.info.medium_result:
            self.medium_results.append(self.info.medium_result)
            self.agent_logger.logger.info(
                "Medium result: %s was %s",
                self.info.medium_result.target,
                self.info.medium_result.result.name,
            )

        # Record executed agent / 処刑者を記録
        if self.info.executed_agent:
            self.executed_agents.append(self.info.executed_agent)
            self.agent_logger.logger.info(
                "Executed yesterday: %s",
                self.info.executed_agent,
            )

        # Record attacked agent / 襲撃者を記録
        if self.info.attacked_agent:
            self.attacked_agents.append(self.info.attacked_agent)
            self.agent_logger.logger.info(
                "Attacked last night: %s",
                self.info.attacked_agent,
            )

        # Log alive agents / 生存者をログ出力
        alive = self.get_alive_agents()
        self.agent_logger.logger.info("Alive agents: %s (%d)", alive, len(alive))

    def whisper(self) -> str:
        """Return response to whisper request.

        囁きリクエストに対する応答を返す.

        Returns:
            str: Whisper message / 囁きメッセージ
        """
        response = self._call_llm("whisper", candidates=self.get_alive_agents())
        text = response if response else self._fallback_talk_like("whisper")
        text = self._sanitize_free_text(text, action_type="whisper")
        if not text:
            return self._RES_OVER
        return text

    def talk(self) -> str:
        """Return response to talk request.

        トークリクエストに対する応答を返す.

        Returns:
            str: Talk message / 発言メッセージ
        """
        response = self._call_llm("talk", candidates=self.get_alive_agents())
        text = response if response else self._fallback_talk_like("talk")
        text = self._sanitize_free_text(text, action_type="talk")
        if not text:
            return self._RES_OVER
        return text

    def daily_finish(self) -> None:
        """Perform processing for daily finish request.

        昼終了リクエストに対する処理を行う.
        投票結果の保存と集計、その日の発言数をログ出力する.
        """
        if not self.info:
            return

        self.agent_logger.logger.info("=== Day %d finished ===", self.day)

        # Store vote results / 投票結果を保存
        if self.info.vote_list:
            self.vote_history.append(list(self.info.vote_list))
            self.agent_logger.logger.debug(
                "Vote results for day %d: %d votes recorded",
                self.day,
                len(self.info.vote_list),
            )
            # Log vote summary / 投票サマリをログ出力
            vote_counts: dict[str, int] = {}
            for vote in self.info.vote_list:
                target = vote.target
                vote_counts[target] = vote_counts.get(target, 0) + 1
            self.agent_logger.logger.info("Vote summary: %s", vote_counts)

        # Log talk count for the day / その日の発言数をログ出力
        day_talks = [t for t in self.talk_history if t.day == self.day]
        self.agent_logger.logger.info(
            "Total talks today: %d",
            len(day_talks),
        )

    def divine(self) -> str:
        """Return response to divine request.

        占いリクエストに対する応答を返す.

        Returns:
            str: Agent name to divine / 占い対象のエージェント名
        """
        candidates = self._get_divine_candidates()
        response = self._call_llm("divine", candidates=candidates)
        if response:
            agent_name = self._extract_candidate_name(response, candidates)
            if agent_name:
                return agent_name

        # Fallback: pick from candidates if possible
        fallback = self._random_choice(candidates) or self._random_choice(self.get_alive_agents())
        return fallback

    def guard(self) -> str:
        """Return response to guard request.

        護衛リクエストに対する応答を返す.

        Returns:
            str: Agent name to guard / 護衛対象のエージェント名
        """
        candidates = self._get_guard_candidates()
        response = self._call_llm("guard", candidates=candidates)
        if response:
            agent_name = self._extract_candidate_name(response, candidates)
            if agent_name:
                return agent_name

        fallback = self._random_choice(candidates) or self._random_choice(self.get_alive_agents())
        return fallback

    def vote(self) -> str:
        """Return response to vote request.

        投票リクエストに対する応答を返す.

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """
        candidates = self._get_vote_candidates()
        response = self._call_llm("vote", candidates=candidates)
        if response:
            agent_name = self._extract_candidate_name(response, candidates)
            if agent_name:
                return agent_name

        fallback = self._random_choice(candidates) or self._random_choice(self.get_alive_agents())
        return fallback

    def attack(self) -> str:
        """Return response to attack request.

        襲撃リクエストに対する応答を返す.

        Returns:
            str: Agent name to attack / 襲撃対象のエージェント名
        """
        candidates = self._get_attack_candidates()
        response = self._call_llm("attack", candidates=candidates)
        if response:
            agent_name = self._extract_candidate_name(response, candidates)
            if agent_name:
                return agent_name

        fallback = self._random_choice(candidates) or self._random_choice(self.get_alive_agents())
        return fallback

    def finish(self) -> None:
        """Perform processing for game finish request.

        ゲーム終了リクエストに対する処理を行う.
        最終生存状態、役職マップ、ゲーム統計（日数、発言数、処刑/襲撃者）をログ出力する.
        """
        self.agent_logger.logger.info("========== GAME FINISHED ==========")

        # Log final game state / 最終ゲーム状態をログ出力
        if self.info:
            is_alive = self.get_my_game_name() in self.get_alive_agents()
            self.agent_logger.logger.info(
                "Final status: %s (Role: %s)",
                "ALIVE" if is_alive else "DEAD",
                self.role.name if self.role else "Unknown",
            )

            # Log final role map if available / 役職マップが利用可能ならログ出力
            if self.info.role_map:
                self.agent_logger.logger.info("Final role map: %s", self.info.role_map)

        # Log game statistics / ゲーム統計をログ出力
        self.agent_logger.logger.info(
            "Game lasted %d days",
            self.day,
        )
        self.agent_logger.logger.info(
            "Total talks: %d, Total whispers: %d",
            len(self.talk_history),
            len(self.whisper_history),
        )
        self.agent_logger.logger.info(
            "Executed agents: %s",
            self.executed_agents,
        )
        self.agent_logger.logger.info(
            "Attacked agents: %s",
            self.attacked_agents,
        )

        # Log role-specific results / 役職固有の結果をログ出力
        if self.divine_results:
            self.agent_logger.logger.info(
                "Divine results: %d divinations performed",
                len(self.divine_results),
            )
        if self.medium_results:
            self.agent_logger.logger.info(
                "Medium results: %d inquiries performed",
                len(self.medium_results),
            )

        self.agent_logger.logger.info("====================================")

    @timeout
    def action(self) -> str | None:  # noqa: C901, PLR0911
        """Execute action according to request type.

        リクエストの種類に応じたアクションを実行する.

        Returns:
            str | None: Action result string or None / アクションの結果文字列またはNone
        """
        match self.request:
            case Request.NAME:
                return self.name()
            case Request.TALK:
                return self.talk()
            case Request.WHISPER:
                return self.whisper()
            case Request.VOTE:
                return self.vote()
            case Request.DIVINE:
                return self.divine()
            case Request.GUARD:
                return self.guard()
            case Request.ATTACK:
                return self.attack()
            case Request.INITIALIZE:
                self.initialize()
            case Request.DAILY_INITIALIZE:
                self.daily_initialize()
            case Request.DAILY_FINISH:
                self.daily_finish()
            case Request.FINISH:
                self.finish()
            case _:
                pass
        return None

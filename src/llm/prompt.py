"""Prompt building utilities for LLM interactions.

LLMとのやり取りのためのプロンプト構築ユーティリティ.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Info, Judge, Role, Talk


class PromptBuilder:
    """Helper class for building prompts from game state.

    ゲーム状態からプロンプトを構築するヘルパークラス.
    """

    @staticmethod
    def build_game_context(  # noqa: PLR0913
        info: Info | None,
        talk_history: list[Talk],
        whisper_history: list[Talk],
        role: Role,
        game_name: str,
        *,
        profile: str | None = None,
        divine_results: list[Judge] | None = None,
        medium_results: list[Judge] | None = None,
        executed_agents: list[str] | None = None,
        attacked_agents: list[str] | None = None,
    ) -> str:
        """Build game context string from current game state.

        現在のゲーム状態からゲームコンテキスト文字列を構築する.

        Args:
            info (Info | None): Current game info / 現在のゲーム情報
            talk_history (list[Talk]): History of public talks / 公開発言の履歴
            whisper_history (list[Talk]): History of whispers (werewolf only) / 囁きの履歴（人狼のみ）
            role (Role): Agent's role / エージェントの役職
            game_name (str): Agent's in-game name / ゲーム内でのエージェント名
            profile (str | None): Character profile from the server (INITIALIZE only) /
                サーバから渡されるキャラクター設定（INITIALIZEのみ）
            divine_results (list[Judge] | None): All divine results so far / 過去の占い結果
            medium_results (list[Judge] | None): All medium results so far / 過去の霊媒結果
            executed_agents (list[str] | None): List of executed agents / 処刑者リスト
            attacked_agents (list[str] | None): List of attacked agents / 襲撃者リスト

        Returns:
            str: Formatted game context / フォーマットされたゲームコンテキスト
        """
        context_parts: list[str] = []

        # Basic info (English for INLG 2025)
        context_parts.append(f"Your in-game name: {game_name}")
        context_parts.append(f"Your role: {role}")

        if profile:
            context_parts.append("\n[Your character profile from the server]")
            context_parts.append(profile)

        if info:
            # Day information
            context_parts.append(f"Current day: {info.day}")

            # Agent status
            context_parts.append("\n[Alive status]")
            alive_agents: list[str] = []
            dead_agents: list[str] = []
            for agent, status in info.status_map.items():
                if str(status) == "ALIVE":
                    alive_agents.append(agent)
                else:
                    dead_agents.append(agent)

            context_parts.append("Alive agents:")
            context_parts.extend([f"- {a}" for a in alive_agents])
            if dead_agents:
                context_parts.append("Dead agents:")
                context_parts.extend([f"- {a}" for a in dead_agents])

            # Role map (what we know)
            if info.role_map:
                context_parts.append("\n[Known roles]")
                for agent, agent_role in info.role_map.items():
                    context_parts.append(f"- {agent}: {agent_role}")

            # Communication limits (if provided)
            remain_count = getattr(info, "remain_count", None)
            remain_length = getattr(info, "remain_length", None)
            remain_skip = getattr(info, "remain_skip", None)
            if remain_count is not None or remain_length is not None or remain_skip is not None:
                context_parts.append("\n[Communication limits]")
                if remain_count is not None:
                    context_parts.append(f"- remaining requests: {remain_count}")
                if remain_length is not None:
                    context_parts.append(f"- remaining length budget: {remain_length}")
                if remain_skip is not None:
                    context_parts.append(f"- remaining skips: {remain_skip}")

        # Divine results (for seer) - show all past results
        if divine_results:
            context_parts.append("\n[Divination results (all)]")
            for result in divine_results:
                context_parts.append(f"- {result.target}: {result.result}")
        elif info and info.divine_result:
            # Fallback to current info if no history provided
            context_parts.append("\n[Divination result]")
            context_parts.append(
                f"- {info.divine_result.target}: {info.divine_result.result}",
            )

        # Medium results - show all past results
        if medium_results:
            context_parts.append("\n[Medium results (all)]")
            for result in medium_results:
                context_parts.append(f"- {result.target}: {result.result}")
        elif info and info.medium_result:
            # Fallback to current info if no history provided
            context_parts.append("\n[Medium result]")
            context_parts.append(
                f"- {info.medium_result.target}: {info.medium_result.result}",
            )

        # Executed agents - show all
        if executed_agents:
            context_parts.append("\nExecuted agents:")
            context_parts.extend([f"- {a}" for a in executed_agents])
        elif info and info.executed_agent:
            context_parts.append("\nExecuted agent:")
            context_parts.append(f"- {info.executed_agent}")

        # Attacked agents - show all
        if attacked_agents:
            context_parts.append("\nAttacked agents:")
            context_parts.extend([f"- {a}" for a in attacked_agents])
        elif info and info.attacked_agent:
            context_parts.append("\nAttacked agent:")
            context_parts.append(f"- {info.attacked_agent}")

        # Vote list
        if info and info.vote_list:
            context_parts.append("\n[Vote history]")
            for vote in info.vote_list:
                context_parts.append(f"- {vote.agent} -> {vote.target}")

        # Talk history
        if talk_history:
            context_parts.append("\n[Recent talk history]")
            recent_talks = talk_history[-20:]  # Last 20 talks
            for talk in recent_talks:
                context_parts.append(f"- {talk.agent}: {talk.text}")

        # Whisper history (for werewolf)
        if whisper_history:
            context_parts.append("\n[Recent whisper history]")
            recent_whispers = whisper_history[-10:]  # Last 10 whispers
            for whisper in recent_whispers:
                context_parts.append(f"- {whisper.agent}: {whisper.text}")

        return "\n".join(context_parts)

    @staticmethod
    def get_base_system_prompt() -> str:
        """Get the base system prompt for all agents.

        全エージェント共通の基本システムプロンプトを取得する.

        Returns:
            str: Base system prompt / 基本システムプロンプト
        """
        return """You are a player agent for AIWolf INLG 2025.
Your objective is to maximize your team win rate while strictly following the server constraints.

Output constraints for TALK and WHISPER
- Output must be a single line of plain text
- Use natural English only, no protocol, no structured commands
- Never output the half width comma character ","
- TALK and WHISPER length must be within the server limit
  - Assume a hard cap of 125 characters excluding spaces
  - Use a safety margin: aim for 110 or fewer non space characters
  - The server may truncate overlong output and the truncated tail is lost
  - Put the essential content first, avoid long prefaces
  - Before sending, self check by counting non space characters
- Mentions
  - To address someone, start the message with "@Name "
- If you choose not to speak, output exactly "Skip"
- If you will not speak again today, output exactly "Over"

Output constraints for target actions
- For VOTE, ATTACK, DIVINE, GUARD, output only the exact target name and nothing else

Game essentials
- Two teams, villager team and werewolf team
- Each day: discussion, vote execution, then night actions
- Do not rely on messages from the same turn

Common decision and talk policy
- Track public role claims, result claims, and declared vote intents
- Flag contradictions and impossible sequences
- Maintain a suspicion score per player and update after each message
- In TALK, do at most one main act: state your vote target with one short reason, or ask one direct question
- Keep statements consistent across days, if changing mind, give one short reason

Role set reminders
- 5 players: Villager x2, Seer x1, Werewolf x1, Possessed x1
- 13 players: Villager x6, Seer x1, Medium x1, Bodyguard x1, Werewolf x3, Possessed x1

Always return a valid output within the time limit.

"""

    @staticmethod
    def get_action_prompt(action_type: str, alive_agents: list[str]) -> str:
        """Get prompt for specific action type.

        特定のアクションタイプに対するプロンプトを取得する.

        Args:
            action_type (str): Type of action (talk, vote, divine, guard, attack, whisper) /
                              アクションの種類
            alive_agents (list[str]): List of alive agent names / 生存エージェント名のリスト

        Returns:
            str: Action-specific prompt / アクション固有のプロンプト
        """
        example_name = alive_agents[0] if alive_agents else "Agent[01]"
        candidates_block = "\n".join([f"- {a}" for a in alive_agents]) if alive_agents else "- (none)"

        prompts = {
            "talk": """Write one short TALK message.
Use natural English in 1-2 short sentences.
Do NOT use commas.
Return only the message text.""",
            "vote": f"""Choose one player to vote for.
Candidates:
{candidates_block}

Return exactly one candidate name (example: {example_name}).
Return ONLY the name with no extra words, no punctuation, and no quotes.""",
            "divine": f"""Choose one player to divine.
Candidates:
{candidates_block}

Return exactly one candidate name (example: {example_name}).
Return ONLY the name with no extra words, no punctuation, and no quotes.""",
            "guard": f"""Choose one player to guard.
Candidates:
{candidates_block}

Return exactly one candidate name (example: {example_name}).
Return ONLY the name with no extra words, no punctuation, and no quotes.""",
            "attack": f"""Choose one player to attack.
Candidates:
{candidates_block}

Return exactly one candidate name (example: {example_name}).
Return ONLY the name with no extra words, no punctuation, and no quotes.""",
            "whisper": """Write one short WHISPER message to your werewolf teammates.
Use natural English in 1-2 short sentences.
Do NOT use commas.
Return only the message text.""",
        }

        return prompts.get(action_type, "Respond.")

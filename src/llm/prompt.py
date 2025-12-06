"""Prompt building utilities for LLM interactions.

LLMとのやり取りのためのプロンプト構築ユーティリティ.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Info, Judge, Role, Status, Talk


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
        agent_name: str,
        *,
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
            agent_name (str): Agent's name / エージェントの名前
            divine_results (list[Judge] | None): All divine results so far / 過去の占い結果
            medium_results (list[Judge] | None): All medium results so far / 過去の霊媒結果
            executed_agents (list[str] | None): List of executed agents / 処刑者リスト
            attacked_agents (list[str] | None): List of attacked agents / 襲撃者リスト

        Returns:
            str: Formatted game context / フォーマットされたゲームコンテキスト
        """
        context_parts: list[str] = []

        # Basic info
        context_parts.append(f"あなたの名前: {agent_name}")
        context_parts.append(f"あなたの役職: {role}")

        if info:
            # Day information
            context_parts.append(f"現在の日数: {info.day}日目")

            # Agent status
            context_parts.append("\n【生存状況】")
            alive_agents: list[str] = []
            dead_agents: list[str] = []
            for agent, status in info.status_map.items():
                if str(status) == "ALIVE":
                    alive_agents.append(agent)
                else:
                    dead_agents.append(agent)

            context_parts.append(f"生存者: {', '.join(alive_agents)}")
            if dead_agents:
                context_parts.append(f"死亡者: {', '.join(dead_agents)}")

            # Role map (what we know)
            if info.role_map:
                context_parts.append("\n【判明している役職】")
                for agent, agent_role in info.role_map.items():
                    context_parts.append(f"  {agent}: {agent_role}")

        # Divine results (for seer) - show all past results
        if divine_results:
            context_parts.append("\n【占い結果（全履歴）】")
            for result in divine_results:
                context_parts.append(f"  {result.target}: {result.result}")
        elif info and info.divine_result:
            # Fallback to current info if no history provided
            context_parts.append("\n【占い結果】")
            context_parts.append(
                f"  {info.divine_result.target}: {info.divine_result.result}",
            )

        # Medium results - show all past results
        if medium_results:
            context_parts.append("\n【霊媒結果（全履歴）】")
            for result in medium_results:
                context_parts.append(f"  {result.target}: {result.result}")
        elif info and info.medium_result:
            # Fallback to current info if no history provided
            context_parts.append("\n【霊媒結果】")
            context_parts.append(
                f"  {info.medium_result.target}: {info.medium_result.result}",
            )

        # Executed agents - show all
        if executed_agents:
            context_parts.append(f"\n処刑されたエージェント: {', '.join(executed_agents)}")
        elif info and info.executed_agent:
            context_parts.append(f"\n処刑されたエージェント: {info.executed_agent}")

        # Attacked agents - show all
        if attacked_agents:
            context_parts.append(f"襲撃されたエージェント: {', '.join(attacked_agents)}")
        elif info and info.attacked_agent:
            context_parts.append(f"襲撃されたエージェント: {info.attacked_agent}")

        # Vote list
        if info and info.vote_list:
            context_parts.append("\n【投票履歴】")
            for vote in info.vote_list:
                context_parts.append(f"  {vote.agent} → {vote.target}")

        # Talk history
        if talk_history:
            context_parts.append("\n【会話履歴】")
            recent_talks = talk_history[-20:]  # Last 20 talks
            for talk in recent_talks:
                context_parts.append(f"  {talk.agent}: {talk.text}")

        # Whisper history (for werewolf)
        if whisper_history:
            context_parts.append("\n【人狼の囁き履歴】")
            recent_whispers = whisper_history[-10:]  # Last 10 whispers
            for whisper in recent_whispers:
                context_parts.append(f"  {whisper.agent}: {whisper.text}")

        return "\n".join(context_parts)

    @staticmethod
    def get_base_system_prompt() -> str:
        """Get the base system prompt for all agents.

        全エージェント共通の基本システムプロンプトを取得する.

        Returns:
            str: Base system prompt / 基本システムプロンプト
        """
        return """あなたは人狼ゲームをプレイするAIエージェントです。

人狼ゲームのルール:
- 村人陣営と人狼陣営に分かれて戦います
- 昼のターンでは議論を行い、投票で処刑する人を決めます
- 夜のターンでは人狼が村人を襲撃します
- 村人陣営は人狼を全員処刑すれば勝利、人狼陣営は村人の数が人狼以下になれば勝利です

役職:
- 村人: 特殊能力なし
- 占い師: 毎晩1人を占い、人狼かどうかを知ることができる
- 霊媒師: 処刑された人が人狼だったかを知ることができる
- 騎士: 毎晩1人を護衛し、人狼の襲撃から守ることができる
- 人狼: 毎晩村人を1人襲撃できる。他の人狼と夜に会話できる
- 狂人: 村人陣営だが、人狼の勝利が自分の勝利となる

重要な指示:
- 常に自分の陣営の勝利を目指してください
- 論理的に考え、他のプレイヤーの発言や行動から情報を推理してください
- 応答は簡潔に、ゲームの文脈に適した形式で返してください"""

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
        agents_str = ", ".join(alive_agents)
        example_name = alive_agents[0] if alive_agents else "ケンジ"

        prompts = {
            "talk": f"""発言してください。
他のプレイヤーとの議論、情報共有、推理の表明などを行ってください。
自然な日本語で、1〜2文程度で簡潔に発言してください。
発言内容のみを返してください。""",
            "vote": f"""投票対象を選んでください。
生存者: {agents_str}

最も怪しいと思う人物、または戦略的に処刑すべき人物を選んでください。
エージェント名のみを返してください（例: {example_name}）。""",
            "divine": f"""占い対象を選んでください。
生存者: {agents_str}

最も占う価値があると思う人物を選んでください。
エージェント名のみを返してください（例: {example_name}）。""",
            "guard": f"""護衛対象を選んでください。
生存者: {agents_str}

最も守る価値があると思う人物を選んでください。
エージェント名のみを返してください（例: {example_name}）。""",
            "attack": f"""襲撃対象を選んでください。
生存者: {agents_str}

村人陣営の勝利に貢献しそうな人物を優先的に襲撃してください。
エージェント名のみを返してください（例: {example_name}）。""",
            "whisper": f"""人狼同士の囁きを行ってください。
仲間の人狼と戦略を共有してください。
自然な日本語で、1〜2文程度で簡潔に発言してください。
発言内容のみを返してください。""",
        }

        return prompts.get(action_type, "応答してください。")

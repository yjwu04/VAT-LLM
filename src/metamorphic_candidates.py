"""Deterministic, semantics-preserving text neighbours for Metamorphic-VAT.

The module deliberately performs one local edit only.  It never rewrites an
answer or system message; callers pass one question string and receive one
candidate plus auditable edit metadata.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


FAMILY_CHARACTER_NOISE = "character_noise"
FAMILY_TOKENIZATION = "tokenization_disruption"
FAMILY_CODE_SWITCH = "local_code_switching"
FAMILY_ORDER = (
    FAMILY_CHARACTER_NOISE,
    FAMILY_TOKENIZATION,
    FAMILY_CODE_SWITCH,
)

CHARACTER_SUBTYPES = (
    "adjacent_swap",
    "repeat_or_insert_character",
    "non_language_symbol",
    "visual_substitution",
)
TOKENIZATION_SUBTYPES = (
    "intra_word_space",
    "punctuation_delimiter",
    "visual_split",
)

_NEGATION = re.compile(
    r"\b(?:no|not|never|none|neither|nor|without|cannot|can't|do\s+not|"
    r"does\s+not|don't|doesn't|must\s+not|should\s+not|except)\b",
    re.IGNORECASE,
)
_QUANTITY = re.compile(
    r"\b(?:(?:exactly|at\s+least|at\s+most|no\s+more\s+than|"
    r"no\s+fewer\s+than|fewer\s+than|less\s+than|more\s+than)\s+"
    r"(?:\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|hundred|thousand|million)(?:\s+[A-Za-z][A-Za-z'-]*)?|"
    r"exactly|at\s+least|at\s+most|no\s+more\s+than|no\s+fewer\s+than|"
    r"minimum|maximum|fewer\s+than|less\s+than|more\s+than|between|zero|one|"
    r"two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|hundred|"
    r"thousand|million|single|double|triple)\b",
    re.IGNORECASE,
)
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_SPECIAL = re.compile(r"<\|[^|\r\n]+\|>")
_NUMBER = re.compile(r"(?<!\w)[+-]?\d(?:[\d,.:/%-]*\d)?(?!\w)")
_FENCED_CODE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")
_INLINE_CODE = re.compile(r"`[^`\r\n]+`")
_MATH = re.compile(r"\$[^$\r\n]+\$|\\\([^\r\n]*?\\\)|\\\[[\s\S]*?\\\]")
_QUOTED = re.compile(
    r'"[^"\r\n]+"|(?<![A-Za-z])\'[^\'\r\n]+\'(?![A-Za-z])'
)
_JSON_KEY = re.compile(r'"(?:[^"\\]|\\.)*"\s*:')
_OPTION_LABEL = re.compile(r"(?m)^\s*(?:[A-Ha-h]|\d{1,2})[.)]\s+")
_FORMULA_ATOM = r"(?:[A-Za-z_]\w*|[+-]?\d+(?:\.\d+)?)"
_FORMULA = re.compile(
    rf"\b[A-Za-z_]\w*\s*(?:==|!=|<=|>=|=|<|>)\s*{_FORMULA_ATOM}"
    rf"(?:\s*[+\-*/^]\s*{_FORMULA_ATOM})*|"
    r"\b(?:sin|cos|tan|log|sqrt)\s*\([^\r\n)]*\)"
)
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
_CJK_WORD = re.compile(r"[\u3400-\u9fff]{2,8}")

_SEMANTICALLY_PROTECTED_WORDS = frozenset(
    {
        "all", "any", "each", "either", "every", "first", "last", "least",
        "most", "must", "none", "only", "same", "second", "third", "true",
        "false", "yes", "no", "not", "never", "without", "except", "unless",
    }
)

_VISUAL_MAP: Mapping[str, str] = {
    "a": "α", "b": "Ь", "c": "ϲ", "e": "е", "i": "1", "l": "1",
    "o": "0", "p": "ρ", "s": "ѕ", "u": "υ", "x": "х", "y": "у",
}

# Phrase-first bilingual lexicon.  Entries are intentionally short and
# unambiguous: this is local code-switching, not free-form translation.
DEFAULT_CODE_SWITCH_LEXICON: tuple[tuple[str, str], ...] = (
    ("what is", "什么是"), ("what are", "哪些是"),
    ("what was", "什么曾是"), ("what were", "哪些曾是"),
    ("how can", "如何能"), ("how do", "如何"),
    ("how does", "如何"), ("how is", "如何"), ("how are", "如何"),
    ("why is", "为什么"), ("why are", "为什么"),
    ("can you", "你能"), ("could you", "你能否"),
    ("would you", "你能否"), ("tell me", "告诉我"),
    ("i want", "我想"), ("i need", "我需要"),
    ("is it", "是否"), ("are there", "是否有"),
    ("do you", "你是否"), ("should i", "我是否应该"),
    ("main reason", "主要原因"), ("key difference", "关键区别"),
    ("step by step", "逐步"),
    ("explain", "解释"), ("describe", "描述"), ("summarize", "总结"),
    ("compare", "比较"), ("analyze", "分析"), ("calculate", "计算"),
    ("determine", "确定"), ("identify", "识别"), ("evaluate", "评估"),
    ("derive", "推导"), ("prove", "证明"), ("demonstrate", "演示"),
    ("illustrate", "说明"), ("discuss", "讨论"), ("provide", "提供"),
    ("create", "创建"), ("generate", "生成"), ("write", "写出"),
    ("list", "列出"), ("classify", "分类"), ("translate", "翻译"),
    ("solve", "求解"), ("find", "找出"), ("choose", "选择"),
    ("select", "选择"), ("predict", "预测"), ("estimate", "估计"),
    ("recommend", "推荐"), ("suggest", "建议"), ("design", "设计"),
    ("implement", "实现"), ("convert", "转换"), ("rewrite", "改写"),
    ("complete", "完成"), ("answer", "回答"), ("question", "问题"),
    ("result", "结果"), ("reason", "原因"), ("difference", "区别"),
    ("similarity", "相似性"), ("relationship", "关系"),
    ("advantage", "优点"), ("disadvantage", "缺点"),
    ("example", "示例"), ("method", "方法"), ("approach", "方法"),
    ("process", "过程"), ("mechanism", "机制"), ("concept", "概念"),
    ("definition", "定义"), ("meaning", "含义"), ("purpose", "目的"),
    ("effect", "影响"), ("impact", "影响"), ("cause", "原因"),
    ("solution", "解决方案"), ("problem", "问题"), ("information", "信息"),
    ("sentence", "句子"), ("paragraph", "段落"), ("article", "文章"),
    ("story", "故事"), ("text", "文本"), ("code", "代码"),
    ("function", "函数"), ("algorithm", "算法"), ("model", "模型"),
    ("data", "数据"), ("system", "系统"), ("language", "语言"),
    ("word", "单词"), ("phrase", "短语"), ("number", "数字"),
    ("value", "值"), ("equation", "方程"), ("formula", "公式"),
    ("using", "使用"), ("given", "给定"), ("following", "以下"),
    ("briefly", "简要地"), ("clearly", "清楚地"), ("correctly", "正确地"),
    ("what", "什么"), ("how", "如何"), ("why", "为什么"),
    ("when", "何时"), ("where", "哪里"), ("who", "谁"),
    ("which", "哪个"), ("whether", "是否"), ("please", "请"),
    ("best", "最佳"), ("ways", "方式"), ("way", "方式"),
    ("common", "常见"), ("help", "帮助"), ("people", "人们"),
    ("benefits", "好处"), ("think", "思考"), ("good", "好的"),
    ("know", "知道"), ("things", "事物"), ("social", "社会的"),
    ("tips", "建议"), ("work", "工作"), ("learn", "学习"),
    ("business", "商业"), ("learning", "学习"), ("new", "新的"),
    ("improve", "改进"), ("different", "不同"), ("someone", "某人"),
    ("health", "健康"), ("need", "需要"), ("important", "重要"),
    ("world", "世界"), ("history", "历史"), ("key", "关键"),
    ("time", "时间"), ("popular", "流行"), ("media", "媒体"),
    ("human", "人类"), ("humans", "人类"), ("effective", "有效"),
    ("life", "生活"), ("differences", "差异"),
    ("successful", "成功"), ("intelligence", "智能"),
    ("mental", "心理的"), ("skills", "技能"),
    ("interesting", "有趣"), ("development", "发展"),
    ("online", "在线"), ("start", "开始"), ("person", "个人"),
    ("types", "类型"), ("affect", "影响"), ("anxiety", "焦虑"),
    ("become", "成为"), ("writing", "写作"), ("digital", "数字化"),
    ("ideas", "想法"), ("facts", "事实"), ("artificial", "人工"),
    ("industry", "行业"), ("machine", "机器"), ("advice", "建议"),
    ("importance", "重要性"), ("understand", "理解"),
    ("creating", "创建"), ("software", "软件"), ("company", "公司"),
    ("technology", "技术"), ("job", "工作"), ("change", "改变"),
    ("home", "家庭"), ("marketing", "营销"), ("creative", "创意"),
    ("potential", "潜力"), ("kind", "种类"), ("research", "研究"),
    ("examples", "示例"), ("challenges", "挑战"), ("plan", "计划"),
    ("strategies", "策略"), ("consider", "考虑"),
    ("possible", "可能"), ("experience", "经验"), ("factors", "因素"),
    ("better", "更好"), ("project", "项目"), ("trip", "旅行"),
    ("reduce", "减少"), ("computer", "计算机"), ("team", "团队"),
    ("reasons", "原因"), ("handle", "处理"), ("mistakes", "错误"),
    ("small", "小型"), ("systems", "系统"), ("practices", "实践"),
    ("personal", "个人的"), ("unique", "独特"), ("science", "科学"),
    ("future", "未来"), ("feel", "感觉"), ("planning", "规划"),
    ("building", "建设"), ("role", "作用"), ("features", "特征"),
    ("steps", "步骤"), ("fun", "有趣"), ("current", "当前"),
    ("characteristics", "特征"), ("books", "书籍"), ("market", "市场"),
    ("increase", "增加"), ("modern", "现代"), ("effects", "影响"),
    ("day", "一天"), ("build", "构建"), ("top", "顶级"),
    ("management", "管理"), ("advantages", "优势"), ("field", "领域"),
    ("strategy", "策略"), ("understanding", "理解"),
    ("students", "学生"), ("pros", "优点"), ("cons", "缺点"),
    ("game", "游戏"), ("significance", "意义"),
    ("programming", "编程"), ("career", "职业"), ("community", "社区"),
    ("climate", "气候"), ("society", "社会"), ("stress", "压力"),
    ("avoid", "避免"), ("great", "出色"), ("children", "儿童"),
    ("money", "金钱"), ("issues", "问题"), ("art", "艺术"),
    ("music", "音乐"), ("food", "食物"), ("daily", "日常"),
    ("education", "教育"), ("culture", "文化"), ("develop", "发展"),
    ("idea", "想法"), ("book", "书"), ("activities", "活动"),
    ("space", "空间"), ("brain", "大脑"), ("product", "产品"),
    ("resources", "资源"), ("content", "内容"), ("teach", "教授"),
    ("prepare", "准备"), ("support", "支持"),
    ("can", "可以"), ("and", "和"), ("about", "关于"),
    ("with", "与"), ("more", "更多"), ("but", "但是"),
    ("both", "两者"), ("together", "一起"), ("other", "其他"),
    ("another", "另一个"), ("really", "确实"), ("ever", "曾经"),
    ("own", "自己的"), ("get", "获得"), ("make", "制作"),
    ("used", "使用"), ("use", "使用"), ("talk", "谈论"),
    ("give", "给出"), ("has", "有"), ("share", "分享"),
    ("consciousness", "意识"), ("analysis", "分析"),
    ("define", "定义"), ("train", "训练"), ("cultural", "文化的"),
    ("show", "展示"), ("truly", "真正地"), ("python", "Python语言"),
    ("high", "高"), ("assist", "协助"), ("recipe", "食谱"),
    ("movie", "电影"), ("quantum", "量子"),
    ("traditional", "传统的"), ("united", "联合的"),
    ("structure", "结构"), ("prompt", "提示词"), ("theory", "理论"),
    ("environment", "环境"), ("early", "早期"),
    ("connection", "联系"), ("humanity", "人类"),
    ("universe", "宇宙"), ("turing", "图灵"), ("specific", "具体的"),
    ("university", "大学"), ("practice", "实践"), ("moral", "道德的"),
    ("cancer", "癌症"), ("thing", "事物"), ("mars", "火星"),
    ("body", "身体"), ("explanation", "解释"), ("draw", "绘制"),
    ("models", "模型"), ("investment", "投资"), ("worth", "值得"),
    ("cats", "猫"), ("weight", "重量"), ("google", "谷歌"),
    ("rights", "权利"), ("elon", "埃隆"), ("organization", "组织"),
    ("cryptocurrency", "加密货币"), ("teaching", "教学"),
    ("emotions", "情绪"), ("american", "美国的"),
    ("natural", "自然的"), ("philosophy", "哲学"),
    ("president", "总统"), ("country", "国家"), ("trust", "信任"),
    ("terms", "术语"), ("web", "网络"), ("death", "死亡"),
    ("companies", "公司"), ("government", "政府"), ("men", "男性"),
    ("path", "路径"), ("morality", "道德"), ("experiment", "实验"),
    ("empathetic", "有同理心的"), ("hair", "头发"),
    ("source", "来源"), ("family", "家庭"), ("creativity", "创造力"),
    ("professional", "专业的"), ("changes", "变化"),
    ("stories", "故事"), ("explore", "探索"),
    ("chatbot", "聊天机器人"), ("travel", "旅行"),
    ("pregnancy", "怀孕"), ("accurate", "准确的"),
    ("program", "程序"), ("cognitive", "认知的"), ("novel", "小说"),
    ("coffee", "咖啡"), ("craft", "工艺"), ("alignment", "对齐"),
    ("poetry", "诗歌"), ("intelligent", "智能的"),
    ("outline", "大纲"), ("poem", "诗"), ("custom", "自定义"),
    ("rank", "排名"), ("empire", "帝国"), ("significant", "重要的"),
    ("ukraine", "乌克兰"), ("buy", "购买"), ("plot", "情节"),
    ("differentiate", "区分"), ("robot", "机器人"),
    ("trained", "训练过的"), ("dark", "黑暗的"), ("script", "脚本"),
    ("africa", "非洲"), ("created", "创建的"),
    ("improved", "改进的"), ("pay", "支付"), ("medical", "医疗的"),
    ("exist", "存在"), ("password", "密码"), ("assistants", "助手"),
    ("goal", "目标"), ("anything", "任何事"), ("sugar", "糖"),
    ("handling", "处理"), ("dementia", "痴呆"), ("techniques", "技术"),
    ("raise", "提高"), ("email", "电子邮件"), ("cat", "猫"),
    ("animals", "动物"), ("adhd", "注意缺陷多动障碍"),
    ("conscious", "有意识的"), ("diet", "饮食"),
    ("comprehensive", "全面的"), ("file", "文件"),
    ("解释", "explain"), ("描述", "describe"), ("总结", "summarize"),
    ("比较", "compare"), ("分析", "analyze"), ("计算", "calculate"),
    ("问题", "question"), ("结果", "result"), ("原因", "reason"),
    ("方法", "method"), ("示例", "example"), ("代码", "code"),
    ("数据", "data"), ("模型", "model"), ("系统", "system"),
)


@dataclass(frozen=True)
class LocalEdit:
    family: str
    subtype: str
    candidate_text: str
    span_start: int
    span_end: int
    original: str
    replacement: str
    seed: int

    def __post_init__(self) -> None:
        if self.family not in FAMILY_ORDER:
            raise ValueError(f"unknown metamorphic family: {self.family}")
        if not (
            0 <= self.span_start <= self.span_end
            and self.span_start + len(self.replacement) <= len(self.candidate_text)
        ):
            raise ValueError("invalid modified span")
        if (
            self.candidate_text[
                self.span_start : self.span_start + len(self.replacement)
            ]
            != self.replacement
        ):
            raise ValueError("candidate text does not contain declared replacement")
        rebuilt = (
            self.candidate_text[: self.span_start]
            + self.original
            + self.candidate_text[self.span_start + len(self.replacement) :]
        )
        if rebuilt == self.candidate_text:
            raise ValueError("metamorphic edit must change text")

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "subtype": self.subtype,
            "candidate_text": self.candidate_text,
            "modified_span": {
                "start": self.span_start,
                "end": self.span_end,
                "original": self.original,
                "replacement": self.replacement,
            },
            "seed": self.seed,
        }


def _merge_spans(spans: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[list[int]] = []
    for start, end in sorted((int(a), int(b)) for a, b in spans if b > a):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


def protected_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Return fail-closed spans for constraints, code, formulas, and quotes."""

    patterns = (
        _NEGATION, _QUANTITY, _URL, _SPECIAL, _NUMBER, _FENCED_CODE,
        _INLINE_CODE, _MATH, _JSON_KEY, _OPTION_LABEL, _FORMULA,
    )
    spans = [match.span() for pattern in patterns for match in pattern.finditer(text)]
    content_start = len(text) - len(text.lstrip())
    content_end = len(text.rstrip())
    for match in _QUOTED.finditer(text):
        # Magpie contains questions wrapped wholesale in quotation marks.  The
        # wrapper is formatting, not quoted material embedded in an instruction.
        if match.start() == content_start and match.end() == content_end:
            continue
        spans.append(match.span())
    return _merge_spans(spans)


def _overlaps(span: tuple[int, int], protected: Sequence[tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in protected)


def _eligible_words(text: str) -> list[re.Match[str]]:
    protected = protected_spans(text)
    return [
        match
        for match in _WORD.finditer(text)
        if not _overlaps(match.span(), protected)
        and match.group(0).lower() not in _SEMANTICALLY_PROTECTED_WORDS
    ]


def _apply(text: str, start: int, end: int, replacement: str) -> str:
    return text[:start] + replacement + text[end:]


def _content_start(text: str) -> int:
    return len(text) - len(text.lstrip())


def _ordered(items: Sequence[object], seed: int) -> list[object]:
    values = list(items)
    random.Random(seed).shuffle(values)
    return values


def character_noise_candidate(text: str, *, subtype: str, seed: int) -> LocalEdit:
    if subtype not in CHARACTER_SUBTYPES:
        raise ValueError(f"unsupported character-noise subtype: {subtype}")
    words = _ordered(_eligible_words(text), seed)
    for value in words:
        match = value
        assert isinstance(match, re.Match)
        word = match.group(0)
        replacement: str | None = None
        if subtype == "adjacent_swap":
            positions = [
                index for index in range(1, len(word) - 2)
                if word[index].lower() != word[index + 1].lower()
            ]
            if positions:
                index = positions[seed % len(positions)]
                replacement = word[:index] + word[index + 1] + word[index] + word[index + 2 :]
        elif subtype == "repeat_or_insert_character":
            index = 1 + seed % max(1, len(word) - 2)
            replacement = word[:index] + word[index] + word[index:]
        elif subtype == "non_language_symbol":
            index = 1 + seed % max(1, len(word) - 1)
            symbol = ("*", "~", "^")[seed % 3]
            replacement = word[:index] + symbol + word[index:]
        else:
            positions = [
                index for index, character in enumerate(word)
                if character.lower() in _VISUAL_MAP
            ]
            if positions:
                index = positions[seed % len(positions)]
                visual = _VISUAL_MAP[word[index].lower()]
                replacement = word[:index] + visual + word[index + 1 :]
        if replacement is None or replacement == word:
            continue
        candidate = _apply(text, match.start(), match.end(), replacement)
        return LocalEdit(
            FAMILY_CHARACTER_NOISE, subtype, candidate, match.start(), match.end(),
            word, replacement, seed,
        )
    if text.strip():
        # A formula-only or fully quoted question has no safe character to
        # corrupt.  Add one boundary noise symbol without touching its content.
        start = _content_start(text)
        replacement = ("*", "~", "^")[seed % 3]
        return LocalEdit(
            FAMILY_CHARACTER_NOISE, "non_language_symbol", _apply(text, start, start, replacement),
            start, start, "", replacement, seed,
        )
    raise ValueError(f"no safe position for character-noise subtype {subtype}")


def tokenization_candidate(text: str, *, subtype: str, seed: int) -> LocalEdit:
    if subtype not in TOKENIZATION_SUBTYPES:
        raise ValueError(f"unsupported tokenization subtype: {subtype}")
    words = _ordered([m for m in _eligible_words(text) if len(m.group(0)) >= 3], seed)
    for value in words:
        match = value
        assert isinstance(match, re.Match)
        word = match.group(0)
        index = 2 + seed % max(1, len(word) - 3)
        delimiter = {
            "intra_word_space": " ",
            "punctuation_delimiter": (".", "·", "/")[seed % 3],
            "visual_split": "\u200b",
        }[subtype]
        replacement = word[:index] + delimiter + word[index:]
        candidate = _apply(text, match.start(), match.end(), replacement)
        return LocalEdit(
            FAMILY_TOKENIZATION, subtype, candidate, match.start(), match.end(),
            word, replacement, seed,
        )
    if text.strip():
        # Preserve protected content while still changing tokenizer input.
        start = _content_start(text)
        return LocalEdit(
            FAMILY_TOKENIZATION, "visual_split", _apply(text, start, start, "\u200b"),
            start, start, "", "\u200b", seed,
        )
    raise ValueError(f"no safe position for tokenization subtype {subtype}")


def code_switch_candidate(
    text: str,
    *,
    seed: int,
    lexicon: Sequence[tuple[str, str]] = DEFAULT_CODE_SWITCH_LEXICON,
) -> LocalEdit:
    protected = protected_spans(text)
    matches: list[tuple[int, int, str, str, int]] = []
    for source, target in lexicon:
        if not source or not target:
            continue
        if _CJK_WORD.fullmatch(source):
            pattern = re.compile(re.escape(source))
        else:
            pattern = re.compile(r"(?<![A-Za-z])" + re.escape(source) + r"(?![A-Za-z])", re.IGNORECASE)
        for match in pattern.finditer(text):
            if not _overlaps(match.span(), protected):
                matches.append(
                    (match.start(), match.end(), match.group(0), target, len(source.split()))
                )
    if not matches:
        if not text.strip():
            raise ValueError("no safe phrase for local code-switching")
        # A bilingual politeness cue is a conservative fallback for formula-only
        # and long-tail questions: it changes style, not intent or constraints.
        start = _content_start(text)
        replacement = "Please " if re.search(r"[\u3400-\u9fff]", text) else "请 "
        return LocalEdit(
            FAMILY_CODE_SWITCH, "bilingual_politeness_cue",
            _apply(text, start, start, replacement), start, start, "", replacement, seed,
        )
    longest = max(match[4] for match in matches)
    start, end, original, replacement, _ = _ordered(
        [match for match in matches if match[4] == longest], seed
    )[0]
    assert isinstance(start, int) and isinstance(end, int)
    candidate = _apply(text, start, end, replacement)
    return LocalEdit(
        FAMILY_CODE_SWITCH, "semantic_equivalent_translation", candidate,
        start, end, original, replacement, seed,
    )


def generate_family_edit(
    text: str,
    *,
    family: str,
    subtype: str,
    seed: int,
) -> LocalEdit:
    if family == FAMILY_CHARACTER_NOISE:
        return character_noise_candidate(text, subtype=subtype, seed=seed)
    if family == FAMILY_TOKENIZATION:
        return tokenization_candidate(text, subtype=subtype, seed=seed)
    if family == FAMILY_CODE_SWITCH:
        if subtype != "semantic_equivalent_translation":
            raise ValueError(f"unsupported code-switch subtype: {subtype}")
        return code_switch_candidate(text, seed=seed)
    raise ValueError(f"unknown metamorphic family: {family}")


def validate_single_local_edit(original_text: str, edit: LocalEdit) -> None:
    """Prove that metadata describes exactly one replacement in one question."""

    if edit.original != original_text[edit.span_start : edit.span_end]:
        raise ValueError("edit original does not match question span")
    expected = _apply(
        original_text, edit.span_start, edit.span_end, edit.replacement
    )
    if edit.candidate_text != expected:
        raise ValueError("candidate contains more than the declared local edit")
    if _overlaps((edit.span_start, edit.span_end), protected_spans(original_text)):
        raise ValueError("candidate modifies a protected span")
    if not edit.replacement or edit.replacement == edit.original:
        raise ValueError("candidate replacement must be non-empty and different")


__all__ = [
    "CHARACTER_SUBTYPES", "DEFAULT_CODE_SWITCH_LEXICON", "FAMILY_CHARACTER_NOISE",
    "FAMILY_CODE_SWITCH", "FAMILY_ORDER", "FAMILY_TOKENIZATION", "LocalEdit",
    "TOKENIZATION_SUBTYPES", "character_noise_candidate", "code_switch_candidate",
    "generate_family_edit", "protected_spans", "tokenization_candidate",
    "validate_single_local_edit",
]

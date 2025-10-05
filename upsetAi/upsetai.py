
"""
Upset Detector by Ksenivu — rule-based tone & action suggester (no external deps).
Library:
  from upset_detector import analyze_message, analyze_conversation
"""

import argparse
import json
import re
from typing import List, Dict, Any, Optional

#  Bad words & patterns

NEG_EMO_WORDS = {
    "whatever", "idc", "dont care", "don’t care", "whatever.", "fine.", "ok.", "k.",
    "k", "sure.", "great.", "cool.", "as you wish", "do what you want", "you do you",
    "it’s fine", "its fine", "i’m fine", "im fine", "i’m okay", "im okay", "okay.",
    "forget it", "leave me alone", "do whatever", "doesn’t matter", "doesnt matter"
}

HURT_CUES = {
    "you never", "you always", "you didn’t", "you didnt", "you wouldn’t", "you wouldnt",
    "you said", "after what you did", "how could you", "that hurt", "i don’t matter",
    "i dont matter"
}

DRY_SINGLETONS = {"k", "ok", "okay", "fine", "whatever"}
SOFTENERS = {"maybe", "perhaps", "could", "might", "i feel", "from my side", "i think"}
EMOJIS_SAD = {"😔", "😞", "😢", "😭", "🙃", "🥲", "😑", "😒", "👎"}
EMOJIS_ANGRY = {"😠", "😡", "💢"}
ELLIPSIS = "..."


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def count_emojis(text: str) -> int:
    return sum(ch in EMOJIS_SAD or ch in EMOJIS_ANGRY for ch in text)

def is_short_dry(text: str) -> bool:
    t = norm(text)
    # one/two-word curt replies or single-letter K.
    words = t.split()
    if len(words) <= 2:
        if t in DRY_SINGLETONS or t.endswith(".") or t == "k":
            return True
    # single punctuation-only
    if re.fullmatch(r"[.!?]+", t):
        return True
    return False

def negations_present(text: str) -> bool:
    return bool(re.search(r"\b(no|not|don’t|dont|never|nothing|noway)\b", norm(text)))

def contains_any(text: str, bag: set) -> bool:
    t = norm(text)
    return any(phrase in t for phrase in bag)

def missing_softeners(text: str) -> bool:
    t = norm(text)
    return not any(w in t for w in SOFTENERS)

def ellipsis_like(text: str) -> bool:
    return ELLIPSIS in text or re.search(r"\.\s*\.\s*\.", text) is not None

def repeated_punct(text: str) -> bool:
    return bool(re.search(r"([!?])\1{1,}", text))

def hostility_score(text: str) -> float:
    t = norm(text)
    score = 0.0
    if contains_any(t, HURT_CUES): score += 0.45
    if negations_present(t): score += 0.15
    if repeated_punct(t): score += 0.15
    if count_emojis(text) > 0: score += 0.1
    if ellipsis_like(text): score += 0.08
    if contains_any(t, NEG_EMO_WORDS): score += 0.25
    if is_short_dry(text): score += 0.22
    if missing_softeners(text): score += 0.05
    return min(score, 1.0)

# API

def analyze_message(text: str) -> Dict[str, Any]:
    """
    Analyze a single message string.

    """
    s = hostility_score(text)
    signals = []

    if contains_any(text, NEG_EMO_WORDS):
        signals.append("phrases suggesting withdrawal (‘fine’, ‘whatever’, ‘I’m okay’).")
    if is_short_dry(text):
        signals.append("short/dry reply.")
    if contains_any(text, HURT_CUES):
        signals.append("hurt/accusatory cues (‘you never’, ‘after what you did’).")
    if repeated_punct(text):
        signals.append("repeated punctuation (emotionally charged).")
    if ellipsis_like(text):
        signals.append("ellipsis (distance/withholding).")
    if count_emojis(text) > 0:
        signals.append("sad/angry emoji present.")
    if missing_softeners(text):
        signals.append("no softening language.")

    is_upset = s >= 0.5
    reason = (
        "Message contains multiple upset signals."
        if is_upset else
        "Few or weak upset signals detected."
    )
    action = suggest_action(text, signals, confidence=s)

    return {
        "is_upset": is_upset,
        "confidence": round(float(s), 2),
        "signals": signals,
        "reason": reason,
        "suggested_action": action
    }


def analyze_conversation(messages: List[Dict[str, str]],
                         focus_author: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze a short conversation.

    """
    if focus_author:
        filtered = [m for m in messages if m.get("author") == focus_author]
    else:
        filtered = messages

    if not filtered:
        return {
            "is_upset": False,
            "confidence": 0.0,
            "signals": [],
            "reason": "No messages to analyze.",
            "suggested_action": "Ask a gentle, open question to invite sharing."
        }


    weights = []
    scores = []
    all_signals = []

    for idx, m in enumerate(filtered):
        w = 0.6 + 0.4 * (idx + 1) / max(1, len(filtered))
        weights.append(w)
        res = analyze_message(m["text"])
        scores.append(res["confidence"])
        all_signals.extend(res["signals"])

    # Weighted average confidence
    total_w = sum(weights)
    avg = sum(s * w for s, w in zip(scores, weights)) / max(total_w, 1e-9)
    is_upset = avg >= 0.5
    reason = "Recent messages tilt upset." if is_upset else "Tone mostly neutral."

    # Suggest based on last focused message
    last_text = filtered[-1]["text"]
    action = suggest_action(last_text, all_signals, confidence=avg)

    return {
        "is_upset": is_upset,
        "confidence": round(avg, 2),
        "signals": list(dict.fromkeys(all_signals)),  
        "reason": reason,
        "suggested_action": action
    }


def suggest_action(text: str, signals: List[str], confidence: float) -> str:
    t = norm(text)
    high = confidence >= 0.7
    med = 0.55 <= confidence < 0.7

    def own_and_apologize():
        return ("I’m sorry I let you down there. I care about how you feel, "
                "and I want to make it right. What would help right now?")

    def acknowledge_and_invite():
        return ("I may have missed something. Your feelings matter to me—"
                "do you want to talk about it? I’m listening.")

    def offer_space():
        return ("I can sense this is heavy. I’m here for you, and I’ll give you a bit "
                "of space if you need it. Message me when you’re ready.")

    def clarify_gently():
        return ("I want to understand you better. Could you help me see what I missed? "
                "I’m asking so I can do better.")

    # Heuristic routing
    if contains_any(t, HURT_CUES) or "hurt/accusatory" in " ".join(signals):
        return own_and_apologize()
    if is_short_dry(text) or contains_any(t, NEG_EMO_WORDS):
        return acknowledge_and_invite() if med or high else clarify_gently()
    if repeated_punct(text) or count_emojis(text) > 0:
        return acknowledge_and_invite()
    if ellipsis_like(text):
        return offer_space() if med or high else clarify_gently()
    # default
    return clarify_gently()


# CLI

def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip("\n") for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Upset Detector CLI")
    parser.add_argument("--text", type=str, help="Single message to analyze")
    parser.add_argument("--from-file", type=str, help="Path to a txt file with one message per line")
    parser.add_argument("--role", type=str, default="sender",
                        help="Whose messages the file represents (e.g., 'sender' or 'her')")
    parser.add_argument("--json", action="store_true", help="Print raw JSON only")
    args = parser.parse_args()

    if args.text and args.from_file:
        print("Provide either --text or --from-file, not both.")
        return
    if not args.text and not args.from_file:
        print("Usage: --text 'message'  OR  --from-file chat.txt")
        return

    if args.text:
        res = analyze_message(args.text)
    else:
        lines = _read_lines(args.from_file)
        convo = [{"text": m, "author": args.role} for m in lines]
        res = analyze_conversation(convo, focus_author=args.role)

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print("is_upset:", res["is_upset"])
        print("confidence:", res["confidence"])
        if res["signals"]:
            print("signals:")
            for s in res["signals"]:
                print("  -", s)
        print("reason:", res["reason"])
        print("suggested_action:", res["suggested_action"])

if __name__ == "__main__":
    def main():
        #Открываем текстовый файл и читаем переписку
        with open("chat.txt", "r", encoding="utf-8") as f:
            chat_text = f.read()

        #Отправляем текст на анализ
        result = analyze_message(chat_text)

        #Выводим результат анализа
        print("is_upset:", result["is_upset"])
        print("confidence:", result["confidence"])
        print("signals:", result["signals"])
        print("reason:", result["reason"])
        print("suggested_action:", result["suggested_action"])


    if __name__ == "__main__":
        main()

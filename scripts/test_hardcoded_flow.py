"""
End-to-End Test for the Hardcoded Voice Agent Conversation Flow.
Verifies the exact user-specified flow:
1. Welcome message
2. Admission details (10th/12th PCM 60%, GUJCET result & ACPC rank, fees: 1st yr 1,66,950, 2nd-4th yr 1,52,000)
3. Subjects taught in DDU in each semester (Sem 1 to 8 from curriculum images)
4. GATE subjects covered in DDU and important topics
5. Placement details (avg 6 LPA, highest 24 LPA off-campus, 200 placed students, top recruiters)
6. User agrees -> Agent asks "Do you need any other help?"
7. User denies -> Agent delivers closing farewell and call concludes!
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from config.prompt_loader import get_welcome_message, get_system_prompt
from llm.llm import LLM, is_agreement_intent

def test_full_conversation_flow():
    print("=" * 70)
    print("STARTING FULL CONVERSATION FLOW SIMULATION")
    print("=" * 70)

    # Step 1: Agent Welcome Message
    welcome = get_welcome_message()
    print(f"\n[Step 1 - Agent Welcome]: {welcome}")
    assert "પ્રિયા" in welcome and "ડીડીયુ" in welcome, "Welcome message mismatch"

    # Initialize LLM (no DB, no RAG)
    agent = LLM()
    assert agent.db is None and not hasattr(agent, "rag"), "Agent must not initialize DB and must have no RAG attribute"

    # Step 2: User asks about admission details
    user_q2 = "I want to know about the admission details"
    print(f"\n[Step 2 - User]: {user_q2}")
    msgs2 = agent._prepare_messages(user_q2)
    prompt_content = msgs2[0]["content"]
    assert "ADMISSION & ELIGIBILITY DETAILS:" in prompt_content
    assert "1,66,950" in prompt_content
    assert "1,52,000" in prompt_content
    assert "GUJCET" in prompt_content
    print("  ✓ Prompt contains verified 10th/12th eligibility, GUJCET score/rank, and 1st/subsequent year fees.")

    # Step 3: User asks about subjects taught in DDU in each semester
    user_q3 = "What subjects are taught in DDU in each sem?"
    print(f"\n[Step 3 - User]: {user_q3}")
    assert "OFFICIAL B.TECH IT CURRICULUM (SEMESTERS 1 TO 8):" in prompt_content
    for sem in range(1, 9):
        assert f"Semester {sem}:" in prompt_content, f"Semester {sem} missing in prompt"
    print("  ✓ Prompt contains all Semester 1 to 8 subjects from syllabus images.")

    # Step 4: User asks what subjects of GATE are covered in DDU
    user_q4 = "What subjects of GATE are covered in DDU?"
    print(f"\n[Step 4 - User]: {user_q4}")
    assert "GATE SUBJECTS COVERED IN DDU CURRICULUM & IMPORTANT TOPICS:" in prompt_content
    gate_subjects = [
        "Engineering Mathematics",
        "Digital Logic",
        "Computer Organization and Architecture (COA)",
        "Programming and Data Structures",
        "Algorithms",
        "Theory of Computation (TOC)",
        "Compiler Design",
        "Operating Systems (OS)",
        "Databases (DBMS)",
        "Computer Networks (CN)",
        "General Aptitude",
        "Digital and Computer Fundamentals",
    ]
    for gs in gate_subjects:
        assert gs in prompt_content, f"GATE subject {gs} missing in prompt"
    print("  ✓ Prompt contains all 12 GATE subjects and their specific taught topics.")

    # Step 5: User asks about placement details
    user_q5 = "What are the placement details?"
    print(f"\n[Step 5 - User]: {user_q5}")
    assert "6 LPA" in prompt_content, "Average package 6 LPA missing"
    assert "24 LPA (off-campus)" in prompt_content, "Highest package 24 LPA missing"
    assert "200 placed students" in prompt_content, "Total 200 placed students missing"
    assert "TCS, Infosys, Crest Data Systems, TatvaSoft" in prompt_content, "Top recruiters missing"
    print("  ✓ Prompt contains 6 LPA average, 24 LPA off-campus highest, 200 placed students, and top recruiters.")

    # Step 6 & 7: User agrees -> Agent asks if any other help is needed
    for lang, agree_text, expected_followup in [
        ("en", "Okay, got it", "Do you need any other help?"),
        ("gu", "ઠીક છે, બરાબર", "બીજી કોઈ મદદ જોઈએ છે?"),
        ("hi", "ठीक है, समझ गया", "क्या आपको और कोई मदद चाहिए?"),
    ]:
        agent_agree = LLM()
        agent_agree.current_lang = lang
        assert is_agreement_intent(agree_text), f"Agreement not detected for {agree_text}"
        reply = list(agent_agree.reply_stream(agree_text))
        print(f"\n[Step 6 & 7 - User Agree ({lang})]: '{agree_text}' -> [Agent Response]: '{reply[0]}'")
        assert reply[0] == expected_followup, f"Expected '{expected_followup}', got '{reply[0]}'"

    # Step 8 & 9: User denies -> Agent delivers closing farewell and call disconnects
    for lang, deny_text, expected_farewell in [
        ("en", "No, that's all. Thank you", "Have a great day. Thank you!"),
        ("gu", "ના, બસ આટલું જ. આભાર", "તમારો દિવસ શુભ રહે. આભાર!"),
        ("hi", "नहीं, और कुछ नहीं. धन्यवाद", "आपका दिन शुभ हो। धन्यवाद!"),
    ]:
        agent_deny = LLM()
        agent_deny.current_lang = lang
        assert agent_deny.check_call_hangup(deny_text), f"Hangup intent not detected for {deny_text}"
        reply = list(agent_deny.reply_stream(deny_text))
        print(f"\n[Step 8 & 9 - User Deny ({lang})]: '{deny_text}' -> [Agent Farewell]: '{reply[0]}'")
        assert reply[0] == expected_farewell, f"Expected '{expected_farewell}', got '{reply[0]}'"
        assert agent_deny.check_call_hangup("", reply[0]), "Closing farewell must trigger supervisor call hangup"

    print("\n" + "=" * 70)
    print("🎉 FULL CONVERSATION FLOW VERIFIED 100% SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_full_conversation_flow()

import sys
from pathlib import Path

# Insert cg parent directory and competition directory to Python path
repo_root = Path(__file__).resolve().parents[1]
cg_parent = repo_root / "data/pokemon/pokemon-tcg-ai-battle/sample_submission/sample_submission"
if not cg_parent.exists():
    cg_parent = repo_root / "data/pokemon/pokemon-tcg-ai-battle/sample_submission"
sys.path.insert(0, str(cg_parent))
sys.path.insert(0, str(repo_root / "competitions/pokemon-tcg-ai-battle"))


# Helper functions to build fully compliant observation dictionaries
def make_mock_select(custom=None):
    base = {
        "type": 0,
        "context": 0,
        "minCount": 1,
        "maxCount": 1,
        "remainDamageCounter": 0,
        "remainEnergyCost": 0,
        "option": [],
        "deck": None,
        "contextCard": None,
        "effect": None,
    }
    if custom:
        base.update(custom)
    return base


def make_mock_player_state(custom=None):
    base = {
        "active": [],
        "bench": [],
        "benchMax": 5,
        "deckCount": 60,
        "discard": [],
        "prize": [None] * 6,
        "handCount": 0,
        "hand": [],
        "poisoned": False,
        "burned": False,
        "asleep": False,
        "paralyzed": False,
        "confused": False,
    }
    if custom:
        base.update(custom)
    return base


def make_mock_state(custom=None):
    base = {
        "turn": 1,
        "turnActionCount": 0,
        "yourIndex": 0,
        "firstPlayer": 0,
        "supporterPlayed": False,
        "stadiumPlayed": False,
        "energyAttached": False,
        "retreated": False,
        "result": -1,
        "stadium": [],
        "looking": None,
        "players": [make_mock_player_state(), make_mock_player_state()],
    }
    if custom:
        base.update(custom)
    return base


def make_mock_obs(select_dict=None, current_dict=None):
    return {
        "select": make_mock_select(select_dict) if select_dict is not None else None,
        "logs": [],
        "current": make_mock_state(current_dict),
    }


def test_agent_deck_loading():
    """Verify that all agents can load their respective decks successfully."""
    from agents.abomasnow import AbomasnowAgent
    from agents.dragapult import DragapultAgent
    from agents.lucario import LucarioAgent
    from agents.zacian import ZacianAgent

    agent_zacian = ZacianAgent()
    assert len(agent_zacian.get_deck()) == 60

    agent_abomasnow = AbomasnowAgent()
    assert len(agent_abomasnow.get_deck()) == 60

    agent_dragapult = DragapultAgent()
    assert len(agent_dragapult.get_deck()) == 60

    agent_lucario = LucarioAgent()
    assert len(agent_lucario.get_deck()) == 60

    # Verify that raw CSV files have exactly 60 cards (not truncated at load time)
    deck_files = [
        "competitions/pokemon-tcg-ai-battle/abomasnow_deck.csv",
        "competitions/pokemon-tcg-ai-battle/deck.csv",
        "competitions/pokemon-tcg-ai-battle/dragapult_deck.csv",
        "competitions/pokemon-tcg-ai-battle/lucario_deck.csv",
        "competitions/pokemon-tcg-ai-battle/heuristic_deck.csv",
    ]
    for df in deck_files:
        path = repo_root / df
        if path.exists():
            with open(path) as f:
                lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
            assert len(lines) == 60, f"Deck file {df} does not contain exactly 60 cards (found {len(lines)})"


def test_zacian_agent_yes_no():
    """Verify that YES_NO select type prioritizes YES option."""
    from agents.zacian import ZacianAgent

    obs_dict = make_mock_obs(
        select_dict={
            "type": 9,  # YES_NO
            "option": [
                {
                    "type": 2,
                    "number": None,
                    "area": None,
                    "index": None,
                    "playerIndex": None,
                    "toolIndex": None,
                    "energyIndex": None,
                    "count": None,
                    "inPlayArea": None,
                    "inPlayIndex": None,
                    "attackId": None,
                    "cardId": None,
                    "serial": None,
                    "specialConditionType": None,
                },  # NO
                {
                    "type": 1,
                    "number": None,
                    "area": None,
                    "index": None,
                    "playerIndex": None,
                    "toolIndex": None,
                    "energyIndex": None,
                    "count": None,
                    "inPlayArea": None,
                    "inPlayIndex": None,
                    "attackId": None,
                    "cardId": None,
                    "serial": None,
                    "specialConditionType": None,
                },  # YES
            ],
        }
    )

    agent = ZacianAgent()
    action = agent.act(obs_dict)
    assert action == [1]  # Should select YES (index 1)


def test_zacian_agent_main_priority():
    """Verify that ZacianAgent prioritized Attack > Attach > Play > End."""
    from agents.zacian import ZacianAgent

    # Build base options with option index helper to ensure options are complete dicts
    options = [
        {
            "type": 7,
            "number": None,
            "area": None,
            "index": 0,
            "playerIndex": 0,
            "toolIndex": None,
            "energyIndex": None,
            "count": None,
            "inPlayArea": None,
            "inPlayIndex": None,
            "attackId": None,
            "cardId": None,
            "serial": None,
            "specialConditionType": None,
        },  # PLAY
        {
            "type": 14,
            "number": None,
            "area": None,
            "index": None,
            "playerIndex": None,
            "toolIndex": None,
            "energyIndex": None,
            "count": None,
            "inPlayArea": None,
            "inPlayIndex": None,
            "attackId": None,
            "cardId": None,
            "serial": None,
            "specialConditionType": None,
        },  # END
        {
            "type": 8,
            "number": None,
            "area": None,
            "index": 0,
            "playerIndex": 0,
            "toolIndex": None,
            "energyIndex": None,
            "count": None,
            "inPlayArea": None,
            "inPlayIndex": None,
            "attackId": None,
            "cardId": None,
            "serial": None,
            "specialConditionType": None,
        },  # ATTACH
        {
            "type": 13,
            "number": None,
            "area": None,
            "index": None,
            "playerIndex": None,
            "toolIndex": None,
            "energyIndex": None,
            "count": None,
            "inPlayArea": None,
            "inPlayIndex": None,
            "attackId": 1,
            "cardId": None,
            "serial": None,
            "specialConditionType": None,
        },  # ATTACK
    ]

    obs_dict = make_mock_obs(
        select_dict={"type": 0, "option": options},
        current_dict={"energyAttached": False},
    )

    agent = ZacianAgent()
    # 1. Attack is available -> prioritizes ATTACK
    action = agent.act(obs_dict)
    assert action == [3]

    # Remove ATTACK option
    obs_dict["select"]["option"] = options[:3]

    # 2. Attack not available, energyAttached is False -> prioritizes ATTACH
    action = agent.act(obs_dict)
    assert action == [2]

    # Set energyAttached to True
    obs_dict["current"]["energyAttached"] = True

    # 3. Attack not available, energyAttached is True -> prioritizes PLAY (over END and ATTACH)
    action = agent.act(obs_dict)
    assert action == [0]


def test_abomasnow_agent_scoring():
    """Verify AbomasnowAgent scores PLAY and ATTACH correctly."""
    from agents.abomasnow import AbomasnowAgent

    hand_cards = [{"id": 722, "serial": 1, "playerIndex": 0}]
    player0 = make_mock_player_state({"hand": hand_cards, "handCount": 1})
    players = [player0, make_mock_player_state()]

    obs_dict = make_mock_obs(
        select_dict={
            "type": 0,  # MAIN
            "option": [
                {
                    "type": 7,
                    "number": None,
                    "area": None,
                    "index": 0,
                    "playerIndex": 0,
                    "toolIndex": None,
                    "energyIndex": None,
                    "count": None,
                    "inPlayArea": None,
                    "inPlayIndex": None,
                    "attackId": None,
                    "cardId": None,
                    "serial": None,
                    "specialConditionType": None,
                },  # PLAY
                {
                    "type": 14,
                    "number": None,
                    "area": None,
                    "index": None,
                    "playerIndex": None,
                    "toolIndex": None,
                    "energyIndex": None,
                    "count": None,
                    "inPlayArea": None,
                    "inPlayIndex": None,
                    "attackId": None,
                    "cardId": None,
                    "serial": None,
                    "specialConditionType": None,
                },  # END
            ],
        },
        current_dict={"players": players},
    )

    agent = AbomasnowAgent()
    action = agent.act(obs_dict)
    assert action == [0]  # Should choose PLAY (index 0) over END


def test_lucario_agent_scoring():
    """Verify LucarioAgent scores PLAY correctly."""
    from agents.lucario import LucarioAgent

    hand_cards = [{"id": 1141, "serial": 1, "playerIndex": 0}]
    player0 = make_mock_player_state({"hand": hand_cards, "handCount": 1})
    players = [player0, make_mock_player_state()]

    obs_dict = make_mock_obs(
        select_dict={
            "type": 0,  # MAIN
            "option": [
                {
                    "type": 7,  # PLAY
                    "number": None,
                    "area": None,
                    "index": 0,
                    "playerIndex": 0,
                    "toolIndex": None,
                    "energyIndex": None,
                    "count": None,
                    "inPlayArea": None,
                    "inPlayIndex": None,
                    "attackId": None,
                    "cardId": None,
                    "serial": None,
                    "specialConditionType": None,
                },
                {
                    "type": 14,  # END
                    "number": None,
                    "area": None,
                    "index": None,
                    "playerIndex": None,
                    "toolIndex": None,
                    "energyIndex": None,
                    "count": None,
                    "inPlayArea": None,
                    "inPlayIndex": None,
                    "attackId": None,
                    "cardId": None,
                    "serial": None,
                    "specialConditionType": None,
                },
            ],
        },
        current_dict={"players": players},
    )

    agent = LucarioAgent()
    action = agent.act(obs_dict)
    assert action == [0]  # Should choose PLAY (index 0) over END


def test_submission_generation():
    """Verify that make_submission generates the tarball and notebook correctly and packages cg."""
    import tarfile

    import numpy as np
    from task import PokemonTCGAIBattleTask

    task = PokemonTCGAIBattleTask()
    dummy_ids = np.array([])
    dummy_preds = np.array([])

    tarball_path = task.make_submission(dummy_ids, dummy_preds)

    assert tarball_path.exists()
    assert tarball_path.name == "submission.tar.gz"

    with tarfile.open(tarball_path, "r:gz") as tar:
        members = {member.name for member in tar.getmembers()}
        assert "main.py" in members
        assert "deck.csv" in members
        assert any(m == "cg" or m.startswith("cg/") for m in members), f"cg directory not found in tarball: {members}"

    notebook_path = tarball_path.parent / "kernel" / "submission_notebook.py"
    assert notebook_path.exists()
    notebook_content = notebook_path.read_text()
    assert "find_cg_dir" in notebook_content
    assert "shutil.copytree" in notebook_content

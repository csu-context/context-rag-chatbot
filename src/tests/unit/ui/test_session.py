from unittest.mock import patch

from src.ui.session import clear_session_val, get_session_val, pop_session_val, set_session_val


def test_session_helpers():
    with patch("src.ui.session.st") as mock_st:

        class MockSessionState(dict):
            pass

        state_dict = MockSessionState()
        mock_st.session_state = state_dict

        # Test get_session_val
        assert get_session_val("missing", "default") == "default"

        # Test set_session_val
        set_session_val("key1", "val1")
        assert state_dict["key1"] == "val1"
        assert get_session_val("key1") == "val1"

        # Test pop_session_val
        assert pop_session_val("key1") == "val1"
        assert "key1" not in state_dict
        assert pop_session_val("key1", "def") == "def"

        # Test clear_session_val
        state_dict["key2"] = "val2"
        clear_session_val("key2")
        assert "key2" not in state_dict

        # Test clear missing key
        clear_session_val("key3")

"""API Schemas 单元测试"""
from app.schemas.auth import RegisterRequest, LoginRequest, TokenResponse
from app.schemas.user import UserResponse, UserProfileResponse
from app.schemas.conversation import ConversationCreate, ConversationUpdate, ConversationResponse
from app.schemas.message import MessageResponse
from app.schemas.chat import ChatStreamRequest


class TestAuthSchemas:
    def test_register_request_valid(self):
        body = RegisterRequest(username="testuser", email="test@example.com", password="123456")
        assert body.username == "testuser"
        assert body.email == "test@example.com"

    def test_register_request_short_username(self):
        try:
            RegisterRequest(username="ab", email="test@example.com", password="123456")
            assert False, "Should have raised validation error"
        except Exception:
            pass

    def test_login_request(self):
        body = LoginRequest(username="testuser", password="123456")
        assert body.username == "testuser"

    def test_token_response(self):
        token = TokenResponse(access_token="eyJxxx", token_type="bearer", expires_in=604800)
        assert token.access_token == "eyJxxx"

    def test_register_request_invalid_email(self):
        try:
            RegisterRequest(username="testuser", email="not-an-email", password="123456")
            assert False, "Should have raised validation error"
        except Exception:
            pass


class TestConversationSchemas:
    def test_create_default_title(self):
        conv = ConversationCreate()
        assert conv.title == "新对话"

    def test_update_partial(self):
        update = ConversationUpdate(title="新标题")
        assert update.title == "新标题"
        assert update.status is None


class TestChatSchemas:
    def test_chat_stream_request(self):
        req = ChatStreamRequest(
            conversation_id="00000000-0000-0000-0000-000000000001",
            message="推荐一个目的地",
        )
        assert "推荐" in req.message

    def test_chat_stream_empty_message(self):
        try:
            ChatStreamRequest(conversation_id="00000000-0000-0000-0000-000000000001", message="")
            assert False, "Should have raised validation error"
        except Exception:
            pass

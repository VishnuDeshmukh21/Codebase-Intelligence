"""
Tests for public Git repository indexing and Google GenAI Gemini integration.
"""
from pathlib import Path
from app.repo_manager import is_git_url, sanitize_repo_id, resolve_repository_source
from app.llm_service.factory import LLMFactory
from app.llm_service.gemini_provider import GeminiProvider


def test_is_git_url():
    assert is_git_url("https://github.com/bottlepy/bottle.git") is True
    assert is_git_url("http://gitlab.com/group/repo") is True
    assert is_git_url("git@github.com:psf/requests.git") is True
    assert is_git_url("https://bitbucket.org/user/repo") is True
    assert is_git_url("./backend") is False
    assert is_git_url("C:\\path\\to\\code") is False
    assert is_git_url("/home/user/project") is False


def test_sanitize_repo_id():
    assert sanitize_repo_id("https://github.com/bottlepy/bottle.git") == "bottlepy_bottle"
    assert sanitize_repo_id("https://github.com/psf/requests") == "psf_requests"
    assert sanitize_repo_id("https://gitlab.com/owner/subowner/project.git") == "subowner_project"
    assert sanitize_repo_id("git@github.com:octocat/Hello-World.git") == "octocat_hello-world"
    assert sanitize_repo_id("./my-local-app") == "my-local-app"


def test_gemini_provider_client():
    client = LLMFactory.get_client()
    assert isinstance(client, GeminiProvider)
    assert client.model_name == "gemini-3.8-flash"
    
    # Verify generation with official google.genai client
    response = client.generate("Reply with only the word: PONG", temperature=0.0)
    assert "PONG" in response.upper()


def test_resolve_local_repository():
    backend_dir = Path(__file__).resolve().parent.parent
    local_path, repo_id, is_remote, normalized_url, details = resolve_repository_source(
        repo_path=str(backend_dir)
    )
    assert Path(local_path).exists()
    assert is_remote is False
    assert normalized_url is None
    assert repo_id == "backend"

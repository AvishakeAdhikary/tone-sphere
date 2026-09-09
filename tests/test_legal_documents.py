"""
The three legal documents in `docs/legal/`, and the GitHub Pages site that publishes them.

Why a test for prose. Two reasons, both concrete:

- The Microsoft Store submission points at a URL that has to keep resolving to a real
  privacy policy. A document quietly reduced to a heading and a TODO, or a front matter
  block a rename broke, fails in a place nobody looks — a store listing and a stranger's
  browser — rather than in a build.
- The in-app viewer strips the YAML front matter block and hands the rest to Qt's markdown
  renderer. Qt's renderer does not process HTML or Liquid: a stray tag reaches the user as
  literal angle-bracketed text. So the front matter contract (the five fields, the
  delimiters) and the "body is plain markdown" rule are the interface between these files
  and Python code, and are checked here as one.

There is no audio to measure here, so this file holds itself to the closest available
equivalent: it reads the actual files, resolves the actual links, and derives the expected
sponsor URLs from `.github/FUNDING.yml` rather than restating them.
"""

import re
from datetime import date
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
LEGAL = DOCS / "legal"

DOCUMENTS = {
    "terms-and-conditions.md": "Terms and Conditions",
    "terms-of-service.md": "Terms of Service",
    "privacy-policy.md": "Privacy Policy",
}

REQUIRED_FRONT_MATTER = ("title", "layout", "permalink", "version", "effective_date")

PUBLISHER = "Neural Nexus Studios"
ISSUES_URL = "https://github.com/AvishakeAdhikary/tone-sphere/issues"
RELEASES_URL = "https://github.com/AvishakeAdhikary/tone-sphere/releases/latest"

# https://pages.github.com/themes/ — a theme outside this list is not built by Pages, and
# every page renders unstyled with no error anywhere the author will see it.
GITHUB_PAGES_THEMES = {
    "minima",
    "jekyll-theme-architect",
    "jekyll-theme-cayman",
    "jekyll-theme-dinky",
    "jekyll-theme-hacker",
    "jekyll-theme-leap-day",
    "jekyll-theme-merlot",
    "jekyll-theme-midnight",
    "jekyll-theme-minimal",
    "jekyll-theme-modernist",
    "jekyll-theme-primer",
    "jekyll-theme-slate",
    "jekyll-theme-tactile",
    "jekyll-theme-time-machine",
}

# The exact block the in-app viewer removes: the file must open with it, and it ends at the
# next line that is only three dashes.
FRONT_MATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)

RAW_HTML = re.compile(r"<\s*/?\s*[A-Za-z!]")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")

FUNDING_URLS = {
    "github": "https://github.com/sponsors/{}",
    "patreon": "https://www.patreon.com/{}",
    "ko_fi": "https://ko-fi.com/{}",
    "buy_me_a_coffee": "https://www.buymeacoffee.com/{}",
    "open_collective": "https://opencollective.com/{}",
    "liberapay": "https://liberapay.com/{}",
    "polar": "https://polar.sh/{}",
    "issuehunt": "https://issuehunt.io/r/{}",
}


def split_front_matter(path: Path) -> tuple[dict, str]:
    """The in-app viewer's job, done here so that the two cannot disagree about the shape."""
    text = path.read_text(encoding="utf-8")
    match = FRONT_MATTER.match(text)
    if match is None:
        pytest.fail(f"{path.name} does not open with a `---` front matter block; Jekyll "
                    f"will not render it as a page and the in-app viewer has nothing to strip")
    return yaml.safe_load(match.group(1)), text[match.end():]


def load_document(filename: str) -> tuple[dict, str]:
    return split_front_matter(LEGAL / filename)


def site_config() -> dict:
    return yaml.safe_load((DOCS / "_config.yml").read_text(encoding="utf-8"))


def funding_accounts() -> dict[str, str]:
    funding = yaml.safe_load((REPO_ROOT / ".github" / "FUNDING.yml").read_text(encoding="utf-8"))
    accounts = {}
    for platform, value in funding.items():
        if not value:
            continue
        accounts[platform] = value[0] if isinstance(value, list) else value
    return accounts


@pytest.mark.parametrize("filename", sorted(DOCUMENTS))
class TestEachDocumentExists:
    def test_file_is_present(self, filename):
        assert (LEGAL / filename).is_file(), f"docs/legal/{filename} is missing"


@pytest.mark.parametrize("filename", sorted(DOCUMENTS))
class TestFrontMatterContract:
    """
    Five fields, agreed with the in-app viewer: `title`, `layout`, `permalink`, `version`,
    `effective_date`. Dropping or renaming one breaks a Pages page or an app screen, and
    neither failure surfaces anywhere near the edit that caused it.
    """

    def test_front_matter_is_a_mapping(self, filename):
        front_matter, _ = load_document(filename)
        assert isinstance(front_matter, dict)

    def test_required_fields_are_present_and_filled_in(self, filename):
        front_matter, _ = load_document(filename)
        for field in REQUIRED_FRONT_MATTER:
            assert field in front_matter, f"{filename}: front matter is missing `{field}`"
            assert str(front_matter[field]).strip(), f"{filename}: `{field}` is empty"

    def test_title_names_the_document(self, filename):
        front_matter, _ = load_document(filename)
        assert front_matter["title"] == DOCUMENTS[filename]

    def test_layout_is_one_the_theme_actually_ships(self, filename):
        front_matter, _ = load_document(filename)
        assert front_matter["layout"] in {"default", "page", "home"}

    def test_permalink_is_a_rooted_directory_url(self, filename):
        front_matter, _ = load_document(filename)
        permalink = front_matter["permalink"]
        assert permalink.startswith("/legal/"), f"{filename}: {permalink} is not under /legal/"
        assert permalink.endswith("/"), f"{filename}: {permalink} should end in a slash"
        assert filename.removesuffix(".md") in permalink

    def test_version_is_a_dotted_number(self, filename):
        front_matter, _ = load_document(filename)
        assert re.fullmatch(r"\d+\.\d+", str(front_matter["version"])), (
            f"{filename}: version {front_matter['version']!r} is not of the form 1.0"
        )

    def test_effective_date_is_an_iso_date(self, filename):
        front_matter, _ = load_document(filename)
        # A string survives YAML round-tripping into the app unchanged; a bare 2026-09-09
        # arrives as a date object. Either is fine, an unparseable date is not.
        assert date.fromisoformat(str(front_matter["effective_date"]))


@pytest.mark.parametrize("filename", sorted(DOCUMENTS))
class TestTheStrippedBodyRendersEverywhere:
    """
    What is left after the front matter comes off has to render identically through Jekyll
    and through `QTextBrowser.setMarkdown()`: headings, lists, emphasis, links, tables.
    Nothing else.
    """

    def test_body_opens_with_its_own_heading(self, filename):
        _, body = load_document(filename)
        assert body.lstrip().startswith("# "), (
            f"{filename}: the body must carry its own H1, because the viewer strips the "
            f"front matter that holds the title"
        )

    def test_body_contains_no_raw_html(self, filename):
        _, body = load_document(filename)
        found = RAW_HTML.search(body)
        assert found is None, (
            f"{filename}: raw HTML near {body[max(0, found.start() - 40):found.start() + 40]!r} "
            f"— Qt's markdown renderer shows it as literal text"
        )

    def test_body_contains_no_jekyll_templating(self, filename):
        _, body = load_document(filename)
        assert "{{" not in body and "{%" not in body, (
            f"{filename}: Liquid tags render as themselves outside Jekyll"
        )

    def test_body_is_a_document_rather_than_a_stub(self, filename):
        _, body = load_document(filename)
        assert len(body.split()) >= 900, f"{filename}: {len(body.split())} words looks like a placeholder"
        assert body.count("\n## ") >= 8, f"{filename}: too few sections to be a real agreement"
        for placeholder in ("TODO", "TBD", "Lorem ipsum", "[insert", "XXX"):
            assert placeholder.lower() not in body.lower(), f"{filename}: contains {placeholder!r}"


@pytest.mark.parametrize("filename", sorted(DOCUMENTS))
class TestTheDocumentsSayWhoAndWhere:
    """
    A store-facing document that does not name its publisher or its jurisdiction is not a
    document, and the contact channel is deliberately the public issue tracker — an email
    address appearing here would be a personal one, which is the mistake worth failing on.
    """

    def test_publisher_is_named_in_the_body(self, filename):
        _, body = load_document(filename)
        assert PUBLISHER in body

    def test_jurisdiction_is_named_in_the_body(self, filename):
        _, body = load_document(filename)
        for token in ("Kolkata", "West Bengal", "India"):
            assert token in body, f"{filename}: does not mention {token}"

    def test_contact_channel_is_the_issue_tracker(self, filename):
        _, body = load_document(filename)
        assert ISSUES_URL in body

    def test_no_email_address_anywhere(self, filename):
        _, body = load_document(filename)
        assert EMAIL.search(body) is None, f"{filename}: contains what looks like an email address"

@pytest.mark.parametrize("filename", sorted(DOCUMENTS))
class TestTheBodySurvivesQtsMarkdownRenderer:
    """
    The checks above describe the markdown; this one runs it through the renderer the app
    will actually use, and looks at what a reader ends up seeing. Qt's importer handles a
    different dialect from Jekyll's, so "no raw HTML" and "tables only" are worth proving
    rather than asserting: if a table stayed a row of pipes or a heading stayed a row of
    hashes, it shows up here as markup in the plain text.
    """

    def rendered(self, filename: str) -> str:
        pytest.importorskip("PySide6", reason="Qt not installed")
        from PySide6.QtGui import QTextDocument
        from PySide6.QtWidgets import QApplication

        # Qt tolerates exactly one QApplication per process, and tests/test_ui.py may have
        # already made it.
        QApplication.instance() or QApplication([])

        _, body = load_document(filename)
        document = QTextDocument()
        document.setMarkdown(body)
        return document.toPlainText()

    def test_no_markup_reaches_the_reader(self, filename):
        text = self.rendered(filename)
        for marker in ("##", "**", "|", "](", "{{"):
            assert marker not in text, f"{filename}: {marker!r} survived rendering as literal text"
        assert RAW_HTML.search(text) is None

    def test_the_document_reads_as_the_document(self, filename):
        text = self.rendered(filename)
        assert text.lstrip().startswith(DOCUMENTS[filename])
        assert PUBLISHER in text
        assert "Kolkata, West Bengal, India" in text
        assert ISSUES_URL in text, "the contact URL has to remain visible once links are rendered"


class TestNoLicenceIsClaimedThatTheRepositoryDoesNotCarry:
    """
    The Terms and Conditions grant a right to use the application and say, in as many
    words, that the repository holds no licence file — public source is not a grant. If a
    licence is ever added, that clause becomes false, and this is where that shows up.
    """

    def test_the_repository_still_has_no_licence_file(self):
        candidates = ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE", "COPYING")
        present = [name for name in candidates if (REPO_ROOT / name).exists()]
        assert not present, (
            f"{present} now exists; the Terms and Conditions state the repository contains "
            f"no licence file, so update the document and this check together"
        )

    def test_the_terms_say_so_explicitly(self):
        _, body = load_document("terms-and-conditions.md")
        assert "contains no licence file" in body.lower()

    def test_every_mention_of_open_source_is_a_denial(self):
        denials = ("no open-source", "no open source", "not be described as open-source",
                   "not open-source", "not open source")
        for filename in DOCUMENTS:
            _, body = load_document(filename)
            for sentence in re.split(r"(?<=[.;])\s+", body):
                lowered = " ".join(sentence.lower().split())
                if "open-source" not in lowered and "open source" not in lowered:
                    continue
                assert any(denial in lowered for denial in denials), (
                    f"{filename}: {lowered!r} reads as a claim that the application is "
                    f"open-source, which no licence in this repository supports"
                )


class TestTheTwoTermsDocumentsHaveDistinctScope:
    """
    The owner asked for Terms and Conditions *and* Terms of Service. The failure mode is
    one document pasted twice under two names, so each is checked for the subjects that
    belong to it alone, and the pair is checked for shared paragraphs.
    """

    def test_terms_and_conditions_covers_the_software_agreement(self):
        _, body = load_document("terms-and-conditions.md")
        lowered = body.lower()
        for subject in ("grant", "acceptable use", "warrant", "liability",
                        "governing law", "jurisdiction", "plugin"):
            assert subject in lowered, f"terms-and-conditions.md does not cover {subject!r}"

    def test_terms_of_service_covers_the_service_facing_subjects(self):
        _, body = load_document("terms-of-service.md")
        lowered = body.lower()
        for subject in ("availability", "support", "updates", "microsoft store",
                        "network streaming", "sponsor"):
            assert subject in lowered, f"terms-of-service.md does not cover {subject!r}"

    def test_terms_of_service_defers_rather_than_repeating(self):
        _, body = load_document("terms-of-service.md")
        assert "terms-and-conditions.md" in body

    def test_privacy_policy_states_the_collection_position_and_its_one_exception(self):
        _, body = load_document("privacy-policy.md")
        lowered = body.lower()
        assert "collects no user information" in lowered
        assert "no user accounts" in lowered or "there are no user accounts" in lowered
        # The one thing that leaves the machine has to be stated, not implied by omission.
        assert "network streaming" in lowered
        assert "off by default" in lowered

    def test_the_two_terms_documents_are_not_the_same_text(self):
        def paragraphs(filename: str) -> set[str]:
            _, body = load_document(filename)
            return {
                " ".join(block.split())
                for block in body.split("\n\n")
                if len(" ".join(block.split())) > 120
            }

        conditions = paragraphs("terms-and-conditions.md")
        service = paragraphs("terms-of-service.md")
        shared = conditions & service
        assert len(shared) / min(len(conditions), len(service)) < 0.1, (
            f"the two documents share {len(shared)} substantial paragraphs: {sorted(shared)[:2]}"
        )


class TestThePagesSiteIsBuildable:
    """
    Pages is served from `main`'s `/docs` folder. Everything below is something that turns
    into an unstyled page, a 404 or a failed build if it drifts — and none of it announces
    itself, because a Pages build failure lands in a repository setting, not in CI.
    """

    def test_config_declares_a_title_description_and_supported_theme(self):
        config = site_config()
        assert config["title"]
        assert config["description"].strip()
        assert config["theme"] in GITHUB_PAGES_THEMES

    def test_baseurl_matches_the_project_page_path(self):
        config = site_config()
        assert config["baseurl"] == "/tone-sphere", (
            "a project site is served under /<repository>, and a wrong baseurl breaks every "
            "stylesheet and every internal link"
        )

    def test_the_theme_plugins_minima_calls_are_declared(self):
        config = site_config()
        assert {"jekyll-seo-tag", "jekyll-feed"} <= set(config["plugins"])

    def test_header_pages_all_exist(self):
        for page in site_config()["header_pages"]:
            assert (DOCS / page).is_file(), f"_config.yml lists {page}, which does not exist"

    def test_every_markdown_page_in_the_site_has_front_matter(self):
        """
        Including `VIRTUAL_AUDIO_DRIVER.md`, which predates the site: a page without front
        matter is the one Jekyll may leave unrendered, which is how a link on the landing
        page becomes a 404.

        Files in `_config.yml`'s `exclude` are skipped, because Jekyll does not serve them
        at all — owner-facing runbooks live in `docs/` for proximity to what they describe
        without being pages. The test below keeps that from becoming a way to satisfy this
        one by excluding everything.
        """
        excluded = set(site_config().get("exclude", []))

        for page in sorted(DOCS.rglob("*.md")):
            if page.relative_to(DOCS).as_posix() in excluded:
                continue
            front_matter, _ = split_front_matter(page)
            assert front_matter.get("title"), f"{page.relative_to(REPO_ROOT)} has no title"
            assert front_matter.get("layout"), f"{page.relative_to(REPO_ROOT)} has no layout"

    def test_the_pages_the_site_exists_for_are_not_excluded(self):
        """
        `exclude` skips the front matter check above, so it must not be able to hide the
        documents that are the entire reason this site is published.
        """
        excluded = set(site_config().get("exclude", []))
        required = {"index.md", *(f"legal/{name}" for name in DOCUMENTS)}

        assert not (excluded & required), (
            f"excluded from the site build: {sorted(excluded & required)} — the Store "
            f"submission needs the privacy policy reachable at a public URL"
        )

        for entry in excluded:
            assert (DOCS / entry).exists(), (
                f"_config.yml excludes '{entry}', which does not exist — a stale exclude "
                f"silently widens what the front matter check skips"
            )

    def test_the_landing_page_links_all_three_documents_and_the_downloads(self):
        _, body = split_front_matter(DOCS / "index.md")
        for filename in DOCUMENTS:
            assert f"legal/{filename}" in body, f"index.md does not link {filename}"
        assert RELEASES_URL in body
        assert "VIRTUAL_AUDIO_DRIVER.md" in body, (
            "the driver notes render as a page, so the landing page should link them rather "
            "than leaving a page nothing points at"
        )

    def test_the_landing_page_has_a_sponsor_section(self):
        _, body = split_front_matter(DOCS / "index.md")
        assert "\n## Sponsor" in body

    def test_relative_links_in_the_site_and_readme_resolve(self):
        """
        Every relative link, checked against the filesystem. This is what stops the site
        from shipping with a dead link in it, which is otherwise invisible until a reader
        clicks one.
        """
        pages = sorted(DOCS.rglob("*.md")) + [REPO_ROOT / "README.md"]
        for page in pages:
            text = page.read_text(encoding="utf-8")
            for target in MARKDOWN_LINK.findall(text):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                resolved = (page.parent / target.split("#")[0]).resolve()
                assert resolved.exists(), (
                    f"{page.relative_to(REPO_ROOT)} links {target}, which does not exist"
                )


class TestSponsorLinksMatchFundingYml:
    """
    `.github/FUNDING.yml` is the source of truth for where sponsorship goes. A platform
    listed there and surfaced nowhere is invisible; a link surfaced here and removed there
    sends money to an account that may no longer be the right one.
    """

    def test_every_funded_platform_has_a_known_url_shape(self):
        for platform in funding_accounts():
            assert platform in FUNDING_URLS, (
                f"FUNDING.yml adds {platform!r}; add its URL shape here and surface it on the "
                f"site and in the README"
            )

    def test_the_landing_page_links_every_funded_platform(self):
        _, body = split_front_matter(DOCS / "index.md")
        for platform, account in funding_accounts().items():
            assert FUNDING_URLS[platform].format(account) in body, (
                f"docs/index.md does not link {platform}"
            )

    def test_the_readme_links_every_funded_platform(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        for platform, account in funding_accounts().items():
            assert FUNDING_URLS[platform].format(account) in readme, (
                f"README.md does not link {platform}"
            )

    def test_the_readme_points_at_the_legal_documents(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        for filename in DOCUMENTS:
            assert f"docs/legal/{filename}" in readme

"""
What can actually be checked about the Microsoft Store package from a repository checkout.

Which is: the manifest is well-formed XML that declares the identity, entry point and
capability set this app really needs; its version tracks the project's rather than drifting
away from it; and every logo path the manifest points at is genuinely produced by
`generate_assets.py`, at the pixel size its filename claims.

Which is not: that the package installs, launches, records audio, or passes Store
certification. Those need a Windows SDK, a signing certificate and a Partner Center
account, and `docs/MICROSOFT_STORE.md` says so plainly. The one step here that touches a
real SDK binary is `makeappx pack`, and it self-skips where makeappx is not installed
instead of pretending to have validated anything.
"""

import importlib.util
import re
import shutil
import subprocess
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from PIL import Image

import tonesphere

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGING_DIR = REPO_ROOT / "packaging" / "msix"
MANIFEST_PATH = PACKAGING_DIR / "AppxManifest.xml"
BRAND_PNG = REPO_ROOT / "assets" / "images" / "ToneSphere.png"

NS = {
    'f': "http://schemas.microsoft.com/appx/manifest/foundation/windows10",
    'uap': "http://schemas.microsoft.com/appx/manifest/uap/windows10",
    'rescap': "http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities",
}


def _load_generator():
    """`packaging/` is not an importable package, so the script is loaded by path."""
    spec = importlib.util.spec_from_file_location("msix_generate_assets", PACKAGING_DIR / "generate_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_assets = _load_generator()


@pytest.fixture(scope="module")
def manifest() -> ET.Element:
    return ET.parse(MANIFEST_PATH).getroot()


def _logo_references(root: ET.Element) -> set[str]:
    """
    Every `assets\\*.png` path the manifest names, from wherever it names it: the Store
    logo in Properties, the two required VisualElements logos, and the optional tiles.
    """
    references: set[str] = set()

    logo = root.find("f:Properties/f:Logo", NS)
    assert logo is not None and logo.text
    references.add(logo.text)

    visual = root.find("f:Applications/f:Application/uap:VisualElements", NS)
    assert visual is not None
    tile = visual.find("uap:DefaultTile", NS)
    assert tile is not None

    for element in (visual, tile):
        for name, value in element.attrib.items():
            if name.endswith("Logo"):
                references.add(value)

    return references


class TestManifestIsRealAndWellFormed:
    def test_it_parses(self, manifest):
        assert manifest.tag == f"{{{NS['f']}}}Package"

    def test_publisher_display_name_is_the_studio(self, manifest):
        assert manifest.findtext("f:Properties/f:PublisherDisplayName", namespaces=NS) == "Neural Nexus Studios"

    def test_display_name_is_the_app(self, manifest):
        assert manifest.findtext("f:Properties/f:DisplayName", namespaces=NS) == "ToneSphere"

    def test_it_describes_itself(self, manifest):
        description = manifest.findtext("f:Properties/f:Description", namespaces=NS)
        assert description and len(description) > 30

        visual = manifest.find("f:Applications/f:Application/uap:VisualElements", NS)
        assert visual.get("DisplayName") == "ToneSphere"
        assert visual.get("Description")

    def test_identity_is_present_and_shaped_like_an_identity(self, manifest):
        """
        Name and Publisher are placeholders until Partner Center issues the real ones, but a
        placeholder that does not even satisfy the schema would not be a usable starting
        point -- makeappx would reject it and the owner would have to guess what shape it
        wanted.
        """
        identity = manifest.find("f:Identity", NS)
        assert identity is not None

        assert re.fullmatch(r"[A-Za-z0-9.\-]{3,50}", identity.get("Name"))
        assert identity.get("Publisher").startswith("CN=")
        assert identity.get("ProcessorArchitecture") == "x64"

    def test_it_targets_the_windows_desktop_family(self, manifest):
        family = manifest.find("f:Dependencies/f:TargetDeviceFamily", NS)
        assert family.get("Name") == "Windows.Desktop"

        # Both are OS build numbers, not the app version. An earlier build_msix.ps1 stamped
        # the app version over MinVersion (the attribute name also ends in `Version="`) and
        # makeappx packed `MinVersion="0.1.0.0"` without complaint -- a package that would
        # have claimed to install on Windows 0.1.
        for attribute in ("MinVersion", "MaxVersionTested"):
            assert re.fullmatch(r"10\.0\.\d{5}\.\d+", family.get(attribute)), \
                f"{attribute}={family.get(attribute)!r} is not a Windows 10/11 build version"


class TestItIsADesktopBridgePackageNotAUwpOne:
    def test_entry_point_is_full_trust(self, manifest):
        application = manifest.find("f:Applications/f:Application", NS)
        assert application.get("EntryPoint") == "Windows.FullTrustApplication"

    def test_executable_is_the_pyinstaller_output(self, manifest):
        """
        The path has to match what build_msix.ps1 stages, or the package installs and then
        fails to launch -- which nothing but a real install would otherwise catch.
        """
        application = manifest.find("f:Applications/f:Application", NS)
        assert application.get("Executable") == r"ToneSphere\ToneSphere.exe"
        assert application.get("Id") == "ToneSphere"

    def test_no_splash_screen_is_declared(self, manifest):
        """A full-trust Win32 app never shows one; declaring an image for it ships a lie."""
        visual = manifest.find("f:Applications/f:Application/uap:VisualElements", NS)
        assert visual.find("uap:SplashScreen", NS) is None


class TestCapabilitiesAreExactlyWhatTheAppUses:
    """
    An over-broad capability list is the packaging equivalent of `CPU: 0%`: it tells the user
    the app needs something it does not. This asserts the whole set, not just that the two
    real ones are present, so a future addition has to be argued for here first.
    """

    def test_the_capability_set_is_run_full_trust_and_microphone(self, manifest):
        capabilities = manifest.find("f:Capabilities", NS)
        assert capabilities is not None

        declared = {
            (child.tag, child.get("Name"))
            for child in capabilities
        }

        assert declared == {
            (f"{{{NS['rescap']}}}Capability", "runFullTrust"),
            (f"{{{NS['f']}}}DeviceCapability", "microphone"),
        }

    def test_microphone_is_a_device_capability(self, manifest):
        """
        Audio capture is gated by the `microphone` *device* capability. A plain
        `<Capability Name="microphone">` is not the same element and grants nothing.
        """
        names = [c.get("Name") for c in manifest.findall("f:Capabilities/f:DeviceCapability", NS)]
        assert names == ["microphone"]

    def test_no_network_or_filesystem_capabilities_are_declared(self, manifest):
        """
        The API server and the TCP audio router do open sockets, but a runFullTrust process
        is not in an AppContainer, so these gate nothing here. Declaring them would widen
        the install prompt for no behaviour.
        """
        declared = {c.get("Name") for c in manifest.find("f:Capabilities", NS)}

        for over_broad in (
            "internetClient", "internetClientServer", "privateNetworkClientServer",
            "broadFileSystemAccess", "documentsLibrary", "musicLibrary",
            "allowElevation", "packageQuery",
        ):
            assert over_broad not in declared


class TestVersionTracksTheProject:
    @staticmethod
    def _pyproject_version() -> str:
        with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
            return tomllib.load(handle)["project"]["version"]

    def test_the_two_python_version_strings_already_agree(self):
        assert tonesphere.__version__ == self._pyproject_version()

    def test_manifest_version_is_a_valid_four_part_msix_version(self, manifest):
        version = manifest.find("f:Identity", NS).get("Version")
        parts = version.split(".")

        assert len(parts) == 4, f"MSIX versions are four-part; got {version!r}"
        assert all(part.isdigit() for part in parts)
        assert int(parts[3]) == 0, "the Store rejects a non-zero fourth part"

    def test_manifest_version_matches_the_project_version(self, manifest):
        """
        The version now lives in three places. This is the check that makes forgetting the
        third one a test failure rather than a package that ships under the wrong version.
        """
        version = manifest.find("f:Identity", NS).get("Version")
        expected = self._pyproject_version()

        assert ".".join(version.split(".")[:3]) == expected
        assert tonesphere.__version__ == expected


class TestGeneratedAssetsCoverTheManifest:
    @staticmethod
    @pytest.fixture(scope="class")
    def generated(tmp_path_factory) -> Path:
        output = tmp_path_factory.mktemp("msix-assets")
        written, _ = generate_assets.generate(BRAND_PNG, output)
        assert written
        return output

    def test_every_logo_the_manifest_references_is_produced(self, manifest, generated):
        missing = []
        for reference in _logo_references(manifest):
            relative = Path(reference.replace("\\", "/"))
            assert relative.parts[0] == "assets", f"{reference} is not under the staged assets directory"
            if not (generated / relative.name).is_file():
                missing.append(reference)

        assert not missing, f"the manifest references logos generate_assets.py does not write: {missing}"

    def test_referenced_logos_have_the_dimensions_their_names_claim(self, manifest, generated):
        expected = generate_assets.expected_assets()

        for reference in _logo_references(manifest):
            name = Path(reference.replace("\\", "/")).name
            assert name in expected, f"{name} is referenced but not in the generator's own asset table"

            with Image.open(generated / name) as image:
                assert image.size == expected[name], f"{name} is {image.size}, expected {expected[name]}"

    def test_the_whole_generated_set_matches_its_declared_sizes(self, generated):
        """
        Not just the manifest-referenced names: the scale and targetsize variants exist for
        Windows to resolve, and a 55x55 file called `.scale-125` would be silently wrong.
        """
        expected = generate_assets.expected_assets()
        assert len(expected) == 40

        for name, size in expected.items():
            path = generated / name
            assert path.is_file(), f"{name} was not generated"
            with Image.open(path) as image:
                assert image.size == size, f"{name} is {image.size}, expected {size}"

    def test_scale_variants_exist_for_every_tile_size(self, generated):
        for logo in generate_assets.LOGOS:
            assert (generated / f"{logo.name}.png").is_file()
            for scale in generate_assets.SCALES:
                assert (generated / f"{logo.name}.scale-{scale}.png").is_file()

    def test_the_app_list_icon_has_unplated_target_sizes(self, generated):
        """The taskbar and Alt-Tab pick these; without them Windows scales the 44px tile."""
        for size in generate_assets.TARGET_SIZES:
            assert (generated / f"Square44x44Logo.targetsize-{size}.png").is_file()
            assert (generated / f"Square44x44Logo.targetsize-{size}_altform-unplated.png").is_file()

    def test_logos_keep_an_alpha_channel(self, manifest, generated):
        """BackgroundColor="transparent" in the manifest is a promise about these files."""
        for reference in _logo_references(manifest):
            name = Path(reference.replace("\\", "/")).name
            with Image.open(generated / name) as image:
                assert image.mode == "RGBA"


def _find_makeappx() -> str | None:
    on_path = shutil.which("makeappx")
    if on_path:
        return on_path

    roots = [
        Path(r"C:\Program Files (x86)\Windows Kits\10\bin"),
        Path(r"C:\Program Files\Windows Kits\10\bin"),
    ]
    candidates = [
        candidate
        for root in roots if root.is_dir()
        for version in sorted(root.iterdir(), reverse=True)
        for candidate in [version / "x64" / "makeappx.exe"]
        if candidate.is_file()
    ]
    return str(candidates[0]) if candidates else None


class TestMakeappxAcceptsTheManifest:
    """
    The only real-SDK check available: pack a layout with the actual manifest and the actual
    generated logos and let makeappx validate it against the AppX schema. This catches what
    XML parsing cannot -- wrong child ordering, an unknown attribute, a logo path that
    resolves to nothing -- and it caught a real one while this was written (an XML comment
    containing a double hyphen, which is illegal and which makeappx rejected outright).

    The staged executable is a stub, not the frozen app: makeappx validates that the
    declared Executable exists, not that it is a working binary. So this proves the manifest
    and the asset layout, and nothing whatsoever about the application running.
    """

    def test_pack_succeeds(self, tmp_path):
        makeappx = _find_makeappx()
        if makeappx is None:
            pytest.skip("makeappx.exe not installed (Windows SDK); manifest schema validation not run")

        layout = tmp_path / "layout"
        (layout / "ToneSphere").mkdir(parents=True)
        (layout / "ToneSphere" / "ToneSphere.exe").write_bytes(b"not a real binary")

        generate_assets.generate(BRAND_PNG, layout / "assets")
        shutil.copy(MANIFEST_PATH, layout / "AppxManifest.xml")

        package = tmp_path / "ToneSphere.msix"
        result = subprocess.run(
            [makeappx, "pack", "/d", str(layout), "/p", str(package), "/o"],
            capture_output=True, text=True, timeout=300,
        )

        assert result.returncode == 0, f"makeappx rejected the package:\n{result.stdout}\n{result.stderr}"
        assert package.is_file()
        assert package.stat().st_size > 0

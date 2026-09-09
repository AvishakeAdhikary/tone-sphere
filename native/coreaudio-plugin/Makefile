# Build, sign, install and remove the ToneSphere AudioServerPlugIn.
#
# Plain clang, no Xcode project: a HAL plug-in is an ordinary CFPlugIn bundle, so the
# whole build is "compile one C file into a bundle binary and put an Info.plist next to
# it". Written on a machine with no macOS on it — every recipe here is exercised for the
# first time by the `build-macos-plugin` job in .github/workflows/ci.yml.
#
#   make            build build/ToneSphereAudio.driver
#   make sign       ad-hoc codesign it (free; no Apple Developer account needed)
#   make install    copy it to /Library/Audio/Plug-Ins/HAL and restart coreaudiod (sudo)
#   make uninstall  remove it and restart coreaudiod (sudo)
#   make clean      delete the build directory

BUNDLE_NAME := ToneSphereAudio.driver
BINARY_NAME := ToneSphereAudio
BUILD_DIR   := build
BUNDLE      := $(BUILD_DIR)/$(BUNDLE_NAME)
INSTALL_DIR := /Library/Audio/Plug-Ins/HAL
INSTALLED   := $(INSTALL_DIR)/$(BUNDLE_NAME)

# Universal by default: an Apple-silicon Mac and an Intel Mac both load this bundle into
# their own coreaudiod, and a single-architecture build silently fails to appear on the
# other one.
ARCHS   := -arch x86_64 -arch arm64
CFLAGS  := -Wall -O2 -fvisibility=hidden -mmacosx-version-min=11.0 $(ARCHS)
LDFLAGS := -bundle -framework CoreFoundation -framework CoreAudio

.PHONY: all sign install uninstall restart-coreaudiod clean

all: $(BUNDLE)

$(BUNDLE): ToneSphereAudio.c Info.plist
	rm -rf "$(BUNDLE)"
	mkdir -p "$(BUNDLE)/Contents/MacOS"
	cp Info.plist "$(BUNDLE)/Contents/Info.plist"
	clang $(CFLAGS) $(LDFLAGS) -o "$(BUNDLE)/Contents/MacOS/$(BINARY_NAME)" ToneSphereAudio.c

# Ad-hoc signing (`--sign -`) is free and is enough for coreaudiod to load a plug-in the
# user built and installed themselves. It is NOT notarisation: a bundle downloaded from
# the internet carries a quarantine attribute that this does not clear.
sign: $(BUNDLE)
	codesign --sign - --force "$(BUNDLE)"
	codesign --verify --verbose "$(BUNDLE)"

install: sign
	sudo rm -rf "$(INSTALLED)"
	sudo cp -R "$(BUNDLE)" "$(INSTALL_DIR)/"
	sudo chown -R root:wheel "$(INSTALLED)"
	$(MAKE) restart-coreaudiod

uninstall:
	sudo rm -rf "$(INSTALLED)"
	$(MAKE) restart-coreaudiod

# coreaudiod only rescans /Library/Audio/Plug-Ins/HAL when it restarts. `launchctl
# kickstart -k` is the documented way; `killall` is the older one that still works on
# systems where the label differs, hence the fallback. Which of the two actually applies
# on the CI runner's macOS version is one of the things the CI job's log will settle.
restart-coreaudiod:
	sudo launchctl kickstart -k system/com.apple.audio.coreaudiod || sudo killall coreaudiod
	sleep 3

clean:
	rm -rf "$(BUILD_DIR)"

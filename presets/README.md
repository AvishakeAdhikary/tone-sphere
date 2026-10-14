# ToneSphere presets

A preset is a routing map saved as a `.tonesphere.json` file. Import one from the **Scenes** menu
(**Import preset…**). Save your own with **Export as preset…**.

## How presets find devices

Presets refer to devices **by name**, for example `"Line (AI-04)"`, not by Windows' internal
IDs. That way the same file works on any PC.

- **Not plugged in yet:** the device appears as *not connected* and its connections wait. When
  you plug it in, ToneSphere connects it automatically.
- **Unplugged later:** its connections are kept. Plug it back in, even into another USB port, and
  everything returns to how it was.
- **Apps that use the ToneSphere ASIO driver** (DAWs, Guitar Rig, …) are matched by their program
  name and join the preset as soon as they open the driver.
- **Importing replaces the current routing,** and the preset becomes a scene you can switch back
  to at any time.

## `guitar-rig-obs-ai04.tonesphere.json`: Guitar Rig live into OBS

```
Guitar ─► AI-04 input 1 (Hi-Z) ─► ToneSphere ─► Guitar Rig 7 (ToneSphere ASIO) ─► ToneSphere ─┬─► Laptop speakers (Realtek)
                                                                                     └─► VB-CABLE ─► OBS
YouTube / desktop audio ─► Laptop speakers (Windows, as usual)
```

Guitar Rig's WASAPI mode wants one device for both input and output. With a single VB-CABLE
used that way, Guitar Rig would record its own output. The ToneSphere ASIO driver gives Guitar Rig
one device, while ToneSphere brings in the AI-04 input and sends the result to both places.

**Set up once:**
1. In ToneSphere, open the **Scenes** menu, choose **Import preset…**, and pick this file.
2. In Guitar Rig 7, open **Audio and MIDI Settings** (File menu, or the settings icon). On the
   **Audio** tab, set **Driver: ASIO** and **Device: ToneSphere ASIO**. On the Routing tab, use
   inputs 1/2 and outputs 1/2. 256 samples is a good buffer size to start with.
3. In OBS, add **Sources → + → Audio Input Capture** and choose
   **CABLE Output (VB-Audio Virtual Cable)**. Mute OBS's own capture of the speakers if you
   don't want desktop audio recorded twice.
4. Keep **Speakers (Realtek(R) Audio)** as Windows' default output. YouTube and other apps keep
   playing there, mixed with your guitar.

**Check it:**
- Play a few notes. In ToneSphere, the *Line (AI-04)* meter moves, then *Guitar Rig 7 (ASIO)*, then
  *Speakers* and *CABLE Output*.
- The AI-04's input 1 is the left channel of "Line (AI-04)". The preset sends it to both of
  Guitar Rig's inputs.

**Lower latency:**
- Lower Guitar Rig's buffer (128 samples) if you hear no crackles.
- In ToneSphere, click the **Line (AI-04)** card and open Advanced. Try **Exclusive mode**: ToneSphere
  then talks to the interface directly. Other apps can't use the AI-04 input while it's on.
- Settings › Audio › **Measure latency** shows the real round trip of a loop.

**If you unplug the AI-04:** its card shows *disconnected* and the connections stay. Plug it
back in and the guitar is back in Guitar Rig a moment later; you don't need to restart anything.

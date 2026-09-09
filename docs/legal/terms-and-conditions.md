---
title: Terms and Conditions
layout: default
permalink: /legal/terms-and-conditions/
version: "1.0"
effective_date: "2026-09-09"
publisher: Neural Nexus Studios
description: The agreement between you and Neural Nexus Studios governing your use of the ToneSphere application, including the grant of use, acceptable use, warranties, liability and governing law.
---

# Terms and Conditions

- **Application:** ToneSphere
- **Publisher:** Neural Nexus Studios, an individual developer, Kolkata, West Bengal, India
- **Document version:** 1.0
- **Effective date:** 9 September 2026

## 1. About this document

These Terms and Conditions are the agreement between you and **Neural Nexus Studios** (in
this document, "we", "us" or "the Publisher") covering **the Application** — the ToneSphere
audio routing and mixing software, in any form in which we distribute it, together with its
documentation.

This document governs your right to install and use the Application. Two related documents
sit alongside it and are part of the same agreement:

- The [Terms of Service](terms-of-service.md) describe what the Application provides and
  does not provide, how it is distributed, updated and supported, and the terms attaching to
  its optional network feature.
- The [Privacy Policy](privacy-policy.md) describes how information is handled, which in
  practice means describing how little of it there is.

Where a subject is covered in one of those documents, it is cross-referenced here rather
than repeated.

## 2. Acceptance

By installing, copying or using the Application you accept these Terms and Conditions. If
you do not accept them, do not install or use the Application, and remove any copy you have
already installed.

If you are using the Application in the course of employment or on behalf of an
organisation, you confirm that you are authorised to accept these terms on its behalf.

## 3. Grant of use

**The Application is released under the MIT Licence.** The full text is in the `LICENSE`
file at the root of the project repository, and it — not this section — is the operative
grant. If anything in this document appears to narrow it, the MIT Licence wins.

What that means in practice, stated plainly because MIT is short enough to be easy to
under-read:

- **You may use, copy, modify, merge, publish, distribute, sublicense and sell** copies of
  the Application, including for commercial purposes, and you do not need our permission
  to do so.
- **The one condition is attribution.** The copyright notice and the permission notice must
  be included in all copies or substantial portions of the Application. Removing them is
  the one thing that takes you outside the licence.
- **No fee is charged** by us for the Application as we distribute it, through either the
  Microsoft Store or the project's releases.
- **The licence covers the software, not the name.** "ToneSphere" and "Neural Nexus Studios"
  are ours. MIT grants no trademark rights, so a modified or redistributed build must not be
  presented in a way that suggests it is the official Application or that we endorse,
  support or published it. Rename it, or say clearly what it is.
- **Third-party components keep their own licences.** See section 8.

## 4. Acceptable use

The MIT Licence in section 3 governs what you may do with the software itself, and this
section does not take any of it back. What follows is about conduct the licence does not
speak to — the law, other people's audio, and other people's trademarks.

You may not:

- remove, obscure or alter the copyright and permission notices. This is not an extra
  restriction we are adding; it is the MIT Licence's one condition, restated here because
  it is the term most often overlooked;
- present a modified or redistributed build as the official Application, or in a way that
  implies our endorsement. MIT grants no rights in the "ToneSphere" or "Neural Nexus
  Studios" names;
- misrepresent the Application's capabilities when describing or redistributing it. The
  project documents, deliberately and in detail, which of its features are proven, which
  are measured on which hardware, and which are unverified. Presenting an unverified
  capability as a working one — including the platform limitations listed in the
  [Terms of Service](terms-of-service.md) — is a misuse of the Application's documentation
  as well as a disservice to whoever believes it;
- use the Application to capture, monitor, route or transmit audio that you do not have the
  right to capture. The Application can capture a single application's output, a whole
  output device, and audio arriving from a network peer. Whether a particular capture or
  recording is lawful depends on where you are and who else is party to the sound —
  recording calls, meetings, performances, broadcasts or other people generally requires
  their consent, notice, or a licence, and complying with the law that applies to you is
  your responsibility;
- use the Application to circumvent a technical measure protecting audio content, or to
  infringe anyone's copyright, performance rights or privacy;
- use the Application for any unlawful purpose.

You are responsible for your use of the Application, and you agree to hold us harmless from
claims arising out of your breach of this section — in particular claims relating to audio
you captured, transmitted or published.

## 5. Third-party plugins

The Application can host VST3 and AU audio plugins that you already own, and load them onto
channels you choose. Those plugins are not ours:

- We do not supply, sell, bundle or endorse any third-party plugin. Your right to use a
  plugin comes from that plugin's own licence, from its vendor, and having a valid licence
  for each plugin you load is your responsibility.
- A hosted plugin runs inside the Application's process, in the real-time audio path. A
  plugin that misbehaves can therefore produce unexpected output, add latency, degrade
  performance, or crash the Application. We are not responsible for a plugin's behaviour,
  its stability, its audio quality, or anything it does on your system.
- What a plugin transmits, stores or checks over the network is governed by its vendor's own
  terms. See section 5 of the [Privacy Policy](privacy-policy.md).

## 6. Third-party components

The Application is built on third-party software, including PortAudio (through
`sounddevice`), Qt (through PySide6), `pedalboard`, NumPy, FastAPI and others. Each of those
components is governed by its own licence, and those licences continue to apply to those
components. Nothing in this agreement restricts a right you have in a third-party component
under its own licence.

## 7. Audio levels, hearing and equipment

The Application processes and monitors audio in real time, at gain settings you control. A
routing mistake, a feedback loop, a mis-set gain or a misbehaving plugin can produce sudden
loud output.

The Application places a limiter on each output, and that limiter exists so that a routing
mistake sounds like a compressed mix rather than a burst of digital noise. It is a
mix-safety measure. It is not a hearing-protection device and it is not an equipment
protection device, and it should not be relied on as either. Set levels carefully, start
low, and protect your ears, your monitors and your headphones yourself.

## 8. No warranty

The Application is provided **"as is" and "as available", without warranty of any kind**,
whether express, implied or statutory, including any implied warranty of merchantability,
fitness for a particular purpose, non-infringement, or uninterrupted or error-free
operation. To the extent that applicable law does not permit the exclusion of a particular
warranty, that warranty is limited to the minimum period and extent the law requires.

In particular, and consistent with how this project documents itself:

- **Performance figures are measurements, not promises.** Latency, dropout and DSP-load
  figures in the project's documentation were measured on specific hardware with a specific
  driver, buffer size and backend, and are published together with those conditions. Your
  machine, drivers, devices and plugins will produce different numbers. Nothing in the
  documentation is a warranty of a particular latency or of glitch-free audio.
- **Features documented as unverified are unverified.** The Terms of Service list the
  platform features that are proven, and the ones — the macOS virtual audio device in
  day-to-day use, in particular — that are proven only in continuous integration and not in
  real-world use. They are provided on exactly those terms.
- **The Application is not certified for safety-critical, life-safety, broadcast-compliance
  or other high-reliability use**, and must not be relied upon where a failure of audio
  would cause injury, significant loss, or a breach of a regulatory obligation.

## 9. Limitation of liability

To the fullest extent permitted by applicable law, we will not be liable for any indirect,
incidental, special, consequential, exemplary or punitive damages, or for any loss of
profit, revenue, goodwill, data, recordings, performances, sessions or business opportunity,
arising out of or in connection with the Application or its use, whether in contract, tort
(including negligence) or otherwise, and whether or not we were advised of the possibility
of such loss.

To the fullest extent permitted by applicable law, our total aggregate liability arising out
of or in connection with the Application and this agreement is limited to the greater of the
amount you actually paid us for the Application — which, as we charge nothing for it, will
ordinarily be zero — or INR 1,000.

Nothing in this agreement excludes or limits liability that cannot lawfully be excluded or
limited, including liability for death or personal injury caused by negligence, or for
fraud or fraudulent misrepresentation.

## 10. Voluntary support

Sponsorships and donations made through GitHub Sponsors, Patreon, Ko-fi or Buy Me a Coffee
are voluntary contributions to the project's continued development. They are not a purchase
of the Application, and they do not grant additional rights, entitlements, warranties or
service commitments. See the [Terms of Service](terms-of-service.md) for how that is
handled.

## 11. Term and termination

This agreement applies for as long as you use the Application.

**Your MIT Licence rights are not revocable by us, and nothing here ends them.** The MIT
Licence is a grant, not a subscription: it continues for as long as you meet its one
condition, and this agreement cannot and does not take it away. What can lapse for breach
of this agreement is anything granted *beyond* MIT — permission to use our names, and any
support or goodwill we extend voluntarily under section 10.

You may stop using the Application at any time by uninstalling it. Sections 8, 9 and 13
survive.

## 12. Changes to these terms

We may update these Terms and Conditions. Every version carries a version number and an
effective date at the top of the document, and the current version is published at the
address where you are reading it. If you continue to use the Application after an updated
version takes effect, you accept the updated version. If you do not accept it, stop using
the Application and uninstall it. Superseded versions remain in the project repository's
history.

## 13. Governing law and jurisdiction

This agreement, and any dispute or claim arising out of or in connection with it or with the
Application, is governed by the laws of **India**, without regard to conflict-of-laws rules.

The courts at **Kolkata, West Bengal, India** have exclusive jurisdiction over any such
dispute or claim, and you and we submit to that jurisdiction. Nothing in this section
removes a right you may have under the consumer-protection law of your own country of
residence to bring proceedings there, where that law gives you that right and it cannot be
excluded by agreement.

## 14. General

- **Severability.** If any provision of this agreement is held unenforceable, it is to be
  read down to the minimum extent needed to make it enforceable, or severed if that is not
  possible, and the remaining provisions continue in force.
- **No waiver.** A failure to enforce a provision is not a waiver of it.
- **Assignment.** You may not assign or transfer this agreement. We may assign it as part of
  a transfer of the project.
- **Entire agreement.** This document, together with the [Terms of Service](terms-of-service.md)
  and the [Privacy Policy](privacy-policy.md), is the entire agreement between you and us
  about the Application, and replaces any earlier understanding about it.
- **Additional terms of a distribution channel.** Where you obtained the Application from a
  store — the Microsoft Store, in particular — that store's own terms also apply to the
  acquisition. See the [Terms of Service](terms-of-service.md).

## 15. Contact

Questions, notices and permission requests relating to this agreement should be raised as an
issue in the project repository:

[https://github.com/AvishakeAdhikary/tone-sphere/issues](https://github.com/AvishakeAdhikary/tone-sphere/issues)

That is the project's contact channel. Issues are public, so do not include anything
confidential in one.

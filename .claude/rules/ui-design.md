# Photone UI design rules

Follow these for any change to the interface (photone/ui/, the UI parts of app.py), so new parts
match what is already there. The code is the current design of record: where a doc disagrees with
the code, the code wins. Copy the canonical example named below before inventing anything.

## Process first

- **Don't restyle what exists unless asked.** A look already in use (wheel, filmstrip, cards,
  sliders, signs) stays as it is. If a render prompts a question, explain it and ask first. A
  test-chart shot is not the app: offer the same render on a real photo.
- **Show renders before settling layout or placement.** Render the options with `./run.sh shots`
  as labelled variants (A/B/C…) side by side, and ask from the pictures. Variants that need code
  go behind `PHOTONE_TRIAL=<name>` (the `trial` context property). Delete the losers once one is picked.
- **After any QML edit,** run `./run.sh shots "(tools)"` (the interface must load), then
  `./run.sh check` and `./run.sh lint`. Look at the PNGs: an audit that passes doesn't mean the screen looks good.
- **Every new panel** gets `objectName "box:…"`, floating items `"screen:…"`, modals `"modal:…"`,
  and interactive items get `slider:key`, `panel:x`, `reset:tool`, `compare:tool` and the like.
  Add a shot for each new state and an audit with real events for each new gesture
  (scripts/shots.py).
- **Never capture the desktop.** Use Qt's window grab.
- **Windows too:** no Linux-only calls, `win_amd64` wheels for any new library, `"utf-8"` on file
  reads, `**QUIET` on subprocesses.
- **Never name the app Photone's workflow is modelled on** in tracked files, commits or branches.
  Icons, code and wording are Photone's own.

## Look

- **Neutral gray chrome.** A tinted interface skews how the eye judges the photo's colors. Color is
  for the photo and for controls about color. Every color comes from `Theme.qml` and goes through
  `fix()`. Don't hard-code hex values in components. If you need a new shade, add a token.
- **Grays by role:** `bg` (window, around the photo, dark list boxes, text fields), `panel`
  (columns, bars, menus, modals), `card`, `raised` (controls, hover), `selected` (chosen
  segment/chip/row), `line` (1 px dividers and borders, slider tracks), `text` / `dim` / `faint`
  (primary / secondary / help, counts and disabled).
- **Colors by meaning:** `mark` amber = adjusted. `yellow` = unsaved, i.e. changed since the
  last checkpoint (name, number, fill and thumb of a slider). `red` = danger and delete. `green` =
  the main positive action (Save checkpoint, Accept, Neural Denoise…). Filmstrip selection = light
  blue. The white-balance pad is the one place color is shown at full strength for its own sake.
- **Every card has its own icon and color** in `Theme.cards`. Down the Adjust tab the colors run as
  a continuous rainbow starting from yellow. A new card takes its color from its place in that order.
  Utility cards (presets, crop, masks, mono) are neutral `#c9c9cf`. The color goes on the icon and
  the title only, plus a very faint tint of the card head and outline via `Theme.cardGround()`.
  It never goes around the photo.
- **Type:** Noto Sans, shipped in ui/fonts. `Theme.fontSize` 13 for body, `titleSize` 13 for card
  titles, `smallSize` 11 for sub-headings. Help text, chips, counts and menus use 12. The only
  weight is `Font.DemiBold`. Every number uses tabular figures: `font.features: {"tnum": 1}`.
- **Headings:** card titles are UPPERCASE, DemiBold, letterSpacing 0.9, in the card's color.
  In-card sub-headings are UPPERCASE `smallSize`, DemiBold, 0.9, in `dim`.
- **Rhythm:** `Theme.row` 28 for list rows, `Theme.slider` 44 for a slider (name and value on one
  line, the 24 px track below). Setting rows are 34 px. Buttons, menu entries and text fields are
  30 px, chips 26 px. Panel content is inset 16 px each side.
- **Equal gaps on all sides** of a box. Neighbouring boxes share the same corner radius.
- **Groups of sliders are separated by a 16 px gap, never a line,** because a line reads as a
  slider's track.
- **Radii:** cards, glass bars and menus 8. Modals 10. Buttons and dark list boxes 6. Segments,
  chips and rows 5. Pills are fully round.
- **Borders** are 1 px. The focus ring is 1 px `Theme.text` on `activeFocus`.
- **Icons** are SVG path strings on a 20-unit grid, drawn by `Icon.qml` with round caps and joins
  and a stroke of about 1.7 (1.8 on cards). They are designed through rounds of labelled renders.
  Shared signs live in Theme (`resetSign`, `pencilSign`, `binSign`, `linkSign`, `sparkSign`).
- **Motion:** OutCubic, short. Zoom glides in 85 ms, a scroll notch in 110 ms, a chevron turns in
  110 ms, fades take 140 to 250 ms, and an opened card's outline flashes back from white in about
  1.5 s. **Only what the user asked for in that moment glides. Anything set by code is instant.**
- **Scrollable frames show no visible scrollbar,** so content is never offset by one. Use
  `Scrollbar` (hidden) for the wheel and `MoreBelow` fades that scale with the distance left to
  scroll.

## Components: reuse, don't rebuild

| Need | Use / copy |
|---|---|
| A section of the right or left column | `Panel.qml` + `Ground.qml`. Copy an Adjust card in Main.qml, e.g. `panel:saturation` |
| A numeric setting | `ParamSlider.qml` through `EditSlider` / `LocalSlider` in Main.qml |
| A choice of 2–4 | a raised track of `Segment.qml` items. As a labelled 34 px row, or full width for tabs |
| A choice of many, wrapping | chips: 26 px, radius 5, `raised`, chosen = `selected` with a 1 px dim border |
| On/off | the `OnOff` row in Main.qml (the whole row clicks; `mark` when on) |
| Button | `PushButton.qml` (with `tint` for green or red), or `TextButton.qml` for a word or sign (`strong`, `danger`, `glow`) |
| A list | the masks list in Main.qml: a `bg` box with a 1 px `line` border, radius 6, a 4 px inset, height `N * Theme.row + 8` (5 rows), hover = `raised`, selected = `selected`, a faint centred empty label, pencil (green glow) and bin (red glow) shown on hover, inline rename |
| Anything floating over the photo | the `Glass` component in Main.qml (blur, dark wash, faint white outline, radius 8) |
| A hint about a gesture | the tip toast at the top middle of the photo. Words separated by `  ·  `, fades after 5 s. No status line |
| An error | the notice above the zoom bar, one sentence, click to dismiss |
| A question | `confirm` in Main.qml: modal, Cancel has focus, arrows/Tab move between buttons, Esc cancels, nothing reaches what is underneath |
| Right-click | the context menu in Main.qml: kept inside the window, closes on Esc, outside press or wheel |
| A text field | `bg`, 30 px, radius 6. Border `faint`, `text` when focused, `red` when invalid. Faint placeholder |
| An empty view | centred: a 15 px title, a dim sentence, a PushButton. Never a blank view while loading: show a stand-in, sized as the result will be |

## Behaviour

- **Every edit is live, with no Apply button.** One gesture is one undo step (`commitEdit()` on
  release). Undo covers edits only, never the view or the selection.
- **Double click resets** a slider, pad, handle or pane width to its default. On a curve or EQ
  point it removes the point. On the photo it toggles Fit and 1:1.
- **Click a slider's number to type.** In every text field, Enter or clicking away keeps the value.
  Only Esc discards it. Where possible, apply the value about 350 ms after typing stops.
- **Pressing a slider's track jumps the thumb there** as the first move of the drag. Shift makes
  any drag 10× finer, with no jump when Shift is pressed or released mid-drag.
- **Card heads:** the whole head folds the card, with a pointing-hand cursor. The status dot shows
  when the card is adjusted. Reset appears right after the title and dot, only when the card has
  changed the photo. Hold A/B (shown even when folded) to see the photo without the card. Crop has
  no A/B.
- **Opening a card** scrolls it smoothly to mid-column and flashes its outline.
- **Show only what's relevant:** hide what can't apply (e.g. raw-only settings on a JPEG). Dim
  what is temporarily unusable (opacity about 0.35) and leave it in place, so the layout never jumps.
- **Hover reveals** row actions, filmstrip marks and the mask frame. Fade things in and out; don't
  pop them.
- **Ask before what can't be undone** (delete, uninstall, remove from library, discard unsaved).
  Don't ask for what Ctrl+Z brings back.
- **Selection:** the open photo has a white outline, selected photos a light-blue outline drawn on
  the picture's own bounds. Filmstrip cells are the image only, with no dark slot behind them.
- **Photo view:** zoom about the pointer in fixed steps. The wheel never lands on Fit. The middle
  button only fits and never pans. Pixels show as squares from 200 %. **An effect shows at every
  zoom:** compute it at full size and shrink. Never skip or fade it by zoom.
- **Keys:** every key except the arrows acts once however long it's held. A modal takes all keys.
  Pressing the photo takes the keys back from any control. Give every control `Accessible.role`
  and `name`, and `activeFocusOnTab` on buttons.
- **Remember what the user sets:** pane widths, strip size, last folder and photo, export settings,
  brush. Double click on a pane edge restores the default.
- **Cursors say what a press does:** pointing hand on clickable heads and rows, open/closed hand on
  draggable content, I-beam on editable numbers, resize on edges. Custom cursors are real OS
  cursors drawn in Python (`TurnCursors` in app.py), not drawn by the window.
- **Speed is part of the design:** fast input goes through `PerFrameInput`, and only the newest
  pending request is handled. Draw with shaders, cull what is out of sight. No
  `grabToImage(function)` or `QByteArray` in QML.

## Wording

- Short, plain, sentence case, American spelling ("color", "gray"). A control is named for what it
  changes.
- **Help text is rare:** 12 px `faint`, saying what to do ("Drag the line at a color: up
  strengthens it…"), not what a graph means. Leave it out when the control explains itself.
- **Buttons are verbs.** Add `…` when another step follows: "Open…", "Add folder…", "Remove…".
  Use "+ Radial" for adding. A mode button names how to leave the mode ("Pick neutral" → "Done").
- **Confirmations:** the title is a question with the item in curly quotes (Delete “X”?). The body
  states the consequence and then what stays. The action button repeats the verb ("Delete",
  "Discard changes"), next to Cancel.
- **Numbers:** show the sign when neutral is 0 ("+0.50"). Units: EV, K, °, %, px, ×.
  Dimensions as "6000 × 4000". Separator `  ·  `. Times are relative: "5 min ago", "2 days ago".
- **Name things the way people say them:** "Sony Alpha 7 IV", not the model code. Unknown goes
  last as "Other". Number masks within their own kind (Radial 1, Linear 1, Radial 2).

## Still open (don't treat as settled, don't "fix" unasked)

- Whether to merge the near-duplicate chips, switches (Export.qml) and zoom bars (Denoise.qml)
  into shared components, and whether to make the 12 px size and the glass overlay colors tokens.
  For a new part, reuse the canonical example above anyway.
- Accent use has grown past "amber = adjusted only" (stars, switch on, pair link). The filmstrip
  pair outline is a "for now" pick. Presets are still a trial. Whether card fold state should be
  saved is undecided.

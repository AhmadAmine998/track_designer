# track_designer
Interactive GUI for interactively designing racetracks and creating their occupancy grids.

## Drawing a track

In **Draw Track** mode you can either click waypoints one at a time, as before, or
hold the mouse down and sweep out the centerline freehand. Release near where you
started and the loop closes itself.

Each sweep continues the track from its last waypoint, so you can build a layout in
as many passes as you like and mix sweeps with single clicks. Release near waypoint
0 to close the loop. Sweeping on an already-closed track starts a new one; use
*Clear All* or *Delete Last Point* to back out otherwise.

The **Detail** slider re-simplifies the sweep you just drew — slide it until the
waypoint count looks right, no need to redraw. It only touches the most recent
sweep, and stops touching it as soon as you edit a point by hand.

In **Select/Edit** mode, drag a waypoint to move it. The selected waypoint also
grows two green diamond handles sitting on the track edges: drag one to widen that
side. **Lock L = R** (on by default) makes both sides follow one handle or slider.
*Apply to All* pushes the selected point's width onto the whole track.

**Ctrl+Z** undoes and **Ctrl+Y** (or Ctrl+Shift+Z) redoes, up to 60 steps back. A
whole sweep, a point drag or a width-handle drag each undo as a single step.
Undo covers placing and deleting points, sweeps, obstacles, point moves, handle
drags, *Apply to All* and *Clear All*; it does not track the width sliders or the
Detail slider, which you can simply slide back.

## Autosave

The design in progress is written to `~/.track_designer/autosave.json` every few
seconds whenever it has changed, and again when the window is closed. If the app
crashes, relaunch it and answer **Yes** to the "Restore previous session" prompt.

The autosave covers waypoints and their per-point widths, obstacles, the space and
track parameters, and the floor plan (path, page, rotation and opacity). A floor
plan that has since moved is reported in the sidebar rather than blocking the
restore.

**Save** now also writes `<timestamp>.json` next to the exported `.png`/`.yaml`
occupancy grid. That JSON is the editable design — reopen it with **Open Project…**
to keep working on a track you exported earlier.

# 3D crime-scene reconstruction (from scratch)

Photos of a room + a text description -> a 3D room with an animated re-enactment.

    pip install -r requirements.txt
    python run.py --demo --out out --serve                      # no models needed, synthetic room
    python run.py --photos photos/ --out out --serve \
      --prompt "A person enters through the door, walks up to the person sitting on the chair, stabs him and then leaves."

Open http://localhost:8000. Mesh options: `--mesh-stride 2` (denser, bigger files), `--no-mesh`. Optional: `export ANTHROPIC_API_KEY=...` for LLM prompt parsing.

## Pipeline
1. `depth.py`      per-photo metric depth (Depth Anything V2 indoor)
2. `register.py`   SIFT matches -> 3D-3D RANSAC similarity (Umeyama) -> pose graph (no COLMAP)
3. `layout.py`     fuse to one coloured cloud, RANSAC the floor, level to Y-up
   `mesh.py`       each photo's depth map -> textured triangle grid (cut at depth jumps), merged into one mesh
4. `script_parser` prompt -> actors/events JSON (Claude, or regex fallback)
5. `semantic.py`   OWL-ViT finds the door/chair/... named in the prompt, lifted to 3D floor positions
6. `navgrid.py`    floor-plan occupancy grid from the cloud + A* so actors walk around furniture (`nav.png` shows it)
7. `planner.py`    events + anchors -> timed walk/run/attack/collapse tracks
8. `viewer/`       Three.js: mesh/points toggle, ceiling cut in top-down view, mannequins, route overlay, Orbit/Top-down/Follow cameras, timeline, video export

## Taking photos that work (important)
- The room is rebuilt from what the photos *saw*. Photos from one spot, or a straight dolly/video walk-through
  down the middle, give a 2.5D "relief" of that view, not a full room. The viewer opens in **Photo view**
  (the first photographer position) for exactly this reason; Orbit from outside shows the hollow back of it.
- For a real 3D room: walk around, shoot from 4+ different spots (corners, both ends), looking in
  different directions, with ~40% overlap between neighbours, floor visible, no zoom changes.
- Video frames work, but pick frames from *different positions*, not 12 frames a few cm apart.

## Limits (be honest in any write-up)
- Monocular depth is approximate: expect centimetre-to-decimetre error, noisy walls, no hidden geometry.
- Needs ~40% overlap between photos and a visible floor. Wrong focal guess -> tune `--f-scale`.
- If a noun isn't detected, give it manually: `--anchor door=1.2,-0.4`.
- It visualises a *hypothesis*; it is not evidence.
- Obstacles come from the cloud (0.2-1.7 m height band); unseen areas count as free, so check `nav.png`.
- The mesh is per-view and unfused: overlapping views can show slight seams or double surfaces, and unseen areas are holes.
- Next steps: fused/watertight mesh (TSDF or Poisson), Gaussian splats, real rigged humans (SMPL/Mixamo), more actions and multiple actors at once.

Tests: `python tests/test_core.py` (no models needed).

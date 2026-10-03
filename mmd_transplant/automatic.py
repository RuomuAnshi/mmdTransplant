"""One-click PMX pipeline with bounded, verified neck alignment."""
import math

from .core import Options, TransplantError, eye_measurement, transplant
from .neck import find_rim


def _alignment_shift(model, report, body):
    """Propose only a small upward correction between credible skin rims."""
    measurement = eye_measurement(model)
    width = measurement['geometry_span'] or measurement['bone_span']
    if not width or not math.isfinite(width):
        return None
    donor = set(report['donor_vertex_map'].values())
    retained = set(report['body_vertex_map'].values())
    head_faces, body_faces = {}, {}
    cursor = 0
    for index, material in enumerate(model.materials):
        count = material.vertex_count // 3
        faces = model.faces[cursor:cursor + count]
        cursor += count
        ids = {i for face in faces for i in face}
        if ids and ids <= donor:
            head_faces[index] = faces
        elif ids and ids <= retained:
            body_faces[index] = faces
    anchor = body.bones[report['body_head_index']].location
    points = [v.co for v in model.vertices]
    head = find_rim(points, head_faces, anchor, width)
    neck = find_rim(points, body_faces, anchor, width)
    if not head or not neck:
        return None
    horizontal = math.hypot(head['center'][0]-neck['center'][0],
                            head['center'][2]-neck['center'][2])
    shift = max(p[1] for p in neck['points']) - min(p[1] for p in head['points']) + width*.02
    if horizontal > width*.15 or not 0 < shift <= width*.2:
        return None
    return shift


def transplant_automatic(pmx, head, body):
    """Ignore stale manual settings; move the entire donor rig if necessary.

    An initial successful fit wins. A failed fit gets at most one bounded
    retry, and that retry replaces the result only after a verified join.
    """
    result, report = transplant(pmx, head, body, Options())
    alignment = {'applied': False, 'vertical_shift': 0.0}
    if report['neck_fit']['status'] == 'skipped':
        shift = _alignment_shift(result, report, body)
        if shift is not None:
            try:
                candidate, candidate_report = transplant(pmx, head, body, Options(offset=(0, shift, 0)))
            except TransplantError:
                candidate_report = None
            if candidate_report and candidate_report['neck_fit']['status'] in ('fitted', 'bridged'):
                result, report = candidate, candidate_report
                alignment = {'applied': True, 'vertical_shift': shift}
    report['automatic_alignment'] = alignment
    return result, report


def assess_result(report, skin_color, missing_textures=()):
    """Expose incomplete operations instead of claiming a finished model."""
    issues = []
    if report['neck_fit']['status'] not in ('fitted', 'bridged'):
        issues.append('颈部连接需要检查')
    if report['scale_estimate']['source'] == 'unchanged':
        issues.append('头部比例缺少可靠定位')
    if 'skin_bridge' in report['neck_fit'] and skin_color['status'] == 'skipped':
        issues.append('颈部肤色需要检查')
    if missing_textures:
        issues.append('存在缺失贴图')
    return {'status': 'review' if issues else 'ready', 'issues': issues}

"""Conservative neck-rim detection and local fitting, independent of bpy."""
from collections import Counter, defaultdict
import math


def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))


def find_rim(points, material_faces, anchor, width, up=1):
    """Find a closed, horizontally enclosing rim near the head joint.

    Work per material: UV splits are grouped geometrically, while facial
    overlays and duplicate surfaces cannot cancel the skin boundary.
    """
    horizontal = [i for i in range(3) if i != up]
    candidates = []
    for material, faces in material_faces.items():
        ids = {i for face in faces for i in face}
        geo = {i: tuple(round(float(x), 6) for x in points[i]) for i in ids}
        members = defaultdict(list)
        for i in ids:
            members[geo[i]].append(i)
        edges = Counter(tuple(sorted((geo[a],geo[b]))) for f in faces
                        for a,b in zip(f, f[1:]+f[:1]) if geo[a] != geo[b])
        adjacent = defaultdict(set)
        for (a,b),count in edges.items():
            if count == 1:
                adjacent[a].add(b)
                adjacent[b].add(a)
        seen = set()
        for seed in adjacent:
            if seed in seen:
                continue
            component, stack = set(), [seed]
            while stack:
                key = stack.pop()
                if key in component:
                    continue
                component.add(key)
                stack.extend(adjacent[key]-component)
            seen.update(component)
            if not 8 <= len(component) <= 256 or any(len(adjacent[k]) != 2 for k in component):
                continue
            center = tuple(sum(k[i] for k in component)/len(component) for i in range(3))
            bounds = [(min(k[i] for k in component), max(k[i] for k in component)) for i in range(3)]
            # A neck rim encloses the head joint in the horizontal plane.
            if any(not low < anchor[i] < high for i,(low,high) in enumerate(bounds) if i != up):
                continue
            if distance(center,anchor) > width*0.75:
                continue
            if any(not width*0.2 < bounds[i][1]-bounds[i][0] < width*1.8 for i in horizontal):
                continue
            if bounds[up][1]-bounds[up][0] > width*0.8:
                continue
            start = min(component)
            loop, previous, current = [], None, start
            while current not in loop:
                loop.append(current)
                choices = adjacent[current]-({previous} if previous is not None else set())
                following = sorted(choices)[0]
                previous,current = current,following
            if current != start or len(loop) != len(component):
                continue
            area = sum(a[horizontal[0]]*b[horizontal[1]]-b[horizontal[0]]*a[horizontal[1]]
                       for a,b in zip(loop,loop[1:]+loop[:1]))
            if area < 0:
                loop.reverse()
            candidates.append({'material':material,'points':loop,'members':[members[k] for k in loop],
                               'center':center,'score':distance(center,anchor)})
    return min(candidates,key=lambda c:c['score']) if candidates else None


def fit_plan(points, head_faces, body_faces, anchor, width, up=1):
    donor = find_rim(points,head_faces,anchor,width,up)
    body = find_rim(points,body_faces,anchor,width,up)
    if not donor or not body:
        return None
    hp,bp = donor['points'],body['points']
    # Initially restrict to equal closed rings. Failing to find one is safer
    # than guessing a match between an eye/mouth hole and an unrelated loop.
    if len(hp) != len(bp):
        return None
    shift = min(range(len(bp)),key=lambda s:sum(distance(hp[i],bp[(i+s)%len(bp)])**2 for i in range(len(hp))))
    targets = [bp[(i+shift)%len(bp)] for i in range(len(hp))]
    references = [body['members'][(i+shift)%len(bp)][0] for i in range(len(hp))]
    max_delta = max(distance(a,b) for a,b in zip(hp,targets))
    if max_delta > width*0.75:
        return None
    falloff = width*0.6
    # Change only the detected skin material, never nearby hair/hat/overlays.
    head_ids = {i for face in head_faces[donor['material']] for i in face}
    boundary = {i: j for j,members in enumerate(donor['members']) for i in members}
    changes = {}
    for index in head_ids:
        point = points[index]
        nearest = min(range(len(hp)),key=lambda j:distance(point,hp[j]))
        dist = distance(point,hp[nearest])
        if dist >= falloff:
            continue
        t = max(0,1-dist/falloff)
        t = t*t*(3-2*t)
        if index in boundary:
            nearest,t = boundary[index],1.0
        delta = tuple((targets[nearest][i]-hp[nearest][i])*t for i in range(3))
        changes[index] = {'delta':delta,'weight_factor':t,'reference':references[nearest],
                          'boundary':index in boundary or dist < width*1e-5}
    return {'changes':changes,'head_rim':donor,'body_rim':body,'pairs':list(zip(hp,targets)),
            'max_delta':max_delta,'rim_vertices':len(hp)}


def bridge_plan(points, head_faces, body_faces, anchor, width, up=1):
    """Triangulate a strip between two unequal, separated neck loops.

    Keep both original rims fixed. A monotonic dynamic-programming path uses
    every boundary edge once, avoiding holes from nearest-vertex snapping.
    """
    head=find_rim(points,head_faces,anchor,width,up)
    body=find_rim(points,body_faces,anchor,width,up)
    if not head or not body or len(head['points'])==len(body['points']):
        return None
    hp,bp=head['points'],body['points']
    if max(max(min(distance(a,b) for b in bp) for a in hp),
           max(min(distance(a,b) for a in hp) for b in bp))>width*.75:
        return None
    # Overlapping/crossing loops need remeshing, not a guessed strip.
    horizontal=[k for k in range(3) if k!=up]
    def height_at(point,loop):
        closest=None
        for a,b in zip(loop,loop[1:]+loop[:1]):
            delta=[b[k]-a[k] for k in horizontal]
            length=sum(v*v for v in delta)
            t=max(0,min(1,sum((point[k]-a[k])*v for k,v in zip(horizontal,delta))/length)) if length else 0
            separation=sum((point[k]-a[k]-v*t)**2 for k,v in zip(horizontal,delta))
            if closest is None or separation<closest[0]:closest=(separation,a[up]+(b[up]-a[up])*t)
        return closest[1]
    if any(p[up]-height_at(p,bp)<=width*1e-5 for p in hp) or any(height_at(p,hp)-p[up]<=width*1e-5 for p in bp):
        return None
    shift=min(range(len(bp)),key=lambda j:distance(hp[0],bp[j]))
    body=dict(body)
    body['points']=bp[shift:]+bp[:shift]
    body['members']=body['members'][shift:]+body['members'][:shift]
    bp=body['points'];n,m=len(hp),len(bp)
    cost={(0,0):0.0};previous={}
    for i in range(n+1):
        for j in range(m+1):
            if (i,j) not in cost:continue
            for ni,nj in ((i+1,j),(i,j+1)):
                if ni>n or nj>m:continue
                candidate=cost[i,j]+distance(hp[ni%n],bp[nj%m])**2
                if candidate<cost.get((ni,nj),math.inf):
                    cost[ni,nj]=candidate;previous[ni,nj]=(i,j)
    triangles=[];i,j=n,m
    while i or j:
        a,b=previous[i,j]
        if a!=i:triangles.append((a%n,i%n,n+j%m))
        else:triangles.append((i%n,n+j%m,n+b%m))
        i,j=a,b
    # Choose winding opposite the existing head boundary, including mirrored
    # or inward-wound source meshes. The geometry grouping matches UV splits.
    keys={index:point for point,ids in zip(head['points'],head['members']) for index in ids}
    directed={(keys[a],keys[b]) for face in head_faces[head['material']]
              for a,b in zip(face,face[1:]+face[:1]) if a in keys and b in keys}
    if (hp[0],hp[1]) in directed:
        triangles=[(a,c,b) for a,b,c in triangles]
    return {'head_rim':head,'body_rim':body,'triangles':triangles,
            'source_indices':[ids[0] for ids in head['members']+body['members']]}

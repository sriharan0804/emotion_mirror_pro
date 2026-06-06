import math


class CentroidTracker:
    def __init__(self, max_distance=80):
        self.next_id = 1
        self.objects = {}
        self.max_distance = max_distance

    def update(self, boxes):
        updated_objects = {}
        used_ids = set()

        for box in boxes:
            x, y, w, h = box

            cx = x + w // 2
            cy = y + h // 2

            best_id = None
            best_distance = float("inf")

            for object_id, old_data in self.objects.items():
                old_cx, old_cy, _ = old_data

                if object_id in used_ids:
                    continue

                distance = math.sqrt(
                    (cx - old_cx) ** 2 +
                    (cy - old_cy) ** 2
                )

                if distance < best_distance:
                    best_distance = distance
                    best_id = object_id

            if best_id is not None and best_distance < self.max_distance:
                updated_objects[best_id] = (cx, cy, box)
                used_ids.add(best_id)
            else:
                updated_objects[self.next_id] = (cx, cy, box)
                used_ids.add(self.next_id)
                self.next_id += 1

        self.objects = updated_objects
        return self.objects
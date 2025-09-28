import cadquery as cq

# --- Part 1: Hammer ---
part_1_length = 0.75 * 0.75  # Sketch length scaled
part_1_width = 0.75 * 0.75  # Sketch width scaled
part_1_height = 0.75

# Create the base plate
plate = cq.Workplane("XY").rect(part_1_length, part_1_width).extrude(part_1_height)

# --- Assembly (if needed, based on the single part) ---
# In this case, there's only one part, so no assembly is needed.
# If there were multiple parts, you would use .union() or .cut() to combine them.

# --- Final Result ---
result = plate

# Export to STL
cq.exporters.export(result, 'bob1.stl')
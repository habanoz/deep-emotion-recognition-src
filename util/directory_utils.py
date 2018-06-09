


def get_path_parts(sample_path):
    """Given a full path to a sample, return its parts."""
    print(sample_path)
    parts = sample_path.replace('\\', '/').split('/')
    part_length=len(parts)
    filename = parts[part_length-1]
    filename_no_ext = filename.split('.')[0]
    classname = parts[part_length-2]
    train_or_test = parts[part_length-3]
    root="/".join(parts[0:part_length-3])

    return root, train_or_test, classname, filename_no_ext, filename
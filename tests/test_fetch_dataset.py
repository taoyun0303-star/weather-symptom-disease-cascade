from scripts.fetch_dataset import download_dataset, md5sum


def test_download_dataset_can_atomically_replace_a_windows_temp_file(tmp_path):
    source = tmp_path / "source.csv"
    source.write_bytes(b"column\nvalue\n")
    output = tmp_path / "nested" / "dataset.csv"

    download_dataset(source.as_uri(), output, md5sum(source), force=False)

    assert output.read_bytes() == source.read_bytes()

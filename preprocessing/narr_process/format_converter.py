from pycaption import DFXPReader, WebVTTReader, SRTReader, DFXPWriter, WebVTTWriter, SRTWriter

def convert_subtitle_format(src_fname, dst_fname, 
        src_format='ttml', dst_format='srt'):
    if src_format == 'ttml':
        reader = DFXPReader()
    elif src_format == 'vtt':
        reader = WebVTTReader()
    elif src_format == 'srt':
        reader = SRTReader()
    else:
        print('Unsupported Subtitle Format:', src_format)
        return

    if dst_format == 'srt':
        writer = SRTWriter()
    elif dst_format == 'vtt':
        writer = WebVTTWriter()
    elif dst_format == 'ttml':
        writer = DFXPWriter()
    else:
        print('Unsupported Subtitle Format:', dst_format)
        return

    with open(src_fname, 'r') as f_src:
        content = reader.read(f_src.read())
    with open(dst_fname, 'w') as f_dst:
        f_dst.write(writer.write(content))

if __name__ == '__main__':
    src_fname = '105222_3vQP3BXHKso.vtt'
    dst_fname = '105222_3vQP3BXHKso.srt'
    convert_subtitle_format(src_fname, dst_fname, 'vtt', 'srt')

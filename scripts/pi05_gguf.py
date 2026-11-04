"""GGUF reader with bulk numeric metadata arrays, preserving contents() indexing."""
import gguf


class _NumericField(gguf.ReaderField):
    def contents(self, index_or_slice=slice(None)):
        if isinstance(index_or_slice, int):
            index_or_slice = int(index_or_slice)
        return self.parts[self.data[0]][index_or_slice].tolist()

    def raw_bytes(self):
        if self.types != [gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT8]:
            return bytes(self.contents())
        return self.parts[self.data[0]].tobytes()


class GGUFReader(gguf.GGUFReader):
    # Keep the installed reader's version/endian/tensor validation. Only replace
    # its per-element allocation for top-level numeric metadata arrays.
    def _get_field_parts(self, offset, raw_type):
        nested = getattr(self, "_nested_array", False)
        if not nested and raw_type == gguf.GGUFValueType.ARRAY:
            subtype = self._get(offset, "uint32")
            length = self._get(offset + 4, "uint64")
            if len(subtype) != 1 or len(length) != 1:
                raise ValueError("Truncated GGUF array header")
            kind = gguf.GGUFValueType(int(subtype[0]))
            dtype = self.gguf_scalar_to_np.get(kind)
            if dtype is not None:
                count = int(length[0])
                values = self._get(offset + 12, dtype, count)
                if len(values) != count:
                    raise ValueError("Truncated GGUF numeric array")
                return 12 + values.nbytes, [subtype, length, values], [2], [gguf.GGUFValueType.ARRAY, kind]
        self._nested_array = True
        try:
            return super()._get_field_parts(offset, raw_type)
        finally:
            self._nested_array = nested

    def _push_field(self, field, skip_sum=False):
        if (len(field.types) == 2 and field.types[0] == gguf.GGUFValueType.ARRAY
                and field.types[1] in self.gguf_scalar_to_np and len(field.data) == 1):
            field = _NumericField(field.offset, field.name, field.parts, field.data, field.types)
        return super()._push_field(field, skip_sum)

#ifndef UBJW_H
#define UBJW_H

#ifdef __cplusplus
extern "C" {
#endif

#include<inttypes.h>
#include<stdio.h>

	typedef enum
	{
		NULLTYPE = 'Z',
		NOOP = 'N',
		BOOL_TRUE = 'T',
		BOOL_FALSE = 'F',

		STRING = 'S',
		CHAR = 'C',

		INT8 = 'i',
		UINT8 = 'U',
		INT16 = 'I',
		INT32 = 'l',
		INT64 = 'L',
		FLOAT32 = 'd',
		FLOAT64 = 'D',
		HIGH_PRECISION = 'H',

		ARRAY = '[',
		OBJECT = '{',

		MIXED = 0     //This should not be used to be written to, only to control arrays
	} UBJW_TYPE;

	struct ubjw_context_t;//this should JUST be container. open array makes a copy of the current container...container can be default NULL which has special rules...no dont' do that

	//mixed children,no seperator


	struct ubjw_context_t* ubjw_open_callback(void* userdata,
		size_t(*write_cb)(const void* data, size_t size, size_t count, void* userdata),
		int(*close_cb)(void* userdata),
		void(*error_cb)(const char* error_msg)
		);
	struct ubjw_context_t* ubjw_open_file(FILE*);
	struct ubjw_context_t* ubjw_open_memory(uint8_t* dst_b, uint8_t* dst_e);

	size_t ubjw_close_context(struct ubjw_context_t* ctx);

	void ubjw_write_string(struct ubjw_context_t* dst, const char* out);
	void ubjw_write_char(struct ubjw_context_t* dst, char out);

	void ubjw_write_uint8(struct ubjw_context_t* dst, uint8_t out);
	void ubjw_write_int8(struct ubjw_context_t* dst, int8_t out);
	void ubjw_write_int16(struct ubjw_context_t* dst, int16_t out);
	void ubjw_write_int32(struct ubjw_context_t* dst, int32_t out);
	void ubjw_write_int64(struct ubjw_context_t* dst, int64_t out);
	void ubjw_write_high_precision(struct ubjw_context_t* dst, const char* hp);

	void ubjw_write_integer(struct ubjw_context_t* dst, int64_t out);

	void ubjw_write_float32(struct ubjw_context_t* dst, float out);
	void ubjw_write_float64(struct ubjw_context_t* dst, double out);

	void ubjw_write_floating_point(struct ubjw_context_t* dst, double out);

	void ubjw_write_noop(struct ubjw_context_t* dst);
	void ubjw_write_null(struct ubjw_context_t* dst);
	void ubjw_write_bool(struct ubjw_context_t* dst, uint8_t out);

	void ubjw_begin_array(struct ubjw_context_t* dst, UBJW_TYPE type, size_t count);
	void ubjw_begin_object(struct ubjw_context_t* dst, UBJW_TYPE type, size_t count);
	void ubjw_write_key(struct ubjw_context_t* dst, const char* key);
	void ubjw_end(struct ubjw_context_t* dst);

	//output an efficient buffer of types
	void ubjw_write_buffer(struct ubjw_context_t* dst, const uint8_t* data, UBJW_TYPE type, size_t count);

#ifdef __cplusplus
}
#endif

#endif

#include "ubj.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static const char expected_mixed_dynamic_array[]= "[i\x05Si\x05helloi\x0a]";
void test_mixed_dynamic_array(ubjw_context_t* ctx)
{
	ubjw_begin_array(ctx, UBJ_MIXED, 0);
	ubjw_write_int8(ctx, 5);
	ubjw_write_string(ctx, "hello");
	ubjw_write_int8(ctx, 10);
	ubjw_end(ctx);
}
static const char expected_mixed_static_array[] = "[#i\x03i\x05Si\x05helloi\x0a";
void test_mixed_static_array(ubjw_context_t* ctx)
{
	ubjw_begin_array(ctx, UBJ_MIXED, 3);
	ubjw_write_int8(ctx, 5);
	ubjw_write_string(ctx, "hello");
	ubjw_write_int8(ctx, 10);
	ubjw_end(ctx);
}

static const char expected_typed_static_array[] = "[$U#i\x03\xFF\xAA\xDD";
void test_typed_static_array(ubjw_context_t* ctx)
{
	ubjw_begin_array(ctx, UBJ_UINT8, 3);
	ubjw_write_uint8(ctx, 0xFF);
	ubjw_write_uint8(ctx, 0xAA);
	ubjw_write_uint8(ctx, 0xDD);
	ubjw_end(ctx);
}
static const char expected_buffer_typed_static_array[] = "[$U#i\x03\xFF\xAA\xDD";
void test_buffer_typed_static_array(ubjw_context_t* ctx)
{
	static const uint8_t data[] = {0xFF,0xAA,0xDD};
	ubjw_write_buffer(ctx,data,UBJ_UINT8,3);
}

static const char expected_mixed_dynamic_object[] = "{i\x2k1i\x05i\x2k2Si\x05helloi\x2k3i\x0a}";
void test_mixed_dynamic_object(ubjw_context_t* ctx)
{
	ubjw_begin_object(ctx, UBJ_MIXED, 0);
		ubjw_write_key(ctx,"k1");ubjw_write_int8(ctx, 5);
		ubjw_write_key(ctx,"k2");ubjw_write_string(ctx, "hello");
		ubjw_write_key(ctx,"k3");ubjw_write_int8(ctx, 10);
	ubjw_end(ctx);
}
static const char expected_mixed_static_object[] = "{#i\x3i\x2k1i\x05i\x2k2Si\x05helloi\x2k3i\x0a";
void test_mixed_static_object(ubjw_context_t* ctx)
{
	ubjw_begin_object(ctx, UBJ_MIXED, 3);
		ubjw_write_key(ctx,"k1");ubjw_write_int8(ctx, 5);
		ubjw_write_key(ctx,"k2");ubjw_write_string(ctx, "hello");
		ubjw_write_key(ctx,"k3");ubjw_write_int8(ctx, 10);
	ubjw_end(ctx);
}
static const char expected_typed_static_object[] = "{$U#i\x3i\x2k1\xFFi\x2k2\xAAi\x2k3\xDD";
void test_typed_static_object(ubjw_context_t* ctx)
{
	ubjw_begin_object(ctx, UBJ_UINT8, 3);
		ubjw_write_key(ctx,"k1");ubjw_write_uint8(ctx, 0xFF);
		ubjw_write_key(ctx,"k2");ubjw_write_uint8(ctx, 0xAA);
		ubjw_write_key(ctx,"k3");ubjw_write_uint8(ctx, 0xDD);
	ubjw_end(ctx);
}
//Todo: arrays of nulltype sizes.

static const char expected_integer_types[] = "[i\x33U\xFFI\x33\x33l\x33\x33\x33\x33L\x33\x33\x33\x33\x33\x33\x33\x33]";
void test_integer_types(ubjw_context_t* ctx)
{
	ubjw_begin_array(ctx,UBJ_MIXED,0);
		ubjw_write_int8(ctx,0x33);
		ubjw_write_uint8(ctx,0xFF);
		ubjw_write_int16(ctx,0x3333);
		ubjw_write_int32(ctx,0x33333333);
		ubjw_write_int64(ctx,0x3333333333333333LL);
	ubjw_end(ctx);
}

static const char expected_integer_compressor[] = "[i\x33U\xFFI\x33\x33l\x33\x33\x33\x33L\x33\x33\x33\x33\x33\x33\x33\x33]";
void test_integer_compressor(ubjw_context_t* ctx)
{
	ubjw_begin_array(ctx,UBJ_MIXED,0);
		ubjw_write_integer(ctx,0x33);
		ubjw_write_integer(ctx,0xFF);
		ubjw_write_integer(ctx,0x3333);
		ubjw_write_integer(ctx,0x33333333);
		ubjw_write_integer(ctx,0x3333333333333333LL);
	ubjw_end(ctx);
}



/*typedef enum
{
UBJ_NULLTYPE = 'Z',
UBJ_NOOP = 'N',
UBJ_BOOL_TRUE = 'T',
UBJ_BOOL_FALSE = 'F',

UBJ_STRING = 'S',
UBJ_CHAR = 'C',

UBJ_INT8 = 'i',
UUBJ_INT8 = 'U',
UBJ_INT16 = 'I',
UBJ_INT32 = 'l',
UBJ_INT64 = 'L',
UBJ_FLOAT32 = 'd',
UBJ_FLOAT64 = 'D',
UBJ_HIGH_PRECISION = 'H',

UBJ_ARRAY = '[',
UBJ_OBJECT = '{',

UBJ_MIXED = 0
*/
void printhex(const char* buffer, size_t sz)
{
	size_t i;
	for (i = 0; i < sz; i++)
	{
		if(isprint(buffer[i]))
		{
			printf("%c",buffer[i]);
		}
		else
		{
			printf("\\x%X",(unsigned char)buffer[i]);
		}
	}
}

void comparemem(const uint8_t* memory,const uint8_t* expected,size_t sz,size_t n)
{
	int c = memcmp(memory, expected, sz);
	if (c == 0 && n==sz)
	{
		printf("PASSED\n");
	}
	else
	{
		printf("FAILED (");
		printhex(memory, n);
		printf("?=");
		printhex(expected, sz);
		printf(")\n");
	}
}

void run_test(const char* name, void(*tp)(ubjw_context_t* ctx), const char* expected)
{
	size_t sz=strlen(expected);
	printf("%s :", name);
	uint8_t* memory = malloc(2 * sz);
	ubjw_context_t* ctx = ubjw_open_memory(memory, memory + 2 * sz);
	tp(ctx);
	size_t n = ubjw_close_context(ctx);//returns total written?  maybe?
	comparemem(memory,expected,sz,n);

	ubjr_context_t* rctx = ubjr_open_memory(expected,expected+sz);
	ubjw_context_t* wctx = ubjw_open_memory(memory,memory+2*sz);
	ubjr_dynamic_t filestruct=ubjr_read_dynamic(rctx);
	ubjrw_write_dynamic(ctx,filestruct,0);

	printf("%s_readwrite :",name);
	comparemem(memory,expected,sz,n);

	ubjr_cleanup_dynamic(&filestruct);

	ubjr_close_context(rctx);
	ubjw_close_context(wctx);
	
	//run readwrite test

	free(memory);
}


int main(int argc, char** argv)
{
	run_test("mixed_dynamic_array", test_mixed_dynamic_array, expected_mixed_dynamic_array);
	run_test("mixed_static_array", test_mixed_static_array, expected_mixed_static_array);
	run_test("typed_static_array", test_typed_static_array, expected_typed_static_array);
	run_test("buffer_typed_static_array",test_buffer_typed_static_array,expected_buffer_typed_static_array);
	
	run_test("mixed_dynamic_object", test_mixed_dynamic_object, expected_mixed_dynamic_object);
	run_test("mixed_static_object", test_mixed_static_object, expected_mixed_static_object);
	run_test("typed_static_object", test_typed_static_object, expected_typed_static_object);
	
	run_test("integer_types",test_integer_types,expected_integer_types);

	return 0;
}

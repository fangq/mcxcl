#include<stdio.h>
#include<ubj.h>

void print_help()
{
	
}

int main(int argc,char** argv)
{
	FILE* fp;
	if(argc > 1)
	{
		if(argv[1][0]=='-' && argv[1][1]=='-')
		{
			fp=stdin;
		}
		else
		{
			fp=fopen(argv[1],"rb");
		}
	}
	else
	{
		print_help();
		return 0;
	}
	ubjr_context_t* rctx=ubjr_open_file(fp);
//	const char testdump[] = "[i\x33U\xFFI\x33\x33l\x33\x33\x33\x33L\x33\x33\x33\x33\x33\x33\x33\x33]";
//	ubjr_context_t* rctx = ubjr_open_memory(testdump, testdump + strlen(testdump));

	ubjr_dynamic_t dyn=ubjr_read_dynamic(rctx);

	ubjw_context_t* wctx = ubjw_open_file(stdout);
	ubjrw_write_dynamic(wctx, dyn,0);

	ubjr_close_context(rctx);
	ubjw_close_context(wctx);
	return 0;
}
#include<iostream>
#include<inttypes.h>
#include <stdexcept>
#include <CL/cl_platform.h>
using namespace std;

class expected_wrong_format_exception : public runtime_error 
{
public:
	expected_wrong_format_exception(const string& __arg) : runtime_error(__arg) {}
};

class node
{
public:
	char type;
	//const uint8_t* begin; the stream begin
	const void* privdata;
};

int64_t parse_integer(const int8_t type,const uint8_t*& it)
{
	uint8_t first=*it++;
	int64_t vs=*((const int8_t*)&first); //sign-extended 64-bit int
	uint64_t vu=*(const uint64_t*)&vs;   //unsignedify it for left shifts
	switch(type)
	{
	case 'L':
	{
		vu<<=8ULL;vu|=*it++;
		vu<<=8ULL;vu|=*it++;
		vu<<=8ULL;vu|=*it++;
		vu<<=8ULL;vu|=*it++;
	}
	case 'l':
	{
		vu<<=8ULL;vu|=*it++;
		vu<<=8ULL;vu|=*it++;
	}
	case 'I':
	{
		vu<<=8ULL;vu|=*it++;
	}
	case 'i':
	{
		return *(const int64_t*)&vu;
	}
	case 'U':
	{
		return first;
	}
	case 'H':
	{
		it--;
		int64_t hsize=parse_integer(type,it);
		it+=hsize;
		//for(int64_t i=0;i<hsize;i++) it++;
		throw std::runtime_error("high-precision number type not supported by this parser");
	}
	default:
	{
		it--;
		throw expected_wrong_format_exception("does not match integer type");
	}
	};
}

double parse_float(const int8_t type,const uint8_t*& it)
{
	uint64_t vu=0;   //unsignedify it for left shifts
	switch(type)
	{
	case 'D':
	{
		vu<<=8;vu|=*it++;
		vu<<=8;vu|=*it++;
		vu<<=8;vu|=*it++;
		vu<<=8;vu|=*it++;
	}
	case 'd':
	{
		vu<<=8;vu|=*it++;
		vu<<=8;vu|=*it++;
		vu<<=8;vu|=*it++;
		vu<<=8;vu|=*it++;
		break;
	}
	default:
	{
		throw expected_wrong_format_exception("does not match float");
	}
	};
	if(type=='d')
	{
		uint32_t vu32=vu;
		return *(const float*)&vu32;//inC++ mode reinterpretcast
	}
	else
	{
		return *(const double*)&vu;
	}
}
void parse_null(int8_t type,const uint8_t*& it)
{
	if(*it=='Z')
	{
		it++;
		return;
	}
	//not null...throw?
}
void parse_noop(int8_t type,const uint8_t*& it)
{
	if(*it=='N')
	{
		it++;
		return;
	}
}
bool parse_bool(int8_t type,const uint8_t*& it)
{
	switch(*it)
	{
	case 'T':
		it++;
		return true;
	case 'F':
		it++;
		return false;
	default:
		break;
		//throw?
	};
}

string parse_string(int8_t type,const uint8_t*& it)
{
	int8_t dt=*it++;
	switch(dt)
	{
	case 'C':
	{
		return string()+*(const char*)it++;
	}
	case 'S':
	{
		int64_t slen=parse_integer(it);
		string s((const char*)it,slen);
		it+=slen;
		return s;
	}
	default:
	{
		break;
	}
	};
}


struct _container_header_t
{
	bool count_known;
	bool type_known;
	char type;
	int64_t count;
	_container_header_t():
		count_known(false),
		type_known(false),
		type('Z'),
		count(0)
	{}
};
_container_header_t _parse_container_header(const uint8_t*& it)
{
	_container_header_t ch;
	if(*it=='$')
	{
		ch.type_known=true;
		it++;
		ch.type=*it++;
	}
	if(*it=='#')
	{
		it++;
		ch.count_known=true;
		ch.count=parse_integer(it);
	}
	else if(ch.type_known)
	{
		//throw exception Type specified without count
	}	
	return ch;
}
bool _has_payload(const int8_t s)
{
	return !(s=='T' || s=='F' || s=='N' || s=='Z');
}
int _payload_constant_size(const int8_t s)
{
	switch(s)
	{
		case 'S':
		case 'H':
		case '[':
		case '{':
			return -1;
		case 'Z':
		case 'N':
		case 'T':
		case 'F':
			return 0;
		case 'i':
		case 'U':
		case 'C':
			return 1;
		case 'I':
			return 2;
		case 'l':
		case 'd':
			return 4;
		case 'L':
		case 'D':
			return 8;
		default:
			break;
			//throw 
	};	
}

void parse_array(int8_t type,const uint8_t*& it)
{
	if(type != '[')
	{
		//throw
		return;
	}
}

void parse_object(int8_t type,const uint8_t*& it)
{
	
}
void _parse_dynamic_noheader(const int8_t type,const uint8_t*& it)
{
	switch(type)
	{
		case 'Z':
			parse_null(type,it);
		case 'N':
			parse_noop(type,it);
		case 'T':
			parse_bool(type,it);
		case 'F':
			parse_bool(type,it);
		case 'i':
		case 'U':
		case 'I':
		case 'l':
		case 'L':
		case 'H':
			parse_integer(type,it);
		case 'D':
			parse_float(type,it);
		case 'd':
			parse_float(type,it);
		case 'C':
		case 'S':
			parse_string(type,it);
		case '[':
			parse_array(type,it);
		case '{':
			parse_object(type,it);
		default:
			break;
			//throw 
	};
}
void parse_dynamic(const uint8_t*& it)
{
	int8_t type=*it;
	_parse_dynamic_noheader();
}
//no streaming mode yet...this could maybe be done with an iterator or something.

void int64_test()
{
	const char* strings[]={"L\xFF\xFF\x0F\xFF\x00\x00\x00\x00"};
	const uint8_t* it=(const uint8_t*)strings[0];
	
	uint64_t v=	0xFFFF0FFF00000000ULL;
	
	cout << parse_integer(it) << " ?= " << *reinterpret_cast<int64_t*>(&v) << endl;
}

void double_test()
{
	const char* strings[] = {"D\x3F\xD5\x55\x55\x55\x55\x55\x55"}; // double is 1/3
	const uint8_t* it=(const uint8_t*)strings[0];
	
	uint64_t v=0x3FD5555555555555ULL;
	
	//cout << parse_float(it) << " ?= " << *reinterpret_cast<double*>(&v) << endl;	
}
void string_test()
{
	const char* strings[] = {"Si\x0DHello, world!"};
	const uint8_t* it=(const uint8_t*)strings[0];
	
	string v="Hello, world!";
	
	//cout << parse_string(it) << " ?= " << v << endl;
}


int main()
{
	int64_test();
	double_test();
	string_test();
	
	return 0;
}



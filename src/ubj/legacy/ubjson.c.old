
class node
{
public:
	char type;
};

int64_t parse_finite_integer(it)
{
	int8_t type=*it++;
	uint8_t first=*it++;
	int64_t vs=*((const int8_t*)&first); //sign-extended 64-bit int
	uint64_t vu=*(const uint64_t*)&vs;   //unsignedify it for left shifts
	switch(type)
	{
	case 'L':
		vu<<=8;vu|=*it++;
		vu<<=8;vu|=*it++;
		vu<<=8;vu|=*it++;
		vu<<=8;vu|=*it++;
	case 'l':
		vu<<=8;vu|=*it++;
		vu<<=8;vu|=*it++;
	case 'I'
		vu<<=8;vu|=*it++;
	case 'i':
		return *(const int64_t*)&vu;
	case 'U':
		return first;
	default:
		it--;it--;
		//throw expected_wrong_format_exception(does not match definite integer type
	};
}

double parse_float(it)
{
	int8_t type=*it++;
	uint64_t vu=0;   //unsignedify it for left shifts
	switch(type)
	{
	case 'D':
		vu<<=8;vu|=*it++;
		vu<<=8;vu|=*it++;
		vu<<=8;vu|=*it++;
		vu<<=8;vu|=*it++;
	case 'd':
		vu<<=8;vu|=*it++;
		vu<<=8;vu|=*it++;
		vu<<=8;vu|=*it++;
		vu<<=8;vu|=*it++;
		break;
	default:
		//throw expected_wrong_format_exception(does not match float)
	};
	if(type=='d')
	{
		uint32_t vu32=vu;
		return *(const float*)&vu32;
	}
	else
	{
		return *(const double*)&vu;
	}
}

//no streaming mode yet...this could maybe be done with an iterator or something.

std::unique_ptr<node_t> parse(it)
{
	
}


int main()
{
	node_t n=parse(b,e);
	return 0;
}



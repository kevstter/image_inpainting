OPTS= -O3 -ftree-vectorize -funroll-loops -fopenmp

GBIN=expl_diff
HBIN=impl_diff
IBIN=expl_tv
JBIN=sbdf1_tv
KBIN=cnab_tv
LBIN=sbdf2_tv

GSRC=io_png.c img_fun.c expl_diff.c 
HSRC=io_png.c img_fun.c impl_diff.c 
ISRC=io_png.c img_fun.c expl_tv.c 
JSRC=io_png.c img_fun.c sbdf1_tv.c 
KSRC=io_png.c img_fun.c cnab_tv.c 
LSRC=io_png.c img_fun.c sbdf2_tv.c 

CFLAGS= $(OPTS) -Wall -std=c99 -Wextra -pedantic -Wno-strict-overflow

LDFLAGS= -lpng -lfftw3 -lm -lgomp


all: $(GBIN) $(HBIN) $(IBIN) $(JBIN) $(KBIN) $(LBIN)

$(GBIN): $(GSRC:.c=.o)
	$(CC) $(GSRC:.c=.o) -o $(GBIN) $(LDFLAGS) 

$(HBIN): $(HSRC:.c=.o)
	$(CC) $(HSRC:.c=.o) -o $(HBIN) $(LDFLAGS)

$(IBIN): $(ISRC:.c=.o)
	$(CC) $(ISRC:.c=.o) -o $(IBIN) $(LDFLAGS) 

$(JBIN): $(JSRC:.c=.o)
	$(CC) $(JSRC:.c=.o) -o $(JBIN) $(LDFLAGS) 

$(KBIN): $(KSRC:.c=.o)
	$(CC) $(KSRC:.c=.o) -o $(KBIN) $(LDFLAGS) 

$(LBIN): $(LSRC:.c=.o)
	$(CC) $(LSRC:.c=.o) -o $(LBIN) $(LDFLAGS) 

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(GSRC:.c=.o)
	rm -f $(HSRC:.c=.o)	
	rm -f $(ISRC:.c=.o)		
	rm -f $(JSRC:.c=.o)		
	rm -f $(KSRC:.c=.o)		
	rm -f $(LSRC:.c=.o)		
	rm -f $(GBIN)
	rm -f $(HBIN)
	rm -f $(IBIN)
	rm -f $(JBIN)
	rm -f $(KBIN)
	rm -f $(LBIN)



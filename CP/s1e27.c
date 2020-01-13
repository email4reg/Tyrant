#include <stdio.h>

int main(void)
{
    int num = 0x12345678;
    unsigned char *p = (unsigned char *)&num;

    if (*p == 0x78)
    {
        printf("您的机器采用小端字节序。\n");
    }
    else
    {
        printf("您的机器采用大端字节序。\n");
    }

    printf("0x12345678 在内存中依次存放为：0x%x 0x%x 0x%x 0x%x\n", p[0], p[1], p[2], p[3]);

    return 0;
}
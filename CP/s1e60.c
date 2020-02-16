#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

int main(void)
{
    FILE *fp;
    int ch;

    if ((fp = fopen("date.txt", "w")) == NULL)
    {
        printf("标准输出\n");
        fputs("打开文件失败!\n", stderr);
        // perror("打开文件失败, 原因是");
        // fprintf(stderr, "出错啦, 原因是%s\n", stderr(errno));
        exit(EXIT_FAILURE);
    }

    while (1)
    {
        ch = fgetc(fp);
        if (feof(fp))
        {
            break;
        }
        putchar(ch);
    }
    
    fputc('c', fp);
    if (ferror(fp))
    {
        fputs("出错啦!", stderr);
    }

    // clearerr(fp);
    // errno
    // printf("打开文件失败, 原因是: %d\n", errno);

    fclose(fp);
}

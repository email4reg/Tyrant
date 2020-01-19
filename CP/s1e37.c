#include <stdio.h>
#include <stdlib.h>

int main1(void)
{
    int *ptr = NULL;
    int n;

    printf("请输入待录入整数的个数: ");
    scanf("%d", &n);

    ptr = (int *)malloc(n * sizeof(int));

    if (ptr == NULL)
    {
        printf("申请动态内存失败\n");
        exit(1);
    }

    for (int i = 0; i < n; i++)
    {
        printf("请录入第%d个整数: ", i + 1);
        scanf("%d", ptr + i); // == &ptr[i]
    }

    printf("你录入的整数是: ");
    for (int i = 0; i < n; i++)
    {
        printf("%d ", *(ptr + i)); // == ptr[i]
    }
    putchar('\n');

    free(ptr);

    return 0;
}

int main(void)
{
    void *block;
    int i, count;
    size_t maximum = 0;
    size_t blocksize[] = {1024 * 1024, 1024, 1};

    // 下面从大到小依次尝试
    // 先尝试以1024 * 1024为扩大粒度去申请内存空间
    // 当malloc返回NULL时，将扩大的粒度缩小为1024继续尝试
    // 最终精确到1个字节的粒度扩大maximum的尺寸
    for (i = 0; i < 3; i++)
    {
        for (count = 1;; count++)
        {
            block = malloc(maximum + blocksize[i] * count);
            if (block)
            {
                maximum += blocksize[i] * count;
                free(block);
            }
            else
            {
                break;
            }
        }
    }

    printf("malloc在当前环境下申请到的最大空间是：%.2fGB\n", maximum * 1.0 / 1024 / 1024 / 1024);

    return 0;
}

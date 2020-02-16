#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define MAX 1024
#define N 4

struct Date
{
    int year;
    int month;
    int day;
};

struct Book
{
    char name[40];
    char author[40];
    char publisher[40];
    struct Date date;
};

struct Stu
{
    char name[24];
    int num;
    float score
} stu[N], sb;



int main1(void)
{
    FILE *fp = NULL;
    int ch;

    if ((fp = fopen("test.txt", "r")) == NULL)
    {
        printf("打开文件失败!\n");
        exit(EXIT_FAILURE);
    }
    
    while ((ch = getc(fp)) != EOF)
    {
        putchar(ch);
    }

    fclose(fp);

    return 0;
}

int main2(void)
{
    FILE *fp1;
    FILE *fp2;
    int ch;

    if ((fp1 = fopen("test.txt", "r")) == NULL)
    {
        printf("文件打开失败!\n");
        exit(EXIT_FAILURE);
    }

    if ((fp2 = fopen("hello.txt", "w")) == NULL)
    {
        printf("文件打开失败!\n");
        exit(EXIT_FAILURE);
    }

    while ((ch = fgetc(fp1)) != EOF)
    {
        fputc(ch, fp2);
    }
    
    fclose(fp1);
    fclose(fp2);

    return 0;
}

int main3(void)
{
    FILE *fp;
    char buffer[MAX];

    if ((fp = fopen("lines.txt", "w")) == NULL)
    {
        printf("文件打开失败!\n");
        exit(EXIT_FAILURE);
    }

    fputs("line one: Hello world\n", fp);
    fputs("line one: Hello world\n", fp);
    fputs("line one: Hello world\n", fp);

    fclose(fp);

    if ((fp = fopen("lines.txt", "r")) == NULL)
    {
        printf("文件打开失败!\n");
        exit(EXIT_FAILURE);
    }

    while (!feof(fp))
    {
        fgets(buffer, MAX, fp);
        printf("%s", buffer);
    }

    return 0;
}

int main4(void)
{
    FILE *fp;
    struct tm *p;
    time_t t;

    time(&t);
    p = localtime(&t);

    if ((fp = fopen("date.txt", "w")) == NULL)
    {
        printf("文件打开失败!\n");
        exit(EXIT_FAILURE);
    }

    fprintf(fp, "%d-%d-%d", 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday);
    fclose(fp);

    int year, month, day;

    if ((fp = fopen("date.txt", "r")) == NULL)
    {
        printf("文件打开失败!\n");
        exit(EXIT_FAILURE);
    }

    fscanf(fp, "%d-%d-%d", &year, &month, &day);
    printf("%d-%d-%d", year, month, day);

    fclose(fp);

    return 0;
}

int main5(void)
{
    FILE *fp;

    if ((fp = fopen("date.txt", "wb")) == NULL)
    {
        printf("文件打开失败!\n");
        exit(EXIT_FAILURE);
    }

    fputc('5', fp);
    fputc('2', fp);
    fputc('0', fp);
    fputc('\n', fp);

    fclose(fp);

    return 0;
}

int main6(void)
{
    FILE *fp;
    struct Book *book_for_write, *book_for_read;

    book_for_write = (struct Book *)malloc(sizeof(struct Book));
    book_for_read = (struct Book *)malloc(sizeof(struct Book));
    if (book_for_write == NULL || book_for_read == NULL)
    {
        printf("内存分配失败!\n");
        exit(EXIT_FAILURE);
    }

    strcpy(book_for_write->name, "Cien años de soledad");
    strcpy(book_for_write->author, "Gabriel García Márquez");
    strcpy(book_for_write->publisher, "xxxx");
    book_for_write->date.year = 1967;
    book_for_write->date.month = 1;
    book_for_write->date.day = 1;

    if ((fp = fopen("date.txt", "w")) == NULL)
    {
        printf("文件打开失败!\n");
        exit(EXIT_FAILURE);
    }

    fwrite(book_for_write, sizeof(struct Book), 1, fp);
    fclose(fp);

    if ((fp = fopen("date.txt", "r")) == NULL)
    {
        printf("文件打开失败!\n");
        exit(EXIT_FAILURE);
    }

    fread(book_for_read, sizeof(struct Book), 1, fp);
    printf("书名: %s\n", book_for_read->name);
    printf("作者: %s\n", book_for_read->author);
    printf("出版社: %s\n", book_for_read->publisher);
    printf("出版日期: %d-%d-%d\n", book_for_read->date.year, book_for_read->date.month, book_for_read->date.day);

    fclose(fp);

    return 0;
}

int main7(void)
{
    FILE *fp;

    if ((fp = fopen("date.txt", "w")) == NULL)
    {
        printf("文件打开失败!\n");
        exit(EXIT_FAILURE);
    }

    printf("%ld\n", ftell(fp));
    fputc('F', fp);
    printf("%ld\n", ftell(fp));
    fputs("ishC\n", fp);
    printf("%ld\n", ftell(fp));

    rewind(fp);
    fputs("Hello", fp);

    fclose(fp);

    return 0;
}

int main(void)
{
    FILE *fp;
    int i;

    if ((fp = fopen("date.txt", "w")) == NULL)
    {
        printf("文件打开失败!\n");
        exit(EXIT_FAILURE);
    }

    printf("请开始录入成绩(格式: 姓名 学号 成绩)");
    for (i = 0; i < N; i++)
    {
        scanf("%s %d %f", stu[i].name, stu[i].num, stu[i].score);
    }

    fwrite(stu, sizeof(struct Stu), N, fp);
    fclose(fp);

    if ((fp = fopen("date.txt", "rb")) == NULL)
    {
        printf("文件打开失败!\n");
        exit(EXIT_FAILURE);
    }

    fseek(fp, sizeof(struct Stu), SEEK_SET);
    fread(&sb, sizeof(struct Stu), 1, fp);
    printf("%s(%d)的成绩是: %.2f\n", sb.name, sb.num, sb.score);

    fclose(fp);

    return 0;
}
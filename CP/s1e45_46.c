#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void getInput(struct Book *book);
void addBook(struct Book **library);
void printLibrary(struct Book *library);
struct Book *searchBook(struct Book *library, char *target);
void printBook(struct Book *book);
void releaseLibrary(struct Book **library);

struct Book
{
    char title[128];
    char auther[40];
    struct Book *next;
};

void getInput(struct Book *book)
{
    printf("请输入书名: ");
    scanf("%s", book->title);
    printf("请输入作者: ");
    scanf("%s", book->auther);
}

void addBook(struct Book **library)
{
    struct Book *book, *temp;

    book = (struct Book *)malloc(sizeof(struct Book));
    if (book == NULL)
    {
        printf("内存分配失败了!\n");
        exit(1);
    }

    getInput(book);

    if (*library != NULL)
    {
        // 头插法
        temp = *library;
        *library = book;
        book->next = temp;
    }
    else
    {
        *library = book;
        book->next = NULL;
    }
}

void addBook(struct Book **library)
{
    struct Book *book;
    static struct Book *tail;

    book = (struct Book *)malloc(sizeof(struct Book));
    if (book == NULL)
    {
        printf("内存分配失败了!\n");
        exit(1);
    }

    getInput(book);

    if (*library != NULL)
    {
        // 尾插法
        // temp = *library;
        // while (temp->next != NULL)
        // {
        //     temp = temp->next;
        // }
        // temp->next = book;
        // book->next = NULL;
        // or
        tail->next = book;
        book->next = NULL;
    }
    else
    {
        *library = book;
        book->next = NULL;
    }

    tail = book;
}

void printLibrary(struct Book *library)
{
    struct Book *book;
    int count = 1;

    book = library;
    while (book != NULL)
    {
        printf("Book%d: \n", count);
        printf("书名: %s\n", book->title);
        printf("作者: %s\n", book->auther);
        book = book->next;
        count++;
    }
}

struct Book *searchBook(struct Book *library, char *target)
{
    struct Book *book;

    book = library;
    while (book != NULL)
    {
        if (!strcmp(book->title, target) || !strcmp(book->auther, target))
        {
            break;
        }
        book = book->next;
    }

    return book;
}

void printBook(struct Book *book)
{
    printf("书名: %s\n", book->title);
    printf("作者: %s\n", book->auther);
}

void releaseLibrary(struct Book **library)
{
    struct Book *temp;

    while (*library != NULL)
    {
        temp = *library;
        *library = (*library)->next;
        free(temp);
    }
}

int main(void)
{
    struct Book *library = NULL;
    char ch;
    char input[128];
    struct Book *book;

    while (1)
    {
        printf("请问是否需要录入书籍信息(Y/N): ");
        do
        {
            ch = getchar();
        } while (ch != 'Y' && ch != 'N');

        if (ch == 'Y')
        {
            addBook(&library);
        }
        else
        {
            break;
        }
    }

    printf("请问是否需要打印图书信息(Y/N): ");
    do
    {
        ch = getchar();
    } while (ch != 'Y' && ch != 'N');

    if (ch == 'Y')
    {
        printLibrary(library);
    }

    printf("请输入书名或作者: ");
    scanf("%s", input);

    book = searchBook(library, input);
    if (book == NULL)
    {
        printf("很抱歉, 没能找到!\n");
    }
    else
    {
        do
        {
            printf("已找到符合条件的图书...\n");
            printBook(book);
        } while ((book = searchBook(library, input) != NULL));
    }
    
    releaseLibrary(&library);

    return 0;
}

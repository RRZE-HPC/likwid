.intel_syntax noprefix

.text
.globl _loadData
.type _loadData, @function
_loadData :
#ifdef __x86_64
xor rax, rax
.align 16
1:
mov  r8,  [rsi + rax]
mov  r9,  [rsi + rax + 64]
mov  r10, [rsi + rax + 128]
mov r11,  [rsi + rax + 192]
add rax, 256
cmp rax, rdi
jb 1b

ret
#else
#ifdef __i386__
push	ebp
mov	ebp, esp
push edi
push esi
xor eax, eax
1:
mov edi, DWORD PTR [ebp + eax + 12]
mov esi, DWORD PTR [ebp + eax + 76]
mov ecx, DWORD PTR [ebp + eax + 140]
mov edx, DWORD PTR [ebp + eax + 204]
add eax, 256
cmp eax, DWORD PTR [ebp+8]
jb 1b
pop esi
pop edi
mov esp, ebp
pop ebp
ret
#endif
#endif
.size _loadData, .-_loadData



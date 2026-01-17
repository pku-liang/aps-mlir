
/home/cloud/aps-mlir/examples/diff_match/v3ddist_vv/v3ddist_vv.out:     file format elf32-littleriscv


Disassembly of section .text:

000100b4 <exit>:
   100b4:	ff010113          	addi	sp,sp,-16
   100b8:	00000593          	li	a1,0
   100bc:	00812423          	sw	s0,8(sp)
   100c0:	00112623          	sw	ra,12(sp)
   100c4:	00050413          	mv	s0,a0
   100c8:	721000ef          	jal	10fe8 <__call_exitprocs>
   100cc:	0981a783          	lw	a5,152(gp) # 22918 <__stdio_exit_handler>
   100d0:	00078463          	beqz	a5,100d8 <exit+0x24>
   100d4:	000780e7          	jalr	a5
   100d8:	00040513          	mv	a0,s0
   100dc:	0d5100ef          	jal	209b0 <_exit>

000100e0 <register_fini>:
   100e0:	00000793          	li	a5,0
   100e4:	00078863          	beqz	a5,100f4 <register_fini+0x14>
   100e8:	00007517          	auipc	a0,0x7
   100ec:	aa050513          	addi	a0,a0,-1376 # 16b88 <__libc_fini_array>
   100f0:	0300106f          	j	11120 <atexit>
   100f4:	00008067          	ret

000100f8 <_start>:
   100f8:	00012197          	auipc	gp,0x12
   100fc:	78818193          	addi	gp,gp,1928 # 22880 <__global_pointer$>
   10100:	09818513          	addi	a0,gp,152 # 22918 <__stdio_exit_handler>
   10104:	4b018613          	addi	a2,gp,1200 # 22d30 <__BSS_END__>
   10108:	40a60633          	sub	a2,a2,a0
   1010c:	00000593          	li	a1,0
   10110:	571000ef          	jal	10e80 <memset>
   10114:	00001517          	auipc	a0,0x1
   10118:	00c50513          	addi	a0,a0,12 # 11120 <atexit>
   1011c:	00050863          	beqz	a0,1012c <_start+0x34>
   10120:	00007517          	auipc	a0,0x7
   10124:	a6850513          	addi	a0,a0,-1432 # 16b88 <__libc_fini_array>
   10128:	7f9000ef          	jal	11120 <atexit>
   1012c:	4c1000ef          	jal	10dec <__libc_init_array>
   10130:	00012503          	lw	a0,0(sp)
   10134:	00410593          	addi	a1,sp,4
   10138:	00000613          	li	a2,0
   1013c:	0ac000ef          	jal	101e8 <main>
   10140:	f75ff06f          	j	100b4 <exit>

00010144 <__do_global_dtors_aux>:
   10144:	ff010113          	addi	sp,sp,-16
   10148:	00812423          	sw	s0,8(sp)
   1014c:	10018413          	addi	s0,gp,256 # 22980 <completed.1>
   10150:	00044783          	lbu	a5,0(s0)
   10154:	00112623          	sw	ra,12(sp)
   10158:	02079263          	bnez	a5,1017c <__do_global_dtors_aux+0x38>
   1015c:	00000793          	li	a5,0
   10160:	00078a63          	beqz	a5,10174 <__do_global_dtors_aux+0x30>
   10164:	00012517          	auipc	a0,0x12
   10168:	e9c50513          	addi	a0,a0,-356 # 22000 <__EH_FRAME_BEGIN__>
   1016c:	00000097          	auipc	ra,0x0
   10170:	000000e7          	jalr	zero # 0 <exit-0x100b4>
   10174:	00100793          	li	a5,1
   10178:	00f40023          	sb	a5,0(s0)
   1017c:	00c12083          	lw	ra,12(sp)
   10180:	00812403          	lw	s0,8(sp)
   10184:	01010113          	addi	sp,sp,16
   10188:	00008067          	ret

0001018c <frame_dummy>:
   1018c:	00000793          	li	a5,0
   10190:	00078c63          	beqz	a5,101a8 <frame_dummy+0x1c>
   10194:	10418593          	addi	a1,gp,260 # 22984 <object.0>
   10198:	00012517          	auipc	a0,0x12
   1019c:	e6850513          	addi	a0,a0,-408 # 22000 <__EH_FRAME_BEGIN__>
   101a0:	00000317          	auipc	t1,0x0
   101a4:	00000067          	jr	zero # 0 <exit-0x100b4>
   101a8:	00008067          	ret

000101ac <v3ddist_vv>:
   101ac:	54b5750b          	.insn	4, 0x54b5750b
   101b0:	00000513          	li	a0,0
   101b4:	00008067          	ret

000101b8 <get_march>:
   101b8:	fff50513          	addi	a0,a0,-1
   101bc:	00400593          	li	a1,4
   101c0:	00a5ee63          	bltu	a1,a0,101dc <get_march+0x24>
   101c4:	00251513          	slli	a0,a0,0x2
   101c8:	00011597          	auipc	a1,0x11
   101cc:	95858593          	addi	a1,a1,-1704 # 20b20 <_exit+0x170>
   101d0:	00a58533          	add	a0,a1,a0
   101d4:	00052503          	lw	a0,0(a0)
   101d8:	00008067          	ret
   101dc:	00010517          	auipc	a0,0x10
   101e0:	7fb50513          	addi	a0,a0,2043 # 209d7 <_exit+0x27>
   101e4:	00008067          	ret

000101e8 <main>:
   101e8:	fd010113          	addi	sp,sp,-48
   101ec:	02112623          	sw	ra,44(sp)
   101f0:	02812423          	sw	s0,40(sp)
   101f4:	02912223          	sw	s1,36(sp)
   101f8:	03212023          	sw	s2,32(sp)
   101fc:	01312e23          	sw	s3,28(sp)
   10200:	01412c23          	sw	s4,24(sp)
   10204:	01512a23          	sw	s5,20(sp)
   10208:	00011517          	auipc	a0,0x11
   1020c:	89750513          	addi	a0,a0,-1897 # 20a9f <_exit+0xef>
   10210:	03d000ef          	jal	10a4c <puts>
   10214:	00010517          	auipc	a0,0x10
   10218:	00012597          	auipc	a1,0x12
   1021c:	7cb50513          	addi	a0,a0,1995 # 209df <_exit+0x2f>
   10220:	e6858413          	addi	s0,a1,-408 # 22080 <input_data>
   10224:	00040593          	mv	a1,s0
   10228:	708000ef          	jal	10930 <printf>
   1022c:	00010517          	auipc	a0,0x10
   10230:	7ce50513          	addi	a0,a0,1998 # 209fa <_exit+0x4a>
   10234:	18018493          	addi	s1,gp,384 # 22a00 <output_data>
   10238:	00048593          	mv	a1,s1
   1023c:	6f4000ef          	jal	10930 <printf>
   10240:	f1202573          	.insn	4, 0xf1202573
   10244:	fff50513          	addi	a0,a0,-1
   10248:	00400593          	li	a1,4
   1024c:	00a5ee63          	bltu	a1,a0,10268 <main+0x80>
   10250:	00251513          	slli	a0,a0,0x2
   10254:	00011597          	auipc	a1,0x11
   10258:	8cc58593          	addi	a1,a1,-1844 # 20b20 <_exit+0x170>
   1025c:	00a58533          	add	a0,a1,a0
   10260:	00052583          	lw	a1,0(a0)
   10264:	00c0006f          	j	10270 <main+0x88>
   10268:	00010597          	auipc	a1,0x10
   1026c:	76f58593          	addi	a1,a1,1903 # 209d7 <_exit+0x27>
   10270:	00010517          	auipc	a0,0x10
   10274:	7a650513          	addi	a0,a0,1958 # 20a16 <_exit+0x66>
   10278:	6b8000ef          	jal	10930 <printf>
   1027c:	00012823          	sw	zero,16(sp)
   10280:	00040513          	mv	a0,s0
   10284:	00048593          	mv	a1,s1
   10288:	f25ff0ef          	jal	101ac <v3ddist_vv>
   1028c:	00a12823          	sw	a0,16(sp)
   10290:	00040513          	mv	a0,s0
   10294:	00048593          	mv	a1,s1
   10298:	f15ff0ef          	jal	101ac <v3ddist_vv>
   1029c:	00a12823          	sw	a0,16(sp)
   102a0:	00040513          	mv	a0,s0
   102a4:	00048593          	mv	a1,s1
   102a8:	f05ff0ef          	jal	101ac <v3ddist_vv>
   102ac:	00a12823          	sw	a0,16(sp)
   102b0:	00040513          	mv	a0,s0
   102b4:	00048593          	mv	a1,s1
   102b8:	ef5ff0ef          	jal	101ac <v3ddist_vv>
   102bc:	00a12823          	sw	a0,16(sp)
   102c0:	00040513          	mv	a0,s0
   102c4:	00048593          	mv	a1,s1
   102c8:	ee5ff0ef          	jal	101ac <v3ddist_vv>
   102cc:	00a12823          	sw	a0,16(sp)
   102d0:	00040513          	mv	a0,s0
   102d4:	00048593          	mv	a1,s1
   102d8:	ed5ff0ef          	jal	101ac <v3ddist_vv>
   102dc:	00a12823          	sw	a0,16(sp)
   102e0:	00040513          	mv	a0,s0
   102e4:	00048593          	mv	a1,s1
   102e8:	ec5ff0ef          	jal	101ac <v3ddist_vv>
   102ec:	00a12823          	sw	a0,16(sp)
   102f0:	00040513          	mv	a0,s0
   102f4:	00048593          	mv	a1,s1
   102f8:	eb5ff0ef          	jal	101ac <v3ddist_vv>
   102fc:	00a12823          	sw	a0,16(sp)
   10300:	00040513          	mv	a0,s0
   10304:	00048593          	mv	a1,s1
   10308:	ea5ff0ef          	jal	101ac <v3ddist_vv>
   1030c:	00a12823          	sw	a0,16(sp)
   10310:	00040513          	mv	a0,s0
   10314:	00048593          	mv	a1,s1
   10318:	e95ff0ef          	jal	101ac <v3ddist_vv>
   1031c:	00a12823          	sw	a0,16(sp)
   10320:	01012583          	lw	a1,16(sp)
   10324:	00010517          	auipc	a0,0x10
   10328:	70350513          	addi	a0,a0,1795 # 20a27 <_exit+0x77>
   1032c:	604000ef          	jal	10930 <printf>
   10330:	00010517          	auipc	a0,0x10
   10334:	7a350513          	addi	a0,a0,1955 # 20ad3 <_exit+0x123>
   10338:	714000ef          	jal	10a4c <puts>
   1033c:	00000913          	li	s2,0
   10340:	0c040a13          	addi	s4,s0,192
   10344:	00010997          	auipc	s3,0x10
   10348:	6fa98993          	addi	s3,s3,1786 # 20a3e <_exit+0x8e>
   1034c:	01000a93          	li	s5,16
   10350:	f40a2603          	lw	a2,-192(s4)
   10354:	f80a2683          	lw	a3,-128(s4)
   10358:	fc0a2703          	lw	a4,-64(s4)
   1035c:	000a2783          	lw	a5,0(s4)
   10360:	040a2803          	lw	a6,64(s4)
   10364:	080a2883          	lw	a7,128(s4)
   10368:	0004a503          	lw	a0,0(s1)
   1036c:	00a12023          	sw	a0,0(sp)
   10370:	00098513          	mv	a0,s3
   10374:	00090593          	mv	a1,s2
   10378:	5b8000ef          	jal	10930 <printf>
   1037c:	00190913          	addi	s2,s2,1
   10380:	00448493          	addi	s1,s1,4
   10384:	004a0a13          	addi	s4,s4,4
   10388:	fd5914e3          	bne	s2,s5,10350 <main+0x168>
   1038c:	00010517          	auipc	a0,0x10
   10390:	76d50513          	addi	a0,a0,1901 # 20af9 <_exit+0x149>
   10394:	6b8000ef          	jal	10a4c <puts>
   10398:	00042503          	lw	a0,0(s0)
   1039c:	04042583          	lw	a1,64(s0)
   103a0:	08042603          	lw	a2,128(s0)
   103a4:	0c042683          	lw	a3,192(s0)
   103a8:	10042703          	lw	a4,256(s0)
   103ac:	14042783          	lw	a5,320(s0)
   103b0:	40d50533          	sub	a0,a0,a3
   103b4:	00010897          	auipc	a7,0x10
   103b8:	40e585b3          	sub	a1,a1,a4
   103bc:	00010717          	auipc	a4,0x10
   103c0:	18018a13          	addi	s4,gp,384 # 22a00 <output_data>
   103c4:	40f60633          	sub	a2,a2,a5
   103c8:	000a2683          	lw	a3,0(s4)
   103cc:	02a50533          	mul	a0,a0,a0
   103d0:	02b585b3          	mul	a1,a1,a1
   103d4:	00a58533          	add	a0,a1,a0
   103d8:	000a2583          	lw	a1,0(s4)
   103dc:	02c60633          	mul	a2,a2,a2
   103e0:	00c50633          	add	a2,a0,a2
   103e4:	6db70493          	addi	s1,a4,1755 # 20a97 <_exit+0xe7>
   103e8:	6e788993          	addi	s3,a7,1767 # 20a9b <_exit+0xeb>
   103ec:	00048713          	mv	a4,s1
   103f0:	00b60463          	beq	a2,a1,103f8 <main+0x210>
   103f4:	00098713          	mv	a4,s3
   103f8:	00010917          	auipc	s2,0x10
   103fc:	67d90913          	addi	s2,s2,1661 # 20a75 <_exit+0xc5>
   10400:	00090513          	mv	a0,s2
   10404:	00000593          	li	a1,0
   10408:	528000ef          	jal	10930 <printf>
   1040c:	00442503          	lw	a0,4(s0)
   10410:	04442583          	lw	a1,68(s0)
   10414:	08442603          	lw	a2,132(s0)
   10418:	0c442703          	lw	a4,196(s0)
   1041c:	10442783          	lw	a5,260(s0)
   10420:	14442803          	lw	a6,324(s0)
   10424:	004a2683          	lw	a3,4(s4)
   10428:	40e50533          	sub	a0,a0,a4
   1042c:	40f585b3          	sub	a1,a1,a5
   10430:	004a2783          	lw	a5,4(s4)
   10434:	41060633          	sub	a2,a2,a6
   10438:	02a50533          	mul	a0,a0,a0
   1043c:	02b585b3          	mul	a1,a1,a1
   10440:	00a58533          	add	a0,a1,a0
   10444:	02c60633          	mul	a2,a2,a2
   10448:	00c50633          	add	a2,a0,a2
   1044c:	00048713          	mv	a4,s1
   10450:	00f60463          	beq	a2,a5,10458 <main+0x270>
   10454:	00098713          	mv	a4,s3
   10458:	00100593          	li	a1,1
   1045c:	00090513          	mv	a0,s2
   10460:	4d0000ef          	jal	10930 <printf>
   10464:	00842503          	lw	a0,8(s0)
   10468:	04842583          	lw	a1,72(s0)
   1046c:	08842603          	lw	a2,136(s0)
   10470:	0c842703          	lw	a4,200(s0)
   10474:	10842783          	lw	a5,264(s0)
   10478:	14842803          	lw	a6,328(s0)
   1047c:	008a2683          	lw	a3,8(s4)
   10480:	40e50533          	sub	a0,a0,a4
   10484:	008a2703          	lw	a4,8(s4)
   10488:	40f585b3          	sub	a1,a1,a5
   1048c:	41060633          	sub	a2,a2,a6
   10490:	02a50533          	mul	a0,a0,a0
   10494:	02b585b3          	mul	a1,a1,a1
   10498:	00a58533          	add	a0,a1,a0
   1049c:	02c60633          	mul	a2,a2,a2
   104a0:	00c50633          	add	a2,a0,a2
   104a4:	00e60463          	beq	a2,a4,104ac <main+0x2c4>
   104a8:	00098493          	mv	s1,s3
   104ac:	00200593          	li	a1,2
   104b0:	00090513          	mv	a0,s2
   104b4:	00048713          	mv	a4,s1
   104b8:	478000ef          	jal	10930 <printf>
   104bc:	00000513          	li	a0,0
   104c0:	02c12083          	lw	ra,44(sp)
   104c4:	02812403          	lw	s0,40(sp)
   104c8:	02412483          	lw	s1,36(sp)
   104cc:	02012903          	lw	s2,32(sp)
   104d0:	01c12983          	lw	s3,28(sp)
   104d4:	01812a03          	lw	s4,24(sp)
   104d8:	01412a83          	lw	s5,20(sp)
   104dc:	03010113          	addi	sp,sp,48
   104e0:	00008067          	ret

000104e4 <__fp_lock>:
   104e4:	00000513          	li	a0,0
   104e8:	00008067          	ret

000104ec <stdio_exit_handler>:
   104ec:	98018613          	addi	a2,gp,-1664 # 22200 <__sglue>
   104f0:	00005597          	auipc	a1,0x5
   104f4:	79458593          	addi	a1,a1,1940 # 15c84 <_fclose_r>
   104f8:	00012517          	auipc	a0,0x12
   104fc:	d1850513          	addi	a0,a0,-744 # 22210 <_impure_data>
   10500:	33c0006f          	j	1083c <_fwalk_sglue>

00010504 <cleanup_stdio>:
   10504:	00452583          	lw	a1,4(a0)
   10508:	ff010113          	addi	sp,sp,-16
   1050c:	00812423          	sw	s0,8(sp)
   10510:	00112623          	sw	ra,12(sp)
   10514:	1c018793          	addi	a5,gp,448 # 22a40 <__sf>
   10518:	00050413          	mv	s0,a0
   1051c:	00f58463          	beq	a1,a5,10524 <cleanup_stdio+0x20>
   10520:	764050ef          	jal	15c84 <_fclose_r>
   10524:	00842583          	lw	a1,8(s0)
   10528:	22818793          	addi	a5,gp,552 # 22aa8 <__sf+0x68>
   1052c:	00f58663          	beq	a1,a5,10538 <cleanup_stdio+0x34>
   10530:	00040513          	mv	a0,s0
   10534:	750050ef          	jal	15c84 <_fclose_r>
   10538:	00c42583          	lw	a1,12(s0)
   1053c:	29018793          	addi	a5,gp,656 # 22b10 <__sf+0xd0>
   10540:	00f58c63          	beq	a1,a5,10558 <cleanup_stdio+0x54>
   10544:	00040513          	mv	a0,s0
   10548:	00812403          	lw	s0,8(sp)
   1054c:	00c12083          	lw	ra,12(sp)
   10550:	01010113          	addi	sp,sp,16
   10554:	7300506f          	j	15c84 <_fclose_r>
   10558:	00c12083          	lw	ra,12(sp)
   1055c:	00812403          	lw	s0,8(sp)
   10560:	01010113          	addi	sp,sp,16
   10564:	00008067          	ret

00010568 <__fp_unlock>:
   10568:	00000513          	li	a0,0
   1056c:	00008067          	ret

00010570 <global_stdio_init.part.0>:
   10570:	fe010113          	addi	sp,sp,-32
   10574:	00000797          	auipc	a5,0x0
   10578:	f7878793          	addi	a5,a5,-136 # 104ec <stdio_exit_handler>
   1057c:	00112e23          	sw	ra,28(sp)
   10580:	00812c23          	sw	s0,24(sp)
   10584:	00912a23          	sw	s1,20(sp)
   10588:	1c018413          	addi	s0,gp,448 # 22a40 <__sf>
   1058c:	01212823          	sw	s2,16(sp)
   10590:	01312623          	sw	s3,12(sp)
   10594:	01412423          	sw	s4,8(sp)
   10598:	08f1ac23          	sw	a5,152(gp) # 22918 <__stdio_exit_handler>
   1059c:	00800613          	li	a2,8
   105a0:	00400793          	li	a5,4
   105a4:	00000593          	li	a1,0
   105a8:	21c18513          	addi	a0,gp,540 # 22a9c <__sf+0x5c>
   105ac:	00f42623          	sw	a5,12(s0)
   105b0:	00042023          	sw	zero,0(s0)
   105b4:	00042223          	sw	zero,4(s0)
   105b8:	00042423          	sw	zero,8(s0)
   105bc:	06042223          	sw	zero,100(s0)
   105c0:	00042823          	sw	zero,16(s0)
   105c4:	00042a23          	sw	zero,20(s0)
   105c8:	00042c23          	sw	zero,24(s0)
   105cc:	0b5000ef          	jal	10e80 <memset>
   105d0:	000107b7          	lui	a5,0x10
   105d4:	00000a17          	auipc	s4,0x0
   105d8:	484a0a13          	addi	s4,s4,1156 # 10a58 <__sread>
   105dc:	00000997          	auipc	s3,0x0
   105e0:	4e098993          	addi	s3,s3,1248 # 10abc <__swrite>
   105e4:	00000917          	auipc	s2,0x0
   105e8:	56090913          	addi	s2,s2,1376 # 10b44 <__sseek>
   105ec:	00000497          	auipc	s1,0x0
   105f0:	5d048493          	addi	s1,s1,1488 # 10bbc <__sclose>
   105f4:	00978793          	addi	a5,a5,9 # 10009 <exit-0xab>
   105f8:	00800613          	li	a2,8
   105fc:	00000593          	li	a1,0
   10600:	28418513          	addi	a0,gp,644 # 22b04 <__sf+0xc4>
   10604:	03442023          	sw	s4,32(s0)
   10608:	03342223          	sw	s3,36(s0)
   1060c:	03242423          	sw	s2,40(s0)
   10610:	02942623          	sw	s1,44(s0)
   10614:	06f42a23          	sw	a5,116(s0)
   10618:	00842e23          	sw	s0,28(s0)
   1061c:	06042423          	sw	zero,104(s0)
   10620:	06042623          	sw	zero,108(s0)
   10624:	06042823          	sw	zero,112(s0)
   10628:	0c042623          	sw	zero,204(s0)
   1062c:	06042c23          	sw	zero,120(s0)
   10630:	06042e23          	sw	zero,124(s0)
   10634:	08042023          	sw	zero,128(s0)
   10638:	049000ef          	jal	10e80 <memset>
   1063c:	000207b7          	lui	a5,0x20
   10640:	01278793          	addi	a5,a5,18 # 20012 <__subtf3+0x10d2>
   10644:	22818713          	addi	a4,gp,552 # 22aa8 <__sf+0x68>
   10648:	00800613          	li	a2,8
   1064c:	00000593          	li	a1,0
   10650:	2ec18513          	addi	a0,gp,748 # 22b6c <__sf+0x12c>
   10654:	09442423          	sw	s4,136(s0)
   10658:	09342623          	sw	s3,140(s0)
   1065c:	09242823          	sw	s2,144(s0)
   10660:	08942a23          	sw	s1,148(s0)
   10664:	0cf42e23          	sw	a5,220(s0)
   10668:	08e42223          	sw	a4,132(s0)
   1066c:	0c042823          	sw	zero,208(s0)
   10670:	0c042a23          	sw	zero,212(s0)
   10674:	0c042c23          	sw	zero,216(s0)
   10678:	12042a23          	sw	zero,308(s0)
   1067c:	0e042023          	sw	zero,224(s0)
   10680:	0e042223          	sw	zero,228(s0)
   10684:	0e042423          	sw	zero,232(s0)
   10688:	7f8000ef          	jal	10e80 <memset>
   1068c:	29018793          	addi	a5,gp,656 # 22b10 <__sf+0xd0>
   10690:	0f442823          	sw	s4,240(s0)
   10694:	0f342a23          	sw	s3,244(s0)
   10698:	0f242c23          	sw	s2,248(s0)
   1069c:	0e942e23          	sw	s1,252(s0)
   106a0:	01c12083          	lw	ra,28(sp)
   106a4:	0ef42623          	sw	a5,236(s0)
   106a8:	01812403          	lw	s0,24(sp)
   106ac:	01412483          	lw	s1,20(sp)
   106b0:	01012903          	lw	s2,16(sp)
   106b4:	00c12983          	lw	s3,12(sp)
   106b8:	00812a03          	lw	s4,8(sp)
   106bc:	02010113          	addi	sp,sp,32
   106c0:	00008067          	ret

000106c4 <__sfp>:
   106c4:	fe010113          	addi	sp,sp,-32
   106c8:	01312623          	sw	s3,12(sp)
   106cc:	00112e23          	sw	ra,28(sp)
   106d0:	00812c23          	sw	s0,24(sp)
   106d4:	00912a23          	sw	s1,20(sp)
   106d8:	01212823          	sw	s2,16(sp)
   106dc:	0981a783          	lw	a5,152(gp) # 22918 <__stdio_exit_handler>
   106e0:	00050993          	mv	s3,a0
   106e4:	0e078663          	beqz	a5,107d0 <__sfp+0x10c>
   106e8:	98018913          	addi	s2,gp,-1664 # 22200 <__sglue>
   106ec:	fff00493          	li	s1,-1
   106f0:	00492783          	lw	a5,4(s2)
   106f4:	00892403          	lw	s0,8(s2)
   106f8:	fff78793          	addi	a5,a5,-1
   106fc:	0007d863          	bgez	a5,1070c <__sfp+0x48>
   10700:	0800006f          	j	10780 <__sfp+0xbc>
   10704:	06840413          	addi	s0,s0,104
   10708:	06978c63          	beq	a5,s1,10780 <__sfp+0xbc>
   1070c:	00c41703          	lh	a4,12(s0)
   10710:	fff78793          	addi	a5,a5,-1
   10714:	fe0718e3          	bnez	a4,10704 <__sfp+0x40>
   10718:	ffff07b7          	lui	a5,0xffff0
   1071c:	00178793          	addi	a5,a5,1 # ffff0001 <__BSS_END__+0xfffcd2d1>
   10720:	00f42623          	sw	a5,12(s0)
   10724:	06042223          	sw	zero,100(s0)
   10728:	00042023          	sw	zero,0(s0)
   1072c:	00042423          	sw	zero,8(s0)
   10730:	00042223          	sw	zero,4(s0)
   10734:	00042823          	sw	zero,16(s0)
   10738:	00042a23          	sw	zero,20(s0)
   1073c:	00042c23          	sw	zero,24(s0)
   10740:	00800613          	li	a2,8
   10744:	00000593          	li	a1,0
   10748:	05c40513          	addi	a0,s0,92
   1074c:	734000ef          	jal	10e80 <memset>
   10750:	02042823          	sw	zero,48(s0)
   10754:	02042a23          	sw	zero,52(s0)
   10758:	04042223          	sw	zero,68(s0)
   1075c:	04042423          	sw	zero,72(s0)
   10760:	01c12083          	lw	ra,28(sp)
   10764:	00040513          	mv	a0,s0
   10768:	01812403          	lw	s0,24(sp)
   1076c:	01412483          	lw	s1,20(sp)
   10770:	01012903          	lw	s2,16(sp)
   10774:	00c12983          	lw	s3,12(sp)
   10778:	02010113          	addi	sp,sp,32
   1077c:	00008067          	ret
   10780:	00092403          	lw	s0,0(s2)
   10784:	00040663          	beqz	s0,10790 <__sfp+0xcc>
   10788:	00040913          	mv	s2,s0
   1078c:	f65ff06f          	j	106f0 <__sfp+0x2c>
   10790:	1ac00593          	li	a1,428
   10794:	00098513          	mv	a0,s3
   10798:	5d9000ef          	jal	11570 <_malloc_r>
   1079c:	00050413          	mv	s0,a0
   107a0:	02050c63          	beqz	a0,107d8 <__sfp+0x114>
   107a4:	00c50513          	addi	a0,a0,12
   107a8:	00400793          	li	a5,4
   107ac:	00042023          	sw	zero,0(s0)
   107b0:	00f42223          	sw	a5,4(s0)
   107b4:	00a42423          	sw	a0,8(s0)
   107b8:	1a000613          	li	a2,416
   107bc:	00000593          	li	a1,0
   107c0:	6c0000ef          	jal	10e80 <memset>
   107c4:	00892023          	sw	s0,0(s2)
   107c8:	00040913          	mv	s2,s0
   107cc:	f25ff06f          	j	106f0 <__sfp+0x2c>
   107d0:	da1ff0ef          	jal	10570 <global_stdio_init.part.0>
   107d4:	f15ff06f          	j	106e8 <__sfp+0x24>
   107d8:	00092023          	sw	zero,0(s2)
   107dc:	00c00793          	li	a5,12
   107e0:	00f9a023          	sw	a5,0(s3)
   107e4:	f7dff06f          	j	10760 <__sfp+0x9c>

000107e8 <__sinit>:
   107e8:	03452783          	lw	a5,52(a0)
   107ec:	00078463          	beqz	a5,107f4 <__sinit+0xc>
   107f0:	00008067          	ret
   107f4:	00000797          	auipc	a5,0x0
   107f8:	d1078793          	addi	a5,a5,-752 # 10504 <cleanup_stdio>
   107fc:	02f52a23          	sw	a5,52(a0)
   10800:	0981a783          	lw	a5,152(gp) # 22918 <__stdio_exit_handler>
   10804:	fe0796e3          	bnez	a5,107f0 <__sinit+0x8>
   10808:	d69ff06f          	j	10570 <global_stdio_init.part.0>

0001080c <__sfp_lock_acquire>:
   1080c:	00008067          	ret

00010810 <__sfp_lock_release>:
   10810:	00008067          	ret

00010814 <__fp_lock_all>:
   10814:	98018613          	addi	a2,gp,-1664 # 22200 <__sglue>
   10818:	00000597          	auipc	a1,0x0
   1081c:	ccc58593          	addi	a1,a1,-820 # 104e4 <__fp_lock>
   10820:	00000513          	li	a0,0
   10824:	0180006f          	j	1083c <_fwalk_sglue>

00010828 <__fp_unlock_all>:
   10828:	98018613          	addi	a2,gp,-1664 # 22200 <__sglue>
   1082c:	00000597          	auipc	a1,0x0
   10830:	d3c58593          	addi	a1,a1,-708 # 10568 <__fp_unlock>
   10834:	00000513          	li	a0,0
   10838:	0040006f          	j	1083c <_fwalk_sglue>

0001083c <_fwalk_sglue>:
   1083c:	fd010113          	addi	sp,sp,-48
   10840:	03212023          	sw	s2,32(sp)
   10844:	01312e23          	sw	s3,28(sp)
   10848:	01412c23          	sw	s4,24(sp)
   1084c:	01512a23          	sw	s5,20(sp)
   10850:	01612823          	sw	s6,16(sp)
   10854:	01712623          	sw	s7,12(sp)
   10858:	02112623          	sw	ra,44(sp)
   1085c:	02812423          	sw	s0,40(sp)
   10860:	02912223          	sw	s1,36(sp)
   10864:	00050b13          	mv	s6,a0
   10868:	00058b93          	mv	s7,a1
   1086c:	00060a93          	mv	s5,a2
   10870:	00000a13          	li	s4,0
   10874:	00100993          	li	s3,1
   10878:	fff00913          	li	s2,-1
   1087c:	004aa483          	lw	s1,4(s5)
   10880:	008aa403          	lw	s0,8(s5)
   10884:	fff48493          	addi	s1,s1,-1
   10888:	0204c863          	bltz	s1,108b8 <_fwalk_sglue+0x7c>
   1088c:	00c45783          	lhu	a5,12(s0)
   10890:	00f9fe63          	bgeu	s3,a5,108ac <_fwalk_sglue+0x70>
   10894:	00e41783          	lh	a5,14(s0)
   10898:	00040593          	mv	a1,s0
   1089c:	000b0513          	mv	a0,s6
   108a0:	01278663          	beq	a5,s2,108ac <_fwalk_sglue+0x70>
   108a4:	000b80e7          	jalr	s7
   108a8:	00aa6a33          	or	s4,s4,a0
   108ac:	fff48493          	addi	s1,s1,-1
   108b0:	06840413          	addi	s0,s0,104
   108b4:	fd249ce3          	bne	s1,s2,1088c <_fwalk_sglue+0x50>
   108b8:	000aaa83          	lw	s5,0(s5)
   108bc:	fc0a90e3          	bnez	s5,1087c <_fwalk_sglue+0x40>
   108c0:	02c12083          	lw	ra,44(sp)
   108c4:	02812403          	lw	s0,40(sp)
   108c8:	02412483          	lw	s1,36(sp)
   108cc:	02012903          	lw	s2,32(sp)
   108d0:	01c12983          	lw	s3,28(sp)
   108d4:	01412a83          	lw	s5,20(sp)
   108d8:	01012b03          	lw	s6,16(sp)
   108dc:	00c12b83          	lw	s7,12(sp)
   108e0:	000a0513          	mv	a0,s4
   108e4:	01812a03          	lw	s4,24(sp)
   108e8:	03010113          	addi	sp,sp,48
   108ec:	00008067          	ret

000108f0 <_printf_r>:
   108f0:	fc010113          	addi	sp,sp,-64
   108f4:	02c12423          	sw	a2,40(sp)
   108f8:	02d12623          	sw	a3,44(sp)
   108fc:	02e12823          	sw	a4,48(sp)
   10900:	02f12a23          	sw	a5,52(sp)
   10904:	03012c23          	sw	a6,56(sp)
   10908:	03112e23          	sw	a7,60(sp)
   1090c:	00058613          	mv	a2,a1
   10910:	00852583          	lw	a1,8(a0)
   10914:	02810693          	addi	a3,sp,40
   10918:	00112e23          	sw	ra,28(sp)
   1091c:	00d12623          	sw	a3,12(sp)
   10920:	420010ef          	jal	11d40 <_vfprintf_r>
   10924:	01c12083          	lw	ra,28(sp)
   10928:	04010113          	addi	sp,sp,64
   1092c:	00008067          	ret

00010930 <printf>:
   10930:	fc010113          	addi	sp,sp,-64
   10934:	02c12423          	sw	a2,40(sp)
   10938:	02d12623          	sw	a3,44(sp)
   1093c:	08c1a303          	lw	t1,140(gp) # 2290c <_impure_ptr>
   10940:	02b12223          	sw	a1,36(sp)
   10944:	02e12823          	sw	a4,48(sp)
   10948:	02f12a23          	sw	a5,52(sp)
   1094c:	03012c23          	sw	a6,56(sp)
   10950:	03112e23          	sw	a7,60(sp)
   10954:	00832583          	lw	a1,8(t1) # 101a8 <frame_dummy+0x1c>
   10958:	02410693          	addi	a3,sp,36
   1095c:	00050613          	mv	a2,a0
   10960:	00030513          	mv	a0,t1
   10964:	00112e23          	sw	ra,28(sp)
   10968:	00d12623          	sw	a3,12(sp)
   1096c:	3d4010ef          	jal	11d40 <_vfprintf_r>
   10970:	01c12083          	lw	ra,28(sp)
   10974:	04010113          	addi	sp,sp,64
   10978:	00008067          	ret

0001097c <_puts_r>:
   1097c:	fc010113          	addi	sp,sp,-64
   10980:	02812c23          	sw	s0,56(sp)
   10984:	00050413          	mv	s0,a0
   10988:	00058513          	mv	a0,a1
   1098c:	02912a23          	sw	s1,52(sp)
   10990:	02112e23          	sw	ra,60(sp)
   10994:	00058493          	mv	s1,a1
   10998:	5c4000ef          	jal	10f5c <strlen>
   1099c:	00150713          	addi	a4,a0,1
   109a0:	00010697          	auipc	a3,0x10
   109a4:	19468693          	addi	a3,a3,404 # 20b34 <_exit+0x184>
   109a8:	00e12e23          	sw	a4,28(sp)
   109ac:	03442783          	lw	a5,52(s0)
   109b0:	02010713          	addi	a4,sp,32
   109b4:	02d12423          	sw	a3,40(sp)
   109b8:	00e12a23          	sw	a4,20(sp)
   109bc:	00100693          	li	a3,1
   109c0:	00200713          	li	a4,2
   109c4:	02912023          	sw	s1,32(sp)
   109c8:	02a12223          	sw	a0,36(sp)
   109cc:	02d12623          	sw	a3,44(sp)
   109d0:	00e12c23          	sw	a4,24(sp)
   109d4:	00842583          	lw	a1,8(s0)
   109d8:	06078063          	beqz	a5,10a38 <_puts_r+0xbc>
   109dc:	00c59783          	lh	a5,12(a1)
   109e0:	01279713          	slli	a4,a5,0x12
   109e4:	02074263          	bltz	a4,10a08 <_puts_r+0x8c>
   109e8:	0645a703          	lw	a4,100(a1)
   109ec:	ffffe6b7          	lui	a3,0xffffe
   109f0:	fff68693          	addi	a3,a3,-1 # ffffdfff <__BSS_END__+0xfffdb2cf>
   109f4:	00002637          	lui	a2,0x2
   109f8:	00c7e7b3          	or	a5,a5,a2
   109fc:	00d77733          	and	a4,a4,a3
   10a00:	00f59623          	sh	a5,12(a1)
   10a04:	06e5a223          	sw	a4,100(a1)
   10a08:	01410613          	addi	a2,sp,20
   10a0c:	00040513          	mv	a0,s0
   10a10:	6b4050ef          	jal	160c4 <__sfvwrite_r>
   10a14:	03c12083          	lw	ra,60(sp)
   10a18:	03812403          	lw	s0,56(sp)
   10a1c:	00153513          	seqz	a0,a0
   10a20:	40a00533          	neg	a0,a0
   10a24:	00b57513          	andi	a0,a0,11
   10a28:	03412483          	lw	s1,52(sp)
   10a2c:	fff50513          	addi	a0,a0,-1
   10a30:	04010113          	addi	sp,sp,64
   10a34:	00008067          	ret
   10a38:	00040513          	mv	a0,s0
   10a3c:	00b12623          	sw	a1,12(sp)
   10a40:	da9ff0ef          	jal	107e8 <__sinit>
   10a44:	00c12583          	lw	a1,12(sp)
   10a48:	f95ff06f          	j	109dc <_puts_r+0x60>

00010a4c <puts>:
   10a4c:	00050593          	mv	a1,a0
   10a50:	08c1a503          	lw	a0,140(gp) # 2290c <_impure_ptr>
   10a54:	f29ff06f          	j	1097c <_puts_r>

00010a58 <__sread>:
   10a58:	ff010113          	addi	sp,sp,-16
   10a5c:	00812423          	sw	s0,8(sp)
   10a60:	00058413          	mv	s0,a1
   10a64:	00e59583          	lh	a1,14(a1)
   10a68:	00112623          	sw	ra,12(sp)
   10a6c:	2c8000ef          	jal	10d34 <_read_r>
   10a70:	02054063          	bltz	a0,10a90 <__sread+0x38>
   10a74:	05042783          	lw	a5,80(s0)
   10a78:	00c12083          	lw	ra,12(sp)
   10a7c:	00a787b3          	add	a5,a5,a0
   10a80:	04f42823          	sw	a5,80(s0)
   10a84:	00812403          	lw	s0,8(sp)
   10a88:	01010113          	addi	sp,sp,16
   10a8c:	00008067          	ret
   10a90:	00c45783          	lhu	a5,12(s0)
   10a94:	fffff737          	lui	a4,0xfffff
   10a98:	fff70713          	addi	a4,a4,-1 # ffffefff <__BSS_END__+0xfffdc2cf>
   10a9c:	00e7f7b3          	and	a5,a5,a4
   10aa0:	00c12083          	lw	ra,12(sp)
   10aa4:	00f41623          	sh	a5,12(s0)
   10aa8:	00812403          	lw	s0,8(sp)
   10aac:	01010113          	addi	sp,sp,16
   10ab0:	00008067          	ret

00010ab4 <__seofread>:
   10ab4:	00000513          	li	a0,0
   10ab8:	00008067          	ret

00010abc <__swrite>:
   10abc:	00c59783          	lh	a5,12(a1)
   10ac0:	fe010113          	addi	sp,sp,-32
   10ac4:	00812c23          	sw	s0,24(sp)
   10ac8:	00912a23          	sw	s1,20(sp)
   10acc:	01212823          	sw	s2,16(sp)
   10ad0:	01312623          	sw	s3,12(sp)
   10ad4:	00112e23          	sw	ra,28(sp)
   10ad8:	1007f713          	andi	a4,a5,256
   10adc:	00058413          	mv	s0,a1
   10ae0:	00050493          	mv	s1,a0
   10ae4:	00060913          	mv	s2,a2
   10ae8:	00068993          	mv	s3,a3
   10aec:	04071063          	bnez	a4,10b2c <__swrite+0x70>
   10af0:	fffff737          	lui	a4,0xfffff
   10af4:	fff70713          	addi	a4,a4,-1 # ffffefff <__BSS_END__+0xfffdc2cf>
   10af8:	00e7f7b3          	and	a5,a5,a4
   10afc:	00e41583          	lh	a1,14(s0)
   10b00:	00f41623          	sh	a5,12(s0)
   10b04:	01812403          	lw	s0,24(sp)
   10b08:	01c12083          	lw	ra,28(sp)
   10b0c:	00098693          	mv	a3,s3
   10b10:	00090613          	mv	a2,s2
   10b14:	00c12983          	lw	s3,12(sp)
   10b18:	01012903          	lw	s2,16(sp)
   10b1c:	00048513          	mv	a0,s1
   10b20:	01412483          	lw	s1,20(sp)
   10b24:	02010113          	addi	sp,sp,32
   10b28:	2680006f          	j	10d90 <_write_r>
   10b2c:	00e59583          	lh	a1,14(a1)
   10b30:	00200693          	li	a3,2
   10b34:	00000613          	li	a2,0
   10b38:	1a0000ef          	jal	10cd8 <_lseek_r>
   10b3c:	00c41783          	lh	a5,12(s0)
   10b40:	fb1ff06f          	j	10af0 <__swrite+0x34>

00010b44 <__sseek>:
   10b44:	ff010113          	addi	sp,sp,-16
   10b48:	00812423          	sw	s0,8(sp)
   10b4c:	00058413          	mv	s0,a1
   10b50:	00e59583          	lh	a1,14(a1)
   10b54:	00112623          	sw	ra,12(sp)
   10b58:	180000ef          	jal	10cd8 <_lseek_r>
   10b5c:	fff00793          	li	a5,-1
   10b60:	02f50863          	beq	a0,a5,10b90 <__sseek+0x4c>
   10b64:	00c45783          	lhu	a5,12(s0)
   10b68:	00001737          	lui	a4,0x1
   10b6c:	00c12083          	lw	ra,12(sp)
   10b70:	00e7e7b3          	or	a5,a5,a4
   10b74:	01079793          	slli	a5,a5,0x10
   10b78:	4107d793          	srai	a5,a5,0x10
   10b7c:	04a42823          	sw	a0,80(s0)
   10b80:	00f41623          	sh	a5,12(s0)
   10b84:	00812403          	lw	s0,8(sp)
   10b88:	01010113          	addi	sp,sp,16
   10b8c:	00008067          	ret
   10b90:	00c45783          	lhu	a5,12(s0)
   10b94:	fffff737          	lui	a4,0xfffff
   10b98:	fff70713          	addi	a4,a4,-1 # ffffefff <__BSS_END__+0xfffdc2cf>
   10b9c:	00e7f7b3          	and	a5,a5,a4
   10ba0:	01079793          	slli	a5,a5,0x10
   10ba4:	4107d793          	srai	a5,a5,0x10
   10ba8:	00c12083          	lw	ra,12(sp)
   10bac:	00f41623          	sh	a5,12(s0)
   10bb0:	00812403          	lw	s0,8(sp)
   10bb4:	01010113          	addi	sp,sp,16
   10bb8:	00008067          	ret

00010bbc <__sclose>:
   10bbc:	00e59583          	lh	a1,14(a1)
   10bc0:	0040006f          	j	10bc4 <_close_r>

00010bc4 <_close_r>:
   10bc4:	ff010113          	addi	sp,sp,-16
   10bc8:	00812423          	sw	s0,8(sp)
   10bcc:	00050413          	mv	s0,a0
   10bd0:	00058513          	mv	a0,a1
   10bd4:	0801ae23          	sw	zero,156(gp) # 2291c <errno>
   10bd8:	00112623          	sw	ra,12(sp)
   10bdc:	5250f0ef          	jal	20900 <_close>
   10be0:	fff00793          	li	a5,-1
   10be4:	00f50a63          	beq	a0,a5,10bf8 <_close_r+0x34>
   10be8:	00c12083          	lw	ra,12(sp)
   10bec:	00812403          	lw	s0,8(sp)
   10bf0:	01010113          	addi	sp,sp,16
   10bf4:	00008067          	ret
   10bf8:	09c1a783          	lw	a5,156(gp) # 2291c <errno>
   10bfc:	fe0786e3          	beqz	a5,10be8 <_close_r+0x24>
   10c00:	00c12083          	lw	ra,12(sp)
   10c04:	00f42023          	sw	a5,0(s0)
   10c08:	00812403          	lw	s0,8(sp)
   10c0c:	01010113          	addi	sp,sp,16
   10c10:	00008067          	ret

00010c14 <_reclaim_reent>:
   10c14:	08c1a783          	lw	a5,140(gp) # 2290c <_impure_ptr>
   10c18:	0aa78e63          	beq	a5,a0,10cd4 <_reclaim_reent+0xc0>
   10c1c:	04452583          	lw	a1,68(a0)
   10c20:	fe010113          	addi	sp,sp,-32
   10c24:	00912a23          	sw	s1,20(sp)
   10c28:	00112e23          	sw	ra,28(sp)
   10c2c:	00050493          	mv	s1,a0
   10c30:	04058c63          	beqz	a1,10c88 <_reclaim_reent+0x74>
   10c34:	01212823          	sw	s2,16(sp)
   10c38:	01312623          	sw	s3,12(sp)
   10c3c:	00812c23          	sw	s0,24(sp)
   10c40:	00000913          	li	s2,0
   10c44:	08000993          	li	s3,128
   10c48:	012587b3          	add	a5,a1,s2
   10c4c:	0007a403          	lw	s0,0(a5)
   10c50:	00040e63          	beqz	s0,10c6c <_reclaim_reent+0x58>
   10c54:	00040593          	mv	a1,s0
   10c58:	00042403          	lw	s0,0(s0)
   10c5c:	00048513          	mv	a0,s1
   10c60:	60c000ef          	jal	1126c <_free_r>
   10c64:	fe0418e3          	bnez	s0,10c54 <_reclaim_reent+0x40>
   10c68:	0444a583          	lw	a1,68(s1)
   10c6c:	00490913          	addi	s2,s2,4
   10c70:	fd391ce3          	bne	s2,s3,10c48 <_reclaim_reent+0x34>
   10c74:	00048513          	mv	a0,s1
   10c78:	5f4000ef          	jal	1126c <_free_r>
   10c7c:	01812403          	lw	s0,24(sp)
   10c80:	01012903          	lw	s2,16(sp)
   10c84:	00c12983          	lw	s3,12(sp)
   10c88:	0384a583          	lw	a1,56(s1)
   10c8c:	00058663          	beqz	a1,10c98 <_reclaim_reent+0x84>
   10c90:	00048513          	mv	a0,s1
   10c94:	5d8000ef          	jal	1126c <_free_r>
   10c98:	04c4a583          	lw	a1,76(s1)
   10c9c:	00058663          	beqz	a1,10ca8 <_reclaim_reent+0x94>
   10ca0:	00048513          	mv	a0,s1
   10ca4:	5c8000ef          	jal	1126c <_free_r>
   10ca8:	0344a783          	lw	a5,52(s1)
   10cac:	00078c63          	beqz	a5,10cc4 <_reclaim_reent+0xb0>
   10cb0:	01c12083          	lw	ra,28(sp)
   10cb4:	00048513          	mv	a0,s1
   10cb8:	01412483          	lw	s1,20(sp)
   10cbc:	02010113          	addi	sp,sp,32
   10cc0:	00078067          	jr	a5
   10cc4:	01c12083          	lw	ra,28(sp)
   10cc8:	01412483          	lw	s1,20(sp)
   10ccc:	02010113          	addi	sp,sp,32
   10cd0:	00008067          	ret
   10cd4:	00008067          	ret

00010cd8 <_lseek_r>:
   10cd8:	ff010113          	addi	sp,sp,-16
   10cdc:	00058713          	mv	a4,a1
   10ce0:	00812423          	sw	s0,8(sp)
   10ce4:	00060593          	mv	a1,a2
   10ce8:	00050413          	mv	s0,a0
   10cec:	00068613          	mv	a2,a3
   10cf0:	00070513          	mv	a0,a4
   10cf4:	0801ae23          	sw	zero,156(gp) # 2291c <errno>
   10cf8:	00112623          	sw	ra,12(sp)
   10cfc:	4550f0ef          	jal	20950 <_lseek>
   10d00:	fff00793          	li	a5,-1
   10d04:	00f50a63          	beq	a0,a5,10d18 <_lseek_r+0x40>
   10d08:	00c12083          	lw	ra,12(sp)
   10d0c:	00812403          	lw	s0,8(sp)
   10d10:	01010113          	addi	sp,sp,16
   10d14:	00008067          	ret
   10d18:	09c1a783          	lw	a5,156(gp) # 2291c <errno>
   10d1c:	fe0786e3          	beqz	a5,10d08 <_lseek_r+0x30>
   10d20:	00c12083          	lw	ra,12(sp)
   10d24:	00f42023          	sw	a5,0(s0)
   10d28:	00812403          	lw	s0,8(sp)
   10d2c:	01010113          	addi	sp,sp,16
   10d30:	00008067          	ret

00010d34 <_read_r>:
   10d34:	ff010113          	addi	sp,sp,-16
   10d38:	00058713          	mv	a4,a1
   10d3c:	00812423          	sw	s0,8(sp)
   10d40:	00060593          	mv	a1,a2
   10d44:	00050413          	mv	s0,a0
   10d48:	00068613          	mv	a2,a3
   10d4c:	00070513          	mv	a0,a4
   10d50:	0801ae23          	sw	zero,156(gp) # 2291c <errno>
   10d54:	00112623          	sw	ra,12(sp)
   10d58:	4090f0ef          	jal	20960 <_read>
   10d5c:	fff00793          	li	a5,-1
   10d60:	00f50a63          	beq	a0,a5,10d74 <_read_r+0x40>
   10d64:	00c12083          	lw	ra,12(sp)
   10d68:	00812403          	lw	s0,8(sp)
   10d6c:	01010113          	addi	sp,sp,16
   10d70:	00008067          	ret
   10d74:	09c1a783          	lw	a5,156(gp) # 2291c <errno>
   10d78:	fe0786e3          	beqz	a5,10d64 <_read_r+0x30>
   10d7c:	00c12083          	lw	ra,12(sp)
   10d80:	00f42023          	sw	a5,0(s0)
   10d84:	00812403          	lw	s0,8(sp)
   10d88:	01010113          	addi	sp,sp,16
   10d8c:	00008067          	ret

00010d90 <_write_r>:
   10d90:	ff010113          	addi	sp,sp,-16
   10d94:	00058713          	mv	a4,a1
   10d98:	00812423          	sw	s0,8(sp)
   10d9c:	00060593          	mv	a1,a2
   10da0:	00050413          	mv	s0,a0
   10da4:	00068613          	mv	a2,a3
   10da8:	00070513          	mv	a0,a4
   10dac:	0801ae23          	sw	zero,156(gp) # 2291c <errno>
   10db0:	00112623          	sw	ra,12(sp)
   10db4:	3ed0f0ef          	jal	209a0 <_write>
   10db8:	fff00793          	li	a5,-1
   10dbc:	00f50a63          	beq	a0,a5,10dd0 <_write_r+0x40>
   10dc0:	00c12083          	lw	ra,12(sp)
   10dc4:	00812403          	lw	s0,8(sp)
   10dc8:	01010113          	addi	sp,sp,16
   10dcc:	00008067          	ret
   10dd0:	09c1a783          	lw	a5,156(gp) # 2291c <errno>
   10dd4:	fe0786e3          	beqz	a5,10dc0 <_write_r+0x30>
   10dd8:	00c12083          	lw	ra,12(sp)
   10ddc:	00f42023          	sw	a5,0(s0)
   10de0:	00812403          	lw	s0,8(sp)
   10de4:	01010113          	addi	sp,sp,16
   10de8:	00008067          	ret

00010dec <__libc_init_array>:
   10dec:	ff010113          	addi	sp,sp,-16
   10df0:	00812423          	sw	s0,8(sp)
   10df4:	01212023          	sw	s2,0(sp)
   10df8:	00011797          	auipc	a5,0x11
   10dfc:	27878793          	addi	a5,a5,632 # 22070 <__init_array_start>
   10e00:	00011417          	auipc	s0,0x11
   10e04:	27040413          	addi	s0,s0,624 # 22070 <__init_array_start>
   10e08:	00112623          	sw	ra,12(sp)
   10e0c:	00912223          	sw	s1,4(sp)
   10e10:	40878933          	sub	s2,a5,s0
   10e14:	02878063          	beq	a5,s0,10e34 <__libc_init_array+0x48>
   10e18:	40295913          	srai	s2,s2,0x2
   10e1c:	00000493          	li	s1,0
   10e20:	00042783          	lw	a5,0(s0)
   10e24:	00148493          	addi	s1,s1,1
   10e28:	00440413          	addi	s0,s0,4
   10e2c:	000780e7          	jalr	a5
   10e30:	ff24e8e3          	bltu	s1,s2,10e20 <__libc_init_array+0x34>
   10e34:	00011797          	auipc	a5,0x11
   10e38:	24478793          	addi	a5,a5,580 # 22078 <__do_global_dtors_aux_fini_array_entry>
   10e3c:	00011417          	auipc	s0,0x11
   10e40:	23440413          	addi	s0,s0,564 # 22070 <__init_array_start>
   10e44:	40878933          	sub	s2,a5,s0
   10e48:	40295913          	srai	s2,s2,0x2
   10e4c:	00878e63          	beq	a5,s0,10e68 <__libc_init_array+0x7c>
   10e50:	00000493          	li	s1,0
   10e54:	00042783          	lw	a5,0(s0)
   10e58:	00148493          	addi	s1,s1,1
   10e5c:	00440413          	addi	s0,s0,4
   10e60:	000780e7          	jalr	a5
   10e64:	ff24e8e3          	bltu	s1,s2,10e54 <__libc_init_array+0x68>
   10e68:	00c12083          	lw	ra,12(sp)
   10e6c:	00812403          	lw	s0,8(sp)
   10e70:	00412483          	lw	s1,4(sp)
   10e74:	00012903          	lw	s2,0(sp)
   10e78:	01010113          	addi	sp,sp,16
   10e7c:	00008067          	ret

00010e80 <memset>:
   10e80:	00f00313          	li	t1,15
   10e84:	00050713          	mv	a4,a0
   10e88:	02c37e63          	bgeu	t1,a2,10ec4 <memset+0x44>
   10e8c:	00f77793          	andi	a5,a4,15
   10e90:	0a079063          	bnez	a5,10f30 <memset+0xb0>
   10e94:	08059263          	bnez	a1,10f18 <memset+0x98>
   10e98:	ff067693          	andi	a3,a2,-16
   10e9c:	00f67613          	andi	a2,a2,15
   10ea0:	00e686b3          	add	a3,a3,a4
   10ea4:	00b72023          	sw	a1,0(a4)
   10ea8:	00b72223          	sw	a1,4(a4)
   10eac:	00b72423          	sw	a1,8(a4)
   10eb0:	00b72623          	sw	a1,12(a4)
   10eb4:	01070713          	addi	a4,a4,16
   10eb8:	fed766e3          	bltu	a4,a3,10ea4 <memset+0x24>
   10ebc:	00061463          	bnez	a2,10ec4 <memset+0x44>
   10ec0:	00008067          	ret
   10ec4:	40c306b3          	sub	a3,t1,a2
   10ec8:	00269693          	slli	a3,a3,0x2
   10ecc:	00000297          	auipc	t0,0x0
   10ed0:	005686b3          	add	a3,a3,t0
   10ed4:	00c68067          	jr	12(a3)
   10ed8:	00b70723          	sb	a1,14(a4)
   10edc:	00b706a3          	sb	a1,13(a4)
   10ee0:	00b70623          	sb	a1,12(a4)
   10ee4:	00b705a3          	sb	a1,11(a4)
   10ee8:	00b70523          	sb	a1,10(a4)
   10eec:	00b704a3          	sb	a1,9(a4)
   10ef0:	00b70423          	sb	a1,8(a4)
   10ef4:	00b703a3          	sb	a1,7(a4)
   10ef8:	00b70323          	sb	a1,6(a4)
   10efc:	00b702a3          	sb	a1,5(a4)
   10f00:	00b70223          	sb	a1,4(a4)
   10f04:	00b701a3          	sb	a1,3(a4)
   10f08:	00b70123          	sb	a1,2(a4)
   10f0c:	00b700a3          	sb	a1,1(a4)
   10f10:	00b70023          	sb	a1,0(a4)
   10f14:	00008067          	ret
   10f18:	0ff5f593          	zext.b	a1,a1
   10f1c:	00859693          	slli	a3,a1,0x8
   10f20:	00d5e5b3          	or	a1,a1,a3
   10f24:	01059693          	slli	a3,a1,0x10
   10f28:	00d5e5b3          	or	a1,a1,a3
   10f2c:	f6dff06f          	j	10e98 <memset+0x18>
   10f30:	00279693          	slli	a3,a5,0x2
   10f34:	00000297          	auipc	t0,0x0
   10f38:	005686b3          	add	a3,a3,t0
   10f3c:	00008293          	mv	t0,ra
   10f40:	fa0680e7          	jalr	-96(a3)
   10f44:	00028093          	mv	ra,t0
   10f48:	ff078793          	addi	a5,a5,-16
   10f4c:	40f70733          	sub	a4,a4,a5
   10f50:	00f60633          	add	a2,a2,a5
   10f54:	f6c378e3          	bgeu	t1,a2,10ec4 <memset+0x44>
   10f58:	f3dff06f          	j	10e94 <memset+0x14>

00010f5c <strlen>:
   10f5c:	00357793          	andi	a5,a0,3
   10f60:	00050713          	mv	a4,a0
   10f64:	04079c63          	bnez	a5,10fbc <strlen+0x60>
   10f68:	7f7f86b7          	lui	a3,0x7f7f8
   10f6c:	f7f68693          	addi	a3,a3,-129 # 7f7f7f7f <__BSS_END__+0x7f7d524f>
   10f70:	fff00593          	li	a1,-1
   10f74:	00072603          	lw	a2,0(a4)
   10f78:	00470713          	addi	a4,a4,4
   10f7c:	00d677b3          	and	a5,a2,a3
   10f80:	00d787b3          	add	a5,a5,a3
   10f84:	00c7e7b3          	or	a5,a5,a2
   10f88:	00d7e7b3          	or	a5,a5,a3
   10f8c:	feb784e3          	beq	a5,a1,10f74 <strlen+0x18>
   10f90:	ffc74683          	lbu	a3,-4(a4)
   10f94:	40a707b3          	sub	a5,a4,a0
   10f98:	04068463          	beqz	a3,10fe0 <strlen+0x84>
   10f9c:	ffd74683          	lbu	a3,-3(a4)
   10fa0:	02068c63          	beqz	a3,10fd8 <strlen+0x7c>
   10fa4:	ffe74503          	lbu	a0,-2(a4)
   10fa8:	00a03533          	snez	a0,a0
   10fac:	00f50533          	add	a0,a0,a5
   10fb0:	ffe50513          	addi	a0,a0,-2
   10fb4:	00008067          	ret
   10fb8:	fa0688e3          	beqz	a3,10f68 <strlen+0xc>
   10fbc:	00074783          	lbu	a5,0(a4)
   10fc0:	00170713          	addi	a4,a4,1
   10fc4:	00377693          	andi	a3,a4,3
   10fc8:	fe0798e3          	bnez	a5,10fb8 <strlen+0x5c>
   10fcc:	40a70733          	sub	a4,a4,a0
   10fd0:	fff70513          	addi	a0,a4,-1
   10fd4:	00008067          	ret
   10fd8:	ffd78513          	addi	a0,a5,-3
   10fdc:	00008067          	ret
   10fe0:	ffc78513          	addi	a0,a5,-4
   10fe4:	00008067          	ret

00010fe8 <__call_exitprocs>:
   10fe8:	fd010113          	addi	sp,sp,-48
   10fec:	01412c23          	sw	s4,24(sp)
   10ff0:	0a018a13          	addi	s4,gp,160 # 22920 <__atexit>
   10ff4:	03212023          	sw	s2,32(sp)
   10ff8:	000a2903          	lw	s2,0(s4)
   10ffc:	02112623          	sw	ra,44(sp)
   11000:	0a090863          	beqz	s2,110b0 <__call_exitprocs+0xc8>
   11004:	01312e23          	sw	s3,28(sp)
   11008:	01512a23          	sw	s5,20(sp)
   1100c:	01612823          	sw	s6,16(sp)
   11010:	01712623          	sw	s7,12(sp)
   11014:	02812423          	sw	s0,40(sp)
   11018:	02912223          	sw	s1,36(sp)
   1101c:	01812423          	sw	s8,8(sp)
   11020:	00050b13          	mv	s6,a0
   11024:	00058b93          	mv	s7,a1
   11028:	fff00993          	li	s3,-1
   1102c:	00100a93          	li	s5,1
   11030:	00492483          	lw	s1,4(s2)
   11034:	fff48413          	addi	s0,s1,-1
   11038:	04044e63          	bltz	s0,11094 <__call_exitprocs+0xac>
   1103c:	00249493          	slli	s1,s1,0x2
   11040:	009904b3          	add	s1,s2,s1
   11044:	080b9063          	bnez	s7,110c4 <__call_exitprocs+0xdc>
   11048:	00492783          	lw	a5,4(s2)
   1104c:	0044a683          	lw	a3,4(s1)
   11050:	fff78793          	addi	a5,a5,-1
   11054:	0a878c63          	beq	a5,s0,1110c <__call_exitprocs+0x124>
   11058:	0004a223          	sw	zero,4(s1)
   1105c:	02068663          	beqz	a3,11088 <__call_exitprocs+0xa0>
   11060:	18892783          	lw	a5,392(s2)
   11064:	008a9733          	sll	a4,s5,s0
   11068:	00492c03          	lw	s8,4(s2)
   1106c:	00f777b3          	and	a5,a4,a5
   11070:	06079663          	bnez	a5,110dc <__call_exitprocs+0xf4>
   11074:	000680e7          	jalr	a3
   11078:	00492703          	lw	a4,4(s2)
   1107c:	000a2783          	lw	a5,0(s4)
   11080:	09871063          	bne	a4,s8,11100 <__call_exitprocs+0x118>
   11084:	07279e63          	bne	a5,s2,11100 <__call_exitprocs+0x118>
   11088:	fff40413          	addi	s0,s0,-1
   1108c:	ffc48493          	addi	s1,s1,-4
   11090:	fb341ae3          	bne	s0,s3,11044 <__call_exitprocs+0x5c>
   11094:	02812403          	lw	s0,40(sp)
   11098:	02412483          	lw	s1,36(sp)
   1109c:	01c12983          	lw	s3,28(sp)
   110a0:	01412a83          	lw	s5,20(sp)
   110a4:	01012b03          	lw	s6,16(sp)
   110a8:	00c12b83          	lw	s7,12(sp)
   110ac:	00812c03          	lw	s8,8(sp)
   110b0:	02c12083          	lw	ra,44(sp)
   110b4:	02012903          	lw	s2,32(sp)
   110b8:	01812a03          	lw	s4,24(sp)
   110bc:	03010113          	addi	sp,sp,48
   110c0:	00008067          	ret
   110c4:	1044a783          	lw	a5,260(s1)
   110c8:	f97780e3          	beq	a5,s7,11048 <__call_exitprocs+0x60>
   110cc:	fff40413          	addi	s0,s0,-1
   110d0:	ffc48493          	addi	s1,s1,-4
   110d4:	ff3418e3          	bne	s0,s3,110c4 <__call_exitprocs+0xdc>
   110d8:	fbdff06f          	j	11094 <__call_exitprocs+0xac>
   110dc:	18c92783          	lw	a5,396(s2)
   110e0:	0844a583          	lw	a1,132(s1)
   110e4:	00f77733          	and	a4,a4,a5
   110e8:	02071663          	bnez	a4,11114 <__call_exitprocs+0x12c>
   110ec:	000b0513          	mv	a0,s6
   110f0:	000680e7          	jalr	a3
   110f4:	00492703          	lw	a4,4(s2)
   110f8:	000a2783          	lw	a5,0(s4)
   110fc:	f98704e3          	beq	a4,s8,11084 <__call_exitprocs+0x9c>
   11100:	f8078ae3          	beqz	a5,11094 <__call_exitprocs+0xac>
   11104:	00078913          	mv	s2,a5
   11108:	f29ff06f          	j	11030 <__call_exitprocs+0x48>
   1110c:	00892223          	sw	s0,4(s2)
   11110:	f4dff06f          	j	1105c <__call_exitprocs+0x74>
   11114:	00058513          	mv	a0,a1
   11118:	000680e7          	jalr	a3
   1111c:	f5dff06f          	j	11078 <__call_exitprocs+0x90>

00011120 <atexit>:
   11120:	00050593          	mv	a1,a0
   11124:	00000693          	li	a3,0
   11128:	00000613          	li	a2,0
   1112c:	00000513          	li	a0,0
   11130:	0780606f          	j	171a8 <__register_exitproc>

00011134 <_malloc_trim_r>:
   11134:	fe010113          	addi	sp,sp,-32
   11138:	00812c23          	sw	s0,24(sp)
   1113c:	00912a23          	sw	s1,20(sp)
   11140:	01212823          	sw	s2,16(sp)
   11144:	01312623          	sw	s3,12(sp)
   11148:	01412423          	sw	s4,8(sp)
   1114c:	00058993          	mv	s3,a1
   11150:	00112e23          	sw	ra,28(sp)
   11154:	00050913          	mv	s2,a0
   11158:	00011a17          	auipc	s4,0x11
   1115c:	1d8a0a13          	addi	s4,s4,472 # 22330 <__malloc_av_>
   11160:	3d9000ef          	jal	11d38 <__malloc_lock>
   11164:	008a2703          	lw	a4,8(s4)
   11168:	000017b7          	lui	a5,0x1
   1116c:	fef78793          	addi	a5,a5,-17 # fef <exit-0xf0c5>
   11170:	00472483          	lw	s1,4(a4)
   11174:	00001737          	lui	a4,0x1
   11178:	ffc4f493          	andi	s1,s1,-4
   1117c:	00f48433          	add	s0,s1,a5
   11180:	41340433          	sub	s0,s0,s3
   11184:	00c45413          	srli	s0,s0,0xc
   11188:	fff40413          	addi	s0,s0,-1
   1118c:	00c41413          	slli	s0,s0,0xc
   11190:	00e44e63          	blt	s0,a4,111ac <_malloc_trim_r+0x78>
   11194:	00000593          	li	a1,0
   11198:	00090513          	mv	a0,s2
   1119c:	19d050ef          	jal	16b38 <_sbrk_r>
   111a0:	008a2783          	lw	a5,8(s4)
   111a4:	009787b3          	add	a5,a5,s1
   111a8:	02f50863          	beq	a0,a5,111d8 <_malloc_trim_r+0xa4>
   111ac:	00090513          	mv	a0,s2
   111b0:	38d000ef          	jal	11d3c <__malloc_unlock>
   111b4:	01c12083          	lw	ra,28(sp)
   111b8:	01812403          	lw	s0,24(sp)
   111bc:	01412483          	lw	s1,20(sp)
   111c0:	01012903          	lw	s2,16(sp)
   111c4:	00c12983          	lw	s3,12(sp)
   111c8:	00812a03          	lw	s4,8(sp)
   111cc:	00000513          	li	a0,0
   111d0:	02010113          	addi	sp,sp,32
   111d4:	00008067          	ret
   111d8:	408005b3          	neg	a1,s0
   111dc:	00090513          	mv	a0,s2
   111e0:	159050ef          	jal	16b38 <_sbrk_r>
   111e4:	fff00793          	li	a5,-1
   111e8:	04f50863          	beq	a0,a5,11238 <_malloc_trim_r+0x104>
   111ec:	2f818713          	addi	a4,gp,760 # 22b78 <__malloc_current_mallinfo>
   111f0:	00072783          	lw	a5,0(a4) # 1000 <exit-0xf0b4>
   111f4:	008a2683          	lw	a3,8(s4)
   111f8:	408484b3          	sub	s1,s1,s0
   111fc:	0014e493          	ori	s1,s1,1
   11200:	408787b3          	sub	a5,a5,s0
   11204:	00090513          	mv	a0,s2
   11208:	0096a223          	sw	s1,4(a3)
   1120c:	00f72023          	sw	a5,0(a4)
   11210:	32d000ef          	jal	11d3c <__malloc_unlock>
   11214:	01c12083          	lw	ra,28(sp)
   11218:	01812403          	lw	s0,24(sp)
   1121c:	01412483          	lw	s1,20(sp)
   11220:	01012903          	lw	s2,16(sp)
   11224:	00c12983          	lw	s3,12(sp)
   11228:	00812a03          	lw	s4,8(sp)
   1122c:	00100513          	li	a0,1
   11230:	02010113          	addi	sp,sp,32
   11234:	00008067          	ret
   11238:	00000593          	li	a1,0
   1123c:	00090513          	mv	a0,s2
   11240:	0f9050ef          	jal	16b38 <_sbrk_r>
   11244:	008a2703          	lw	a4,8(s4)
   11248:	00f00693          	li	a3,15
   1124c:	40e507b3          	sub	a5,a0,a4
   11250:	f4f6dee3          	bge	a3,a5,111ac <_malloc_trim_r+0x78>
   11254:	0901a683          	lw	a3,144(gp) # 22910 <__malloc_sbrk_base>
   11258:	40d50533          	sub	a0,a0,a3
   1125c:	0017e793          	ori	a5,a5,1
   11260:	2ea1ac23          	sw	a0,760(gp) # 22b78 <__malloc_current_mallinfo>
   11264:	00f72223          	sw	a5,4(a4)
   11268:	f45ff06f          	j	111ac <_malloc_trim_r+0x78>

0001126c <_free_r>:
   1126c:	18058263          	beqz	a1,113f0 <_free_r+0x184>
   11270:	ff010113          	addi	sp,sp,-16
   11274:	00812423          	sw	s0,8(sp)
   11278:	00912223          	sw	s1,4(sp)
   1127c:	00058413          	mv	s0,a1
   11280:	00050493          	mv	s1,a0
   11284:	00112623          	sw	ra,12(sp)
   11288:	2b1000ef          	jal	11d38 <__malloc_lock>
   1128c:	ffc42583          	lw	a1,-4(s0)
   11290:	ff840713          	addi	a4,s0,-8
   11294:	00011517          	auipc	a0,0x11
   11298:	09c50513          	addi	a0,a0,156 # 22330 <__malloc_av_>
   1129c:	ffe5f793          	andi	a5,a1,-2
   112a0:	00f70633          	add	a2,a4,a5
   112a4:	00462683          	lw	a3,4(a2) # 2004 <exit-0xe0b0>
   112a8:	00852803          	lw	a6,8(a0)
   112ac:	ffc6f693          	andi	a3,a3,-4
   112b0:	1ac80263          	beq	a6,a2,11454 <_free_r+0x1e8>
   112b4:	00d62223          	sw	a3,4(a2)
   112b8:	0015f593          	andi	a1,a1,1
   112bc:	00d60833          	add	a6,a2,a3
   112c0:	0a059063          	bnez	a1,11360 <_free_r+0xf4>
   112c4:	ff842303          	lw	t1,-8(s0)
   112c8:	00482583          	lw	a1,4(a6)
   112cc:	00011897          	auipc	a7,0x11
   112d0:	06c88893          	addi	a7,a7,108 # 22338 <__malloc_av_+0x8>
   112d4:	40670733          	sub	a4,a4,t1
   112d8:	00872803          	lw	a6,8(a4)
   112dc:	006787b3          	add	a5,a5,t1
   112e0:	0015f593          	andi	a1,a1,1
   112e4:	15180263          	beq	a6,a7,11428 <_free_r+0x1bc>
   112e8:	00c72303          	lw	t1,12(a4)
   112ec:	00682623          	sw	t1,12(a6)
   112f0:	01032423          	sw	a6,8(t1)
   112f4:	1a058663          	beqz	a1,114a0 <_free_r+0x234>
   112f8:	0017e693          	ori	a3,a5,1
   112fc:	00d72223          	sw	a3,4(a4)
   11300:	00f62023          	sw	a5,0(a2)
   11304:	1ff00693          	li	a3,511
   11308:	06f6ec63          	bltu	a3,a5,11380 <_free_r+0x114>
   1130c:	ff87f693          	andi	a3,a5,-8
   11310:	00868693          	addi	a3,a3,8
   11314:	00452583          	lw	a1,4(a0)
   11318:	00d506b3          	add	a3,a0,a3
   1131c:	0006a603          	lw	a2,0(a3)
   11320:	0057d813          	srli	a6,a5,0x5
   11324:	00100793          	li	a5,1
   11328:	010797b3          	sll	a5,a5,a6
   1132c:	00b7e7b3          	or	a5,a5,a1
   11330:	ff868593          	addi	a1,a3,-8
   11334:	00b72623          	sw	a1,12(a4)
   11338:	00c72423          	sw	a2,8(a4)
   1133c:	00f52223          	sw	a5,4(a0)
   11340:	00e6a023          	sw	a4,0(a3)
   11344:	00e62623          	sw	a4,12(a2)
   11348:	00812403          	lw	s0,8(sp)
   1134c:	00c12083          	lw	ra,12(sp)
   11350:	00048513          	mv	a0,s1
   11354:	00412483          	lw	s1,4(sp)
   11358:	01010113          	addi	sp,sp,16
   1135c:	1e10006f          	j	11d3c <__malloc_unlock>
   11360:	00482583          	lw	a1,4(a6)
   11364:	0015f593          	andi	a1,a1,1
   11368:	08058663          	beqz	a1,113f4 <_free_r+0x188>
   1136c:	0017e693          	ori	a3,a5,1
   11370:	fed42e23          	sw	a3,-4(s0)
   11374:	00f62023          	sw	a5,0(a2)
   11378:	1ff00693          	li	a3,511
   1137c:	f8f6f8e3          	bgeu	a3,a5,1130c <_free_r+0xa0>
   11380:	0097d693          	srli	a3,a5,0x9
   11384:	00400613          	li	a2,4
   11388:	12d66063          	bltu	a2,a3,114a8 <_free_r+0x23c>
   1138c:	0067d693          	srli	a3,a5,0x6
   11390:	03968593          	addi	a1,a3,57
   11394:	03868613          	addi	a2,a3,56
   11398:	00359593          	slli	a1,a1,0x3
   1139c:	00b505b3          	add	a1,a0,a1
   113a0:	0005a683          	lw	a3,0(a1)
   113a4:	ff858593          	addi	a1,a1,-8
   113a8:	00d59863          	bne	a1,a3,113b8 <_free_r+0x14c>
   113ac:	1540006f          	j	11500 <_free_r+0x294>
   113b0:	0086a683          	lw	a3,8(a3)
   113b4:	00d58863          	beq	a1,a3,113c4 <_free_r+0x158>
   113b8:	0046a603          	lw	a2,4(a3)
   113bc:	ffc67613          	andi	a2,a2,-4
   113c0:	fec7e8e3          	bltu	a5,a2,113b0 <_free_r+0x144>
   113c4:	00c6a583          	lw	a1,12(a3)
   113c8:	00b72623          	sw	a1,12(a4)
   113cc:	00d72423          	sw	a3,8(a4)
   113d0:	00812403          	lw	s0,8(sp)
   113d4:	00c12083          	lw	ra,12(sp)
   113d8:	00e5a423          	sw	a4,8(a1)
   113dc:	00048513          	mv	a0,s1
   113e0:	00412483          	lw	s1,4(sp)
   113e4:	00e6a623          	sw	a4,12(a3)
   113e8:	01010113          	addi	sp,sp,16
   113ec:	1510006f          	j	11d3c <__malloc_unlock>
   113f0:	00008067          	ret
   113f4:	00d787b3          	add	a5,a5,a3
   113f8:	00011897          	auipc	a7,0x11
   113fc:	f4088893          	addi	a7,a7,-192 # 22338 <__malloc_av_+0x8>
   11400:	00862683          	lw	a3,8(a2)
   11404:	0d168c63          	beq	a3,a7,114dc <_free_r+0x270>
   11408:	00c62803          	lw	a6,12(a2)
   1140c:	0017e593          	ori	a1,a5,1
   11410:	00f70633          	add	a2,a4,a5
   11414:	0106a623          	sw	a6,12(a3)
   11418:	00d82423          	sw	a3,8(a6)
   1141c:	00b72223          	sw	a1,4(a4)
   11420:	00f62023          	sw	a5,0(a2)
   11424:	ee1ff06f          	j	11304 <_free_r+0x98>
   11428:	12059c63          	bnez	a1,11560 <_free_r+0x2f4>
   1142c:	00862583          	lw	a1,8(a2)
   11430:	00c62603          	lw	a2,12(a2)
   11434:	00f686b3          	add	a3,a3,a5
   11438:	0016e793          	ori	a5,a3,1
   1143c:	00c5a623          	sw	a2,12(a1)
   11440:	00b62423          	sw	a1,8(a2)
   11444:	00f72223          	sw	a5,4(a4)
   11448:	00d70733          	add	a4,a4,a3
   1144c:	00d72023          	sw	a3,0(a4)
   11450:	ef9ff06f          	j	11348 <_free_r+0xdc>
   11454:	0015f593          	andi	a1,a1,1
   11458:	00d786b3          	add	a3,a5,a3
   1145c:	02059063          	bnez	a1,1147c <_free_r+0x210>
   11460:	ff842583          	lw	a1,-8(s0)
   11464:	40b70733          	sub	a4,a4,a1
   11468:	00c72783          	lw	a5,12(a4)
   1146c:	00872603          	lw	a2,8(a4)
   11470:	00b686b3          	add	a3,a3,a1
   11474:	00f62623          	sw	a5,12(a2)
   11478:	00c7a423          	sw	a2,8(a5)
   1147c:	0016e793          	ori	a5,a3,1
   11480:	00f72223          	sw	a5,4(a4)
   11484:	00e52423          	sw	a4,8(a0)
   11488:	0941a783          	lw	a5,148(gp) # 22914 <__malloc_trim_threshold>
   1148c:	eaf6eee3          	bltu	a3,a5,11348 <_free_r+0xdc>
   11490:	0ac1a583          	lw	a1,172(gp) # 2292c <__malloc_top_pad>
   11494:	00048513          	mv	a0,s1
   11498:	c9dff0ef          	jal	11134 <_malloc_trim_r>
   1149c:	eadff06f          	j	11348 <_free_r+0xdc>
   114a0:	00d787b3          	add	a5,a5,a3
   114a4:	f5dff06f          	j	11400 <_free_r+0x194>
   114a8:	01400613          	li	a2,20
   114ac:	02d67063          	bgeu	a2,a3,114cc <_free_r+0x260>
   114b0:	05400613          	li	a2,84
   114b4:	06d66463          	bltu	a2,a3,1151c <_free_r+0x2b0>
   114b8:	00c7d693          	srli	a3,a5,0xc
   114bc:	06f68593          	addi	a1,a3,111
   114c0:	06e68613          	addi	a2,a3,110
   114c4:	00359593          	slli	a1,a1,0x3
   114c8:	ed5ff06f          	j	1139c <_free_r+0x130>
   114cc:	05c68593          	addi	a1,a3,92
   114d0:	05b68613          	addi	a2,a3,91
   114d4:	00359593          	slli	a1,a1,0x3
   114d8:	ec5ff06f          	j	1139c <_free_r+0x130>
   114dc:	00e52a23          	sw	a4,20(a0)
   114e0:	00e52823          	sw	a4,16(a0)
   114e4:	0017e693          	ori	a3,a5,1
   114e8:	01172623          	sw	a7,12(a4)
   114ec:	01172423          	sw	a7,8(a4)
   114f0:	00d72223          	sw	a3,4(a4)
   114f4:	00f70733          	add	a4,a4,a5
   114f8:	00f72023          	sw	a5,0(a4)
   114fc:	e4dff06f          	j	11348 <_free_r+0xdc>
   11500:	00452803          	lw	a6,4(a0)
   11504:	40265613          	srai	a2,a2,0x2
   11508:	00100793          	li	a5,1
   1150c:	00c797b3          	sll	a5,a5,a2
   11510:	0107e7b3          	or	a5,a5,a6
   11514:	00f52223          	sw	a5,4(a0)
   11518:	eb1ff06f          	j	113c8 <_free_r+0x15c>
   1151c:	15400613          	li	a2,340
   11520:	00d66c63          	bltu	a2,a3,11538 <_free_r+0x2cc>
   11524:	00f7d693          	srli	a3,a5,0xf
   11528:	07868593          	addi	a1,a3,120
   1152c:	07768613          	addi	a2,a3,119
   11530:	00359593          	slli	a1,a1,0x3
   11534:	e69ff06f          	j	1139c <_free_r+0x130>
   11538:	55400613          	li	a2,1364
   1153c:	00d66c63          	bltu	a2,a3,11554 <_free_r+0x2e8>
   11540:	0127d693          	srli	a3,a5,0x12
   11544:	07d68593          	addi	a1,a3,125
   11548:	07c68613          	addi	a2,a3,124
   1154c:	00359593          	slli	a1,a1,0x3
   11550:	e4dff06f          	j	1139c <_free_r+0x130>
   11554:	3f800593          	li	a1,1016
   11558:	07e00613          	li	a2,126
   1155c:	e41ff06f          	j	1139c <_free_r+0x130>
   11560:	0017e693          	ori	a3,a5,1
   11564:	00d72223          	sw	a3,4(a4)
   11568:	00f62023          	sw	a5,0(a2)
   1156c:	dddff06f          	j	11348 <_free_r+0xdc>

00011570 <_malloc_r>:
   11570:	fd010113          	addi	sp,sp,-48
   11574:	03212023          	sw	s2,32(sp)
   11578:	02112623          	sw	ra,44(sp)
   1157c:	02812423          	sw	s0,40(sp)
   11580:	02912223          	sw	s1,36(sp)
   11584:	01312e23          	sw	s3,28(sp)
   11588:	00b58793          	addi	a5,a1,11
   1158c:	01600713          	li	a4,22
   11590:	00050913          	mv	s2,a0
   11594:	08f76263          	bltu	a4,a5,11618 <_malloc_r+0xa8>
   11598:	01000793          	li	a5,16
   1159c:	20b7e663          	bltu	a5,a1,117a8 <_malloc_r+0x238>
   115a0:	798000ef          	jal	11d38 <__malloc_lock>
   115a4:	01800793          	li	a5,24
   115a8:	00200593          	li	a1,2
   115ac:	01000493          	li	s1,16
   115b0:	00011997          	auipc	s3,0x11
   115b4:	d8098993          	addi	s3,s3,-640 # 22330 <__malloc_av_>
   115b8:	00f987b3          	add	a5,s3,a5
   115bc:	0047a403          	lw	s0,4(a5)
   115c0:	ff878713          	addi	a4,a5,-8
   115c4:	34e40a63          	beq	s0,a4,11918 <_malloc_r+0x3a8>
   115c8:	00442783          	lw	a5,4(s0)
   115cc:	00c42683          	lw	a3,12(s0)
   115d0:	00842603          	lw	a2,8(s0)
   115d4:	ffc7f793          	andi	a5,a5,-4
   115d8:	00f407b3          	add	a5,s0,a5
   115dc:	0047a703          	lw	a4,4(a5)
   115e0:	00d62623          	sw	a3,12(a2)
   115e4:	00c6a423          	sw	a2,8(a3)
   115e8:	00176713          	ori	a4,a4,1
   115ec:	00090513          	mv	a0,s2
   115f0:	00e7a223          	sw	a4,4(a5)
   115f4:	748000ef          	jal	11d3c <__malloc_unlock>
   115f8:	00840513          	addi	a0,s0,8
   115fc:	02c12083          	lw	ra,44(sp)
   11600:	02812403          	lw	s0,40(sp)
   11604:	02412483          	lw	s1,36(sp)
   11608:	02012903          	lw	s2,32(sp)
   1160c:	01c12983          	lw	s3,28(sp)
   11610:	03010113          	addi	sp,sp,48
   11614:	00008067          	ret
   11618:	ff87f493          	andi	s1,a5,-8
   1161c:	1807c663          	bltz	a5,117a8 <_malloc_r+0x238>
   11620:	18b4e463          	bltu	s1,a1,117a8 <_malloc_r+0x238>
   11624:	714000ef          	jal	11d38 <__malloc_lock>
   11628:	1f700793          	li	a5,503
   1162c:	4097f063          	bgeu	a5,s1,11a2c <_malloc_r+0x4bc>
   11630:	0094d793          	srli	a5,s1,0x9
   11634:	18078263          	beqz	a5,117b8 <_malloc_r+0x248>
   11638:	00400713          	li	a4,4
   1163c:	34f76663          	bltu	a4,a5,11988 <_malloc_r+0x418>
   11640:	0064d793          	srli	a5,s1,0x6
   11644:	03978593          	addi	a1,a5,57
   11648:	03878813          	addi	a6,a5,56
   1164c:	00359613          	slli	a2,a1,0x3
   11650:	00011997          	auipc	s3,0x11
   11654:	ce098993          	addi	s3,s3,-800 # 22330 <__malloc_av_>
   11658:	00c98633          	add	a2,s3,a2
   1165c:	00462403          	lw	s0,4(a2)
   11660:	ff860613          	addi	a2,a2,-8
   11664:	02860863          	beq	a2,s0,11694 <_malloc_r+0x124>
   11668:	00f00513          	li	a0,15
   1166c:	0140006f          	j	11680 <_malloc_r+0x110>
   11670:	00c42683          	lw	a3,12(s0)
   11674:	28075e63          	bgez	a4,11910 <_malloc_r+0x3a0>
   11678:	00d60e63          	beq	a2,a3,11694 <_malloc_r+0x124>
   1167c:	00068413          	mv	s0,a3
   11680:	00442783          	lw	a5,4(s0)
   11684:	ffc7f793          	andi	a5,a5,-4
   11688:	40978733          	sub	a4,a5,s1
   1168c:	fee552e3          	bge	a0,a4,11670 <_malloc_r+0x100>
   11690:	00080593          	mv	a1,a6
   11694:	0109a403          	lw	s0,16(s3)
   11698:	00011897          	auipc	a7,0x11
   1169c:	ca088893          	addi	a7,a7,-864 # 22338 <__malloc_av_+0x8>
   116a0:	27140463          	beq	s0,a7,11908 <_malloc_r+0x398>
   116a4:	00442783          	lw	a5,4(s0)
   116a8:	00f00693          	li	a3,15
   116ac:	ffc7f793          	andi	a5,a5,-4
   116b0:	40978733          	sub	a4,a5,s1
   116b4:	38e6c263          	blt	a3,a4,11a38 <_malloc_r+0x4c8>
   116b8:	0119aa23          	sw	a7,20(s3)
   116bc:	0119a823          	sw	a7,16(s3)
   116c0:	34075663          	bgez	a4,11a0c <_malloc_r+0x49c>
   116c4:	1ff00713          	li	a4,511
   116c8:	0049a503          	lw	a0,4(s3)
   116cc:	24f76e63          	bltu	a4,a5,11928 <_malloc_r+0x3b8>
   116d0:	ff87f713          	andi	a4,a5,-8
   116d4:	00870713          	addi	a4,a4,8
   116d8:	00e98733          	add	a4,s3,a4
   116dc:	00072683          	lw	a3,0(a4)
   116e0:	0057d613          	srli	a2,a5,0x5
   116e4:	00100793          	li	a5,1
   116e8:	00c797b3          	sll	a5,a5,a2
   116ec:	00f56533          	or	a0,a0,a5
   116f0:	ff870793          	addi	a5,a4,-8
   116f4:	00f42623          	sw	a5,12(s0)
   116f8:	00d42423          	sw	a3,8(s0)
   116fc:	00a9a223          	sw	a0,4(s3)
   11700:	00872023          	sw	s0,0(a4)
   11704:	0086a623          	sw	s0,12(a3)
   11708:	4025d793          	srai	a5,a1,0x2
   1170c:	00100613          	li	a2,1
   11710:	00f61633          	sll	a2,a2,a5
   11714:	0ac56a63          	bltu	a0,a2,117c8 <_malloc_r+0x258>
   11718:	00a677b3          	and	a5,a2,a0
   1171c:	02079463          	bnez	a5,11744 <_malloc_r+0x1d4>
   11720:	00161613          	slli	a2,a2,0x1
   11724:	ffc5f593          	andi	a1,a1,-4
   11728:	00a677b3          	and	a5,a2,a0
   1172c:	00458593          	addi	a1,a1,4
   11730:	00079a63          	bnez	a5,11744 <_malloc_r+0x1d4>
   11734:	00161613          	slli	a2,a2,0x1
   11738:	00a677b3          	and	a5,a2,a0
   1173c:	00458593          	addi	a1,a1,4
   11740:	fe078ae3          	beqz	a5,11734 <_malloc_r+0x1c4>
   11744:	00f00813          	li	a6,15
   11748:	00359313          	slli	t1,a1,0x3
   1174c:	00698333          	add	t1,s3,t1
   11750:	00030513          	mv	a0,t1
   11754:	00c52783          	lw	a5,12(a0)
   11758:	00058e13          	mv	t3,a1
   1175c:	24f50863          	beq	a0,a5,119ac <_malloc_r+0x43c>
   11760:	0047a703          	lw	a4,4(a5)
   11764:	00078413          	mv	s0,a5
   11768:	00c7a783          	lw	a5,12(a5)
   1176c:	ffc77713          	andi	a4,a4,-4
   11770:	409706b3          	sub	a3,a4,s1
   11774:	24d84863          	blt	a6,a3,119c4 <_malloc_r+0x454>
   11778:	fe06c2e3          	bltz	a3,1175c <_malloc_r+0x1ec>
   1177c:	00e40733          	add	a4,s0,a4
   11780:	00472683          	lw	a3,4(a4)
   11784:	00842603          	lw	a2,8(s0)
   11788:	00090513          	mv	a0,s2
   1178c:	0016e693          	ori	a3,a3,1
   11790:	00d72223          	sw	a3,4(a4)
   11794:	00f62623          	sw	a5,12(a2)
   11798:	00c7a423          	sw	a2,8(a5)
   1179c:	5a0000ef          	jal	11d3c <__malloc_unlock>
   117a0:	00840513          	addi	a0,s0,8
   117a4:	e59ff06f          	j	115fc <_malloc_r+0x8c>
   117a8:	00c00793          	li	a5,12
   117ac:	00f92023          	sw	a5,0(s2)
   117b0:	00000513          	li	a0,0
   117b4:	e49ff06f          	j	115fc <_malloc_r+0x8c>
   117b8:	20000613          	li	a2,512
   117bc:	04000593          	li	a1,64
   117c0:	03f00813          	li	a6,63
   117c4:	e8dff06f          	j	11650 <_malloc_r+0xe0>
   117c8:	0089a403          	lw	s0,8(s3)
   117cc:	01612823          	sw	s6,16(sp)
   117d0:	00442783          	lw	a5,4(s0)
   117d4:	ffc7fb13          	andi	s6,a5,-4
   117d8:	009b6863          	bltu	s6,s1,117e8 <_malloc_r+0x278>
   117dc:	409b0733          	sub	a4,s6,s1
   117e0:	00f00793          	li	a5,15
   117e4:	0ee7c063          	blt	a5,a4,118c4 <_malloc_r+0x354>
   117e8:	01912223          	sw	s9,4(sp)
   117ec:	09018c93          	addi	s9,gp,144 # 22910 <__malloc_sbrk_base>
   117f0:	000ca703          	lw	a4,0(s9)
   117f4:	01412c23          	sw	s4,24(sp)
   117f8:	01512a23          	sw	s5,20(sp)
   117fc:	01712623          	sw	s7,12(sp)
   11800:	0ac1aa83          	lw	s5,172(gp) # 2292c <__malloc_top_pad>
   11804:	fff00793          	li	a5,-1
   11808:	01640a33          	add	s4,s0,s6
   1180c:	01548ab3          	add	s5,s1,s5
   11810:	3cf70a63          	beq	a4,a5,11be4 <_malloc_r+0x674>
   11814:	000017b7          	lui	a5,0x1
   11818:	00f78793          	addi	a5,a5,15 # 100f <exit-0xf0a5>
   1181c:	00fa8ab3          	add	s5,s5,a5
   11820:	fffff7b7          	lui	a5,0xfffff
   11824:	00fafab3          	and	s5,s5,a5
   11828:	000a8593          	mv	a1,s5
   1182c:	00090513          	mv	a0,s2
   11830:	308050ef          	jal	16b38 <_sbrk_r>
   11834:	fff00793          	li	a5,-1
   11838:	00050b93          	mv	s7,a0
   1183c:	44f50e63          	beq	a0,a5,11c98 <_malloc_r+0x728>
   11840:	01812423          	sw	s8,8(sp)
   11844:	25456263          	bltu	a0,s4,11a88 <_malloc_r+0x518>
   11848:	2f818c13          	addi	s8,gp,760 # 22b78 <__malloc_current_mallinfo>
   1184c:	000c2583          	lw	a1,0(s8)
   11850:	00ba85b3          	add	a1,s5,a1
   11854:	00bc2023          	sw	a1,0(s8)
   11858:	00058713          	mv	a4,a1
   1185c:	2aaa1a63          	bne	s4,a0,11b10 <_malloc_r+0x5a0>
   11860:	01451793          	slli	a5,a0,0x14
   11864:	2a079663          	bnez	a5,11b10 <_malloc_r+0x5a0>
   11868:	0089ab83          	lw	s7,8(s3)
   1186c:	015b07b3          	add	a5,s6,s5
   11870:	0017e793          	ori	a5,a5,1
   11874:	00fba223          	sw	a5,4(s7)
   11878:	0a818713          	addi	a4,gp,168 # 22928 <__malloc_max_sbrked_mem>
   1187c:	00072683          	lw	a3,0(a4)
   11880:	00b6f463          	bgeu	a3,a1,11888 <_malloc_r+0x318>
   11884:	00b72023          	sw	a1,0(a4)
   11888:	0a418713          	addi	a4,gp,164 # 22924 <__malloc_max_total_mem>
   1188c:	00072683          	lw	a3,0(a4)
   11890:	00b6f463          	bgeu	a3,a1,11898 <_malloc_r+0x328>
   11894:	00b72023          	sw	a1,0(a4)
   11898:	00812c03          	lw	s8,8(sp)
   1189c:	000b8413          	mv	s0,s7
   118a0:	ffc7f793          	andi	a5,a5,-4
   118a4:	40978733          	sub	a4,a5,s1
   118a8:	3897ea63          	bltu	a5,s1,11c3c <_malloc_r+0x6cc>
   118ac:	00f00793          	li	a5,15
   118b0:	38e7d663          	bge	a5,a4,11c3c <_malloc_r+0x6cc>
   118b4:	01812a03          	lw	s4,24(sp)
   118b8:	01412a83          	lw	s5,20(sp)
   118bc:	00c12b83          	lw	s7,12(sp)
   118c0:	00412c83          	lw	s9,4(sp)
   118c4:	0014e793          	ori	a5,s1,1
   118c8:	00f42223          	sw	a5,4(s0)
   118cc:	009404b3          	add	s1,s0,s1
   118d0:	0099a423          	sw	s1,8(s3)
   118d4:	00176713          	ori	a4,a4,1
   118d8:	00090513          	mv	a0,s2
   118dc:	00e4a223          	sw	a4,4(s1)
   118e0:	45c000ef          	jal	11d3c <__malloc_unlock>
   118e4:	02c12083          	lw	ra,44(sp)
   118e8:	00840513          	addi	a0,s0,8
   118ec:	02812403          	lw	s0,40(sp)
   118f0:	01012b03          	lw	s6,16(sp)
   118f4:	02412483          	lw	s1,36(sp)
   118f8:	02012903          	lw	s2,32(sp)
   118fc:	01c12983          	lw	s3,28(sp)
   11900:	03010113          	addi	sp,sp,48
   11904:	00008067          	ret
   11908:	0049a503          	lw	a0,4(s3)
   1190c:	dfdff06f          	j	11708 <_malloc_r+0x198>
   11910:	00842603          	lw	a2,8(s0)
   11914:	cc5ff06f          	j	115d8 <_malloc_r+0x68>
   11918:	00c7a403          	lw	s0,12(a5) # fffff00c <__BSS_END__+0xfffdc2dc>
   1191c:	00258593          	addi	a1,a1,2
   11920:	d6878ae3          	beq	a5,s0,11694 <_malloc_r+0x124>
   11924:	ca5ff06f          	j	115c8 <_malloc_r+0x58>
   11928:	0097d713          	srli	a4,a5,0x9
   1192c:	00400693          	li	a3,4
   11930:	14e6f263          	bgeu	a3,a4,11a74 <_malloc_r+0x504>
   11934:	01400693          	li	a3,20
   11938:	32e6e463          	bltu	a3,a4,11c60 <_malloc_r+0x6f0>
   1193c:	05c70613          	addi	a2,a4,92
   11940:	05b70693          	addi	a3,a4,91
   11944:	00361613          	slli	a2,a2,0x3
   11948:	00c98633          	add	a2,s3,a2
   1194c:	00062703          	lw	a4,0(a2)
   11950:	ff860613          	addi	a2,a2,-8
   11954:	00e61863          	bne	a2,a4,11964 <_malloc_r+0x3f4>
   11958:	2940006f          	j	11bec <_malloc_r+0x67c>
   1195c:	00872703          	lw	a4,8(a4)
   11960:	00e60863          	beq	a2,a4,11970 <_malloc_r+0x400>
   11964:	00472683          	lw	a3,4(a4)
   11968:	ffc6f693          	andi	a3,a3,-4
   1196c:	fed7e8e3          	bltu	a5,a3,1195c <_malloc_r+0x3ec>
   11970:	00c72603          	lw	a2,12(a4)
   11974:	00c42623          	sw	a2,12(s0)
   11978:	00e42423          	sw	a4,8(s0)
   1197c:	00862423          	sw	s0,8(a2)
   11980:	00872623          	sw	s0,12(a4)
   11984:	d85ff06f          	j	11708 <_malloc_r+0x198>
   11988:	01400713          	li	a4,20
   1198c:	10f77863          	bgeu	a4,a5,11a9c <_malloc_r+0x52c>
   11990:	05400713          	li	a4,84
   11994:	2ef76463          	bltu	a4,a5,11c7c <_malloc_r+0x70c>
   11998:	00c4d793          	srli	a5,s1,0xc
   1199c:	06f78593          	addi	a1,a5,111
   119a0:	06e78813          	addi	a6,a5,110
   119a4:	00359613          	slli	a2,a1,0x3
   119a8:	ca9ff06f          	j	11650 <_malloc_r+0xe0>
   119ac:	001e0e13          	addi	t3,t3,1
   119b0:	003e7793          	andi	a5,t3,3
   119b4:	00850513          	addi	a0,a0,8
   119b8:	10078063          	beqz	a5,11ab8 <_malloc_r+0x548>
   119bc:	00c52783          	lw	a5,12(a0)
   119c0:	d9dff06f          	j	1175c <_malloc_r+0x1ec>
   119c4:	00842603          	lw	a2,8(s0)
   119c8:	0014e593          	ori	a1,s1,1
   119cc:	00b42223          	sw	a1,4(s0)
   119d0:	00f62623          	sw	a5,12(a2)
   119d4:	00c7a423          	sw	a2,8(a5)
   119d8:	009404b3          	add	s1,s0,s1
   119dc:	0099aa23          	sw	s1,20(s3)
   119e0:	0099a823          	sw	s1,16(s3)
   119e4:	0016e793          	ori	a5,a3,1
   119e8:	0114a623          	sw	a7,12(s1)
   119ec:	0114a423          	sw	a7,8(s1)
   119f0:	00f4a223          	sw	a5,4(s1)
   119f4:	00e40733          	add	a4,s0,a4
   119f8:	00090513          	mv	a0,s2
   119fc:	00d72023          	sw	a3,0(a4)
   11a00:	33c000ef          	jal	11d3c <__malloc_unlock>
   11a04:	00840513          	addi	a0,s0,8
   11a08:	bf5ff06f          	j	115fc <_malloc_r+0x8c>
   11a0c:	00f407b3          	add	a5,s0,a5
   11a10:	0047a703          	lw	a4,4(a5)
   11a14:	00090513          	mv	a0,s2
   11a18:	00176713          	ori	a4,a4,1
   11a1c:	00e7a223          	sw	a4,4(a5)
   11a20:	31c000ef          	jal	11d3c <__malloc_unlock>
   11a24:	00840513          	addi	a0,s0,8
   11a28:	bd5ff06f          	j	115fc <_malloc_r+0x8c>
   11a2c:	0034d593          	srli	a1,s1,0x3
   11a30:	00848793          	addi	a5,s1,8
   11a34:	b7dff06f          	j	115b0 <_malloc_r+0x40>
   11a38:	0014e693          	ori	a3,s1,1
   11a3c:	00d42223          	sw	a3,4(s0)
   11a40:	009404b3          	add	s1,s0,s1
   11a44:	0099aa23          	sw	s1,20(s3)
   11a48:	0099a823          	sw	s1,16(s3)
   11a4c:	00176693          	ori	a3,a4,1
   11a50:	0114a623          	sw	a7,12(s1)
   11a54:	0114a423          	sw	a7,8(s1)
   11a58:	00d4a223          	sw	a3,4(s1)
   11a5c:	00f407b3          	add	a5,s0,a5
   11a60:	00090513          	mv	a0,s2
   11a64:	00e7a023          	sw	a4,0(a5)
   11a68:	2d4000ef          	jal	11d3c <__malloc_unlock>
   11a6c:	00840513          	addi	a0,s0,8
   11a70:	b8dff06f          	j	115fc <_malloc_r+0x8c>
   11a74:	0067d713          	srli	a4,a5,0x6
   11a78:	03970613          	addi	a2,a4,57
   11a7c:	03870693          	addi	a3,a4,56
   11a80:	00361613          	slli	a2,a2,0x3
   11a84:	ec5ff06f          	j	11948 <_malloc_r+0x3d8>
   11a88:	07340c63          	beq	s0,s3,11b00 <_malloc_r+0x590>
   11a8c:	0089a403          	lw	s0,8(s3)
   11a90:	00812c03          	lw	s8,8(sp)
   11a94:	00442783          	lw	a5,4(s0)
   11a98:	e09ff06f          	j	118a0 <_malloc_r+0x330>
   11a9c:	05c78593          	addi	a1,a5,92
   11aa0:	05b78813          	addi	a6,a5,91
   11aa4:	00359613          	slli	a2,a1,0x3
   11aa8:	ba9ff06f          	j	11650 <_malloc_r+0xe0>
   11aac:	00832783          	lw	a5,8(t1)
   11ab0:	fff58593          	addi	a1,a1,-1
   11ab4:	26679e63          	bne	a5,t1,11d30 <_malloc_r+0x7c0>
   11ab8:	0035f793          	andi	a5,a1,3
   11abc:	ff830313          	addi	t1,t1,-8
   11ac0:	fe0796e3          	bnez	a5,11aac <_malloc_r+0x53c>
   11ac4:	0049a703          	lw	a4,4(s3)
   11ac8:	fff64793          	not	a5,a2
   11acc:	00e7f7b3          	and	a5,a5,a4
   11ad0:	00f9a223          	sw	a5,4(s3)
   11ad4:	00161613          	slli	a2,a2,0x1
   11ad8:	cec7e8e3          	bltu	a5,a2,117c8 <_malloc_r+0x258>
   11adc:	ce0606e3          	beqz	a2,117c8 <_malloc_r+0x258>
   11ae0:	00f67733          	and	a4,a2,a5
   11ae4:	00071a63          	bnez	a4,11af8 <_malloc_r+0x588>
   11ae8:	00161613          	slli	a2,a2,0x1
   11aec:	00f67733          	and	a4,a2,a5
   11af0:	004e0e13          	addi	t3,t3,4
   11af4:	fe070ae3          	beqz	a4,11ae8 <_malloc_r+0x578>
   11af8:	000e0593          	mv	a1,t3
   11afc:	c4dff06f          	j	11748 <_malloc_r+0x1d8>
   11b00:	2f818c13          	addi	s8,gp,760 # 22b78 <__malloc_current_mallinfo>
   11b04:	000c2703          	lw	a4,0(s8)
   11b08:	00ea8733          	add	a4,s5,a4
   11b0c:	00ec2023          	sw	a4,0(s8)
   11b10:	000ca683          	lw	a3,0(s9)
   11b14:	fff00793          	li	a5,-1
   11b18:	18f68663          	beq	a3,a5,11ca4 <_malloc_r+0x734>
   11b1c:	414b87b3          	sub	a5,s7,s4
   11b20:	00e787b3          	add	a5,a5,a4
   11b24:	00fc2023          	sw	a5,0(s8)
   11b28:	007bfc93          	andi	s9,s7,7
   11b2c:	0c0c8c63          	beqz	s9,11c04 <_malloc_r+0x694>
   11b30:	419b8bb3          	sub	s7,s7,s9
   11b34:	000017b7          	lui	a5,0x1
   11b38:	00878793          	addi	a5,a5,8 # 1008 <exit-0xf0ac>
   11b3c:	008b8b93          	addi	s7,s7,8
   11b40:	419785b3          	sub	a1,a5,s9
   11b44:	015b8ab3          	add	s5,s7,s5
   11b48:	415585b3          	sub	a1,a1,s5
   11b4c:	01459593          	slli	a1,a1,0x14
   11b50:	0145da13          	srli	s4,a1,0x14
   11b54:	000a0593          	mv	a1,s4
   11b58:	00090513          	mv	a0,s2
   11b5c:	7dd040ef          	jal	16b38 <_sbrk_r>
   11b60:	fff00793          	li	a5,-1
   11b64:	18f50063          	beq	a0,a5,11ce4 <_malloc_r+0x774>
   11b68:	41750533          	sub	a0,a0,s7
   11b6c:	01450ab3          	add	s5,a0,s4
   11b70:	000c2703          	lw	a4,0(s8)
   11b74:	0179a423          	sw	s7,8(s3)
   11b78:	001ae793          	ori	a5,s5,1
   11b7c:	00ea05b3          	add	a1,s4,a4
   11b80:	00bc2023          	sw	a1,0(s8)
   11b84:	00fba223          	sw	a5,4(s7)
   11b88:	cf3408e3          	beq	s0,s3,11878 <_malloc_r+0x308>
   11b8c:	00f00693          	li	a3,15
   11b90:	0b66f063          	bgeu	a3,s6,11c30 <_malloc_r+0x6c0>
   11b94:	00442703          	lw	a4,4(s0)
   11b98:	ff4b0793          	addi	a5,s6,-12
   11b9c:	ff87f793          	andi	a5,a5,-8
   11ba0:	00177713          	andi	a4,a4,1
   11ba4:	00f76733          	or	a4,a4,a5
   11ba8:	00e42223          	sw	a4,4(s0)
   11bac:	00500613          	li	a2,5
   11bb0:	00f40733          	add	a4,s0,a5
   11bb4:	00c72223          	sw	a2,4(a4)
   11bb8:	00c72423          	sw	a2,8(a4)
   11bbc:	00f6e663          	bltu	a3,a5,11bc8 <_malloc_r+0x658>
   11bc0:	004ba783          	lw	a5,4(s7)
   11bc4:	cb5ff06f          	j	11878 <_malloc_r+0x308>
   11bc8:	00840593          	addi	a1,s0,8
   11bcc:	00090513          	mv	a0,s2
   11bd0:	e9cff0ef          	jal	1126c <_free_r>
   11bd4:	0089ab83          	lw	s7,8(s3)
   11bd8:	000c2583          	lw	a1,0(s8)
   11bdc:	004ba783          	lw	a5,4(s7)
   11be0:	c99ff06f          	j	11878 <_malloc_r+0x308>
   11be4:	010a8a93          	addi	s5,s5,16
   11be8:	c41ff06f          	j	11828 <_malloc_r+0x2b8>
   11bec:	4026d693          	srai	a3,a3,0x2
   11bf0:	00100793          	li	a5,1
   11bf4:	00d797b3          	sll	a5,a5,a3
   11bf8:	00f56533          	or	a0,a0,a5
   11bfc:	00a9a223          	sw	a0,4(s3)
   11c00:	d75ff06f          	j	11974 <_malloc_r+0x404>
   11c04:	015b85b3          	add	a1,s7,s5
   11c08:	40b005b3          	neg	a1,a1
   11c0c:	01459593          	slli	a1,a1,0x14
   11c10:	0145da13          	srli	s4,a1,0x14
   11c14:	000a0593          	mv	a1,s4
   11c18:	00090513          	mv	a0,s2
   11c1c:	71d040ef          	jal	16b38 <_sbrk_r>
   11c20:	fff00793          	li	a5,-1
   11c24:	f4f512e3          	bne	a0,a5,11b68 <_malloc_r+0x5f8>
   11c28:	00000a13          	li	s4,0
   11c2c:	f45ff06f          	j	11b70 <_malloc_r+0x600>
   11c30:	00812c03          	lw	s8,8(sp)
   11c34:	00100793          	li	a5,1
   11c38:	00fba223          	sw	a5,4(s7)
   11c3c:	00090513          	mv	a0,s2
   11c40:	0fc000ef          	jal	11d3c <__malloc_unlock>
   11c44:	00000513          	li	a0,0
   11c48:	01812a03          	lw	s4,24(sp)
   11c4c:	01412a83          	lw	s5,20(sp)
   11c50:	01012b03          	lw	s6,16(sp)
   11c54:	00c12b83          	lw	s7,12(sp)
   11c58:	00412c83          	lw	s9,4(sp)
   11c5c:	9a1ff06f          	j	115fc <_malloc_r+0x8c>
   11c60:	05400693          	li	a3,84
   11c64:	04e6e463          	bltu	a3,a4,11cac <_malloc_r+0x73c>
   11c68:	00c7d713          	srli	a4,a5,0xc
   11c6c:	06f70613          	addi	a2,a4,111
   11c70:	06e70693          	addi	a3,a4,110
   11c74:	00361613          	slli	a2,a2,0x3
   11c78:	cd1ff06f          	j	11948 <_malloc_r+0x3d8>
   11c7c:	15400713          	li	a4,340
   11c80:	04f76463          	bltu	a4,a5,11cc8 <_malloc_r+0x758>
   11c84:	00f4d793          	srli	a5,s1,0xf
   11c88:	07878593          	addi	a1,a5,120
   11c8c:	07778813          	addi	a6,a5,119
   11c90:	00359613          	slli	a2,a1,0x3
   11c94:	9bdff06f          	j	11650 <_malloc_r+0xe0>
   11c98:	0089a403          	lw	s0,8(s3)
   11c9c:	00442783          	lw	a5,4(s0)
   11ca0:	c01ff06f          	j	118a0 <_malloc_r+0x330>
   11ca4:	017ca023          	sw	s7,0(s9)
   11ca8:	e81ff06f          	j	11b28 <_malloc_r+0x5b8>
   11cac:	15400693          	li	a3,340
   11cb0:	04e6e463          	bltu	a3,a4,11cf8 <_malloc_r+0x788>
   11cb4:	00f7d713          	srli	a4,a5,0xf
   11cb8:	07870613          	addi	a2,a4,120
   11cbc:	07770693          	addi	a3,a4,119
   11cc0:	00361613          	slli	a2,a2,0x3
   11cc4:	c85ff06f          	j	11948 <_malloc_r+0x3d8>
   11cc8:	55400713          	li	a4,1364
   11ccc:	04f76463          	bltu	a4,a5,11d14 <_malloc_r+0x7a4>
   11cd0:	0124d793          	srli	a5,s1,0x12
   11cd4:	07d78593          	addi	a1,a5,125
   11cd8:	07c78813          	addi	a6,a5,124
   11cdc:	00359613          	slli	a2,a1,0x3
   11ce0:	971ff06f          	j	11650 <_malloc_r+0xe0>
   11ce4:	ff8c8c93          	addi	s9,s9,-8
   11ce8:	019a8ab3          	add	s5,s5,s9
   11cec:	417a8ab3          	sub	s5,s5,s7
   11cf0:	00000a13          	li	s4,0
   11cf4:	e7dff06f          	j	11b70 <_malloc_r+0x600>
   11cf8:	55400693          	li	a3,1364
   11cfc:	02e6e463          	bltu	a3,a4,11d24 <_malloc_r+0x7b4>
   11d00:	0127d713          	srli	a4,a5,0x12
   11d04:	07d70613          	addi	a2,a4,125
   11d08:	07c70693          	addi	a3,a4,124
   11d0c:	00361613          	slli	a2,a2,0x3
   11d10:	c39ff06f          	j	11948 <_malloc_r+0x3d8>
   11d14:	3f800613          	li	a2,1016
   11d18:	07f00593          	li	a1,127
   11d1c:	07e00813          	li	a6,126
   11d20:	931ff06f          	j	11650 <_malloc_r+0xe0>
   11d24:	3f800613          	li	a2,1016
   11d28:	07e00693          	li	a3,126
   11d2c:	c1dff06f          	j	11948 <_malloc_r+0x3d8>
   11d30:	0049a783          	lw	a5,4(s3)
   11d34:	da1ff06f          	j	11ad4 <_malloc_r+0x564>

00011d38 <__malloc_lock>:
   11d38:	00008067          	ret

00011d3c <__malloc_unlock>:
   11d3c:	00008067          	ret

00011d40 <_vfprintf_r>:
   11d40:	e3010113          	addi	sp,sp,-464
   11d44:	1c112623          	sw	ra,460(sp)
   11d48:	1c812423          	sw	s0,456(sp)
   11d4c:	1c912223          	sw	s1,452(sp)
   11d50:	1d212023          	sw	s2,448(sp)
   11d54:	00058493          	mv	s1,a1
   11d58:	00060913          	mv	s2,a2
   11d5c:	00d12a23          	sw	a3,20(sp)
   11d60:	00050413          	mv	s0,a0
   11d64:	00a12423          	sw	a0,8(sp)
   11d68:	5c1040ef          	jal	16b28 <_localeconv_r>
   11d6c:	00052703          	lw	a4,0(a0)
   11d70:	00070513          	mv	a0,a4
   11d74:	02e12623          	sw	a4,44(sp)
   11d78:	9e4ff0ef          	jal	10f5c <strlen>
   11d7c:	02a12423          	sw	a0,40(sp)
   11d80:	0c012823          	sw	zero,208(sp)
   11d84:	0c012a23          	sw	zero,212(sp)
   11d88:	0c012c23          	sw	zero,216(sp)
   11d8c:	0c012e23          	sw	zero,220(sp)
   11d90:	00040863          	beqz	s0,11da0 <_vfprintf_r+0x60>
   11d94:	03442703          	lw	a4,52(s0)
   11d98:	00071463          	bnez	a4,11da0 <_vfprintf_r+0x60>
   11d9c:	10d0106f          	j	136a8 <_vfprintf_r+0x1968>
   11da0:	00c49703          	lh	a4,12(s1)
   11da4:	01271693          	slli	a3,a4,0x12
   11da8:	0206c663          	bltz	a3,11dd4 <_vfprintf_r+0x94>
   11dac:	0644a683          	lw	a3,100(s1)
   11db0:	000025b7          	lui	a1,0x2
   11db4:	ffffe637          	lui	a2,0xffffe
   11db8:	00b76733          	or	a4,a4,a1
   11dbc:	fff60613          	addi	a2,a2,-1 # ffffdfff <__BSS_END__+0xfffdb2cf>
   11dc0:	01071713          	slli	a4,a4,0x10
   11dc4:	41075713          	srai	a4,a4,0x10
   11dc8:	00c6f6b3          	and	a3,a3,a2
   11dcc:	00e49623          	sh	a4,12(s1)
   11dd0:	06d4a223          	sw	a3,100(s1)
   11dd4:	00877693          	andi	a3,a4,8
   11dd8:	2e068e63          	beqz	a3,120d4 <_vfprintf_r+0x394>
   11ddc:	0104a683          	lw	a3,16(s1)
   11de0:	2e068a63          	beqz	a3,120d4 <_vfprintf_r+0x394>
   11de4:	01a77713          	andi	a4,a4,26
   11de8:	00a00693          	li	a3,10
   11dec:	30d70663          	beq	a4,a3,120f8 <_vfprintf_r+0x3b8>
   11df0:	1b312e23          	sw	s3,444(sp)
   11df4:	1b412c23          	sw	s4,440(sp)
   11df8:	1ba12023          	sw	s10,416(sp)
   11dfc:	1b512a23          	sw	s5,436(sp)
   11e00:	1b612823          	sw	s6,432(sp)
   11e04:	1b712623          	sw	s7,428(sp)
   11e08:	1b812423          	sw	s8,424(sp)
   11e0c:	1b912223          	sw	s9,420(sp)
   11e10:	19b12e23          	sw	s11,412(sp)
   11e14:	00090d13          	mv	s10,s2
   11e18:	000d4703          	lbu	a4,0(s10)
   11e1c:	0ec10993          	addi	s3,sp,236
   11e20:	0d312223          	sw	s3,196(sp)
   11e24:	0c012623          	sw	zero,204(sp)
   11e28:	0c012423          	sw	zero,200(sp)
   11e2c:	00012e23          	sw	zero,28(sp)
   11e30:	02012823          	sw	zero,48(sp)
   11e34:	02012e23          	sw	zero,60(sp)
   11e38:	02012c23          	sw	zero,56(sp)
   11e3c:	04012223          	sw	zero,68(sp)
   11e40:	04012023          	sw	zero,64(sp)
   11e44:	00012623          	sw	zero,12(sp)
   11e48:	01000413          	li	s0,16
   11e4c:	00098a13          	mv	s4,s3
   11e50:	22070463          	beqz	a4,12078 <_vfprintf_r+0x338>
   11e54:	000d0a93          	mv	s5,s10
   11e58:	02500693          	li	a3,37
   11e5c:	3ed70e63          	beq	a4,a3,12258 <_vfprintf_r+0x518>
   11e60:	001ac703          	lbu	a4,1(s5)
   11e64:	001a8a93          	addi	s5,s5,1
   11e68:	fe071ae3          	bnez	a4,11e5c <_vfprintf_r+0x11c>
   11e6c:	41aa8933          	sub	s2,s5,s10
   11e70:	21aa8463          	beq	s5,s10,12078 <_vfprintf_r+0x338>
   11e74:	0cc12683          	lw	a3,204(sp)
   11e78:	0c812703          	lw	a4,200(sp)
   11e7c:	01aa2023          	sw	s10,0(s4)
   11e80:	012686b3          	add	a3,a3,s2
   11e84:	00170713          	addi	a4,a4,1
   11e88:	012a2223          	sw	s2,4(s4)
   11e8c:	0cd12623          	sw	a3,204(sp)
   11e90:	0ce12423          	sw	a4,200(sp)
   11e94:	00700693          	li	a3,7
   11e98:	008a0a13          	addi	s4,s4,8
   11e9c:	3ce6c663          	blt	a3,a4,12268 <_vfprintf_r+0x528>
   11ea0:	00c12783          	lw	a5,12(sp)
   11ea4:	000ac703          	lbu	a4,0(s5)
   11ea8:	012787b3          	add	a5,a5,s2
   11eac:	00f12623          	sw	a5,12(sp)
   11eb0:	1c070463          	beqz	a4,12078 <_vfprintf_r+0x338>
   11eb4:	001ac883          	lbu	a7,1(s5)
   11eb8:	0a0103a3          	sb	zero,167(sp)
   11ebc:	001a8a93          	addi	s5,s5,1
   11ec0:	fff00b13          	li	s6,-1
   11ec4:	00000b93          	li	s7,0
   11ec8:	00000c93          	li	s9,0
   11ecc:	05a00913          	li	s2,90
   11ed0:	001a8a93          	addi	s5,s5,1
   11ed4:	fe088793          	addi	a5,a7,-32
   11ed8:	04f96a63          	bltu	s2,a5,11f2c <_vfprintf_r+0x1ec>
   11edc:	0000f717          	auipc	a4,0xf
   11ee0:	dac70713          	addi	a4,a4,-596 # 20c88 <_exit+0x2d8>
   11ee4:	00279793          	slli	a5,a5,0x2
   11ee8:	00e787b3          	add	a5,a5,a4
   11eec:	0007a783          	lw	a5,0(a5)
   11ef0:	00e787b3          	add	a5,a5,a4
   11ef4:	00078067          	jr	a5
   11ef8:	00000b93          	li	s7,0
   11efc:	fd088793          	addi	a5,a7,-48
   11f00:	00900693          	li	a3,9
   11f04:	000ac883          	lbu	a7,0(s5)
   11f08:	002b9713          	slli	a4,s7,0x2
   11f0c:	01770bb3          	add	s7,a4,s7
   11f10:	001b9b93          	slli	s7,s7,0x1
   11f14:	01778bb3          	add	s7,a5,s7
   11f18:	fd088793          	addi	a5,a7,-48
   11f1c:	001a8a93          	addi	s5,s5,1
   11f20:	fef6f2e3          	bgeu	a3,a5,11f04 <_vfprintf_r+0x1c4>
   11f24:	fe088793          	addi	a5,a7,-32
   11f28:	faf97ae3          	bgeu	s2,a5,11edc <_vfprintf_r+0x19c>
   11f2c:	14088663          	beqz	a7,12078 <_vfprintf_r+0x338>
   11f30:	13110623          	sb	a7,300(sp)
   11f34:	0a0103a3          	sb	zero,167(sp)
   11f38:	00100d93          	li	s11,1
   11f3c:	00100913          	li	s2,1
   11f40:	12c10d13          	addi	s10,sp,300
   11f44:	00012823          	sw	zero,16(sp)
   11f48:	00000b13          	li	s6,0
   11f4c:	02012223          	sw	zero,36(sp)
   11f50:	02012023          	sw	zero,32(sp)
   11f54:	00012c23          	sw	zero,24(sp)
   11f58:	002cf293          	andi	t0,s9,2
   11f5c:	00028463          	beqz	t0,11f64 <_vfprintf_r+0x224>
   11f60:	002d8d93          	addi	s11,s11,2
   11f64:	084cff93          	andi	t6,s9,132
   11f68:	0cc12603          	lw	a2,204(sp)
   11f6c:	000f9663          	bnez	t6,11f78 <_vfprintf_r+0x238>
   11f70:	41bb8733          	sub	a4,s7,s11
   11f74:	46e04ae3          	bgtz	a4,12be8 <_vfprintf_r+0xea8>
   11f78:	0a714703          	lbu	a4,167(sp)
   11f7c:	02070a63          	beqz	a4,11fb0 <_vfprintf_r+0x270>
   11f80:	0c812703          	lw	a4,200(sp)
   11f84:	0a710593          	addi	a1,sp,167
   11f88:	00ba2023          	sw	a1,0(s4)
   11f8c:	00160613          	addi	a2,a2,1
   11f90:	00100593          	li	a1,1
   11f94:	00170713          	addi	a4,a4,1
   11f98:	00ba2223          	sw	a1,4(s4)
   11f9c:	0cc12623          	sw	a2,204(sp)
   11fa0:	0ce12423          	sw	a4,200(sp)
   11fa4:	00700593          	li	a1,7
   11fa8:	008a0a13          	addi	s4,s4,8
   11fac:	3ee5c663          	blt	a1,a4,12398 <_vfprintf_r+0x658>
   11fb0:	02028a63          	beqz	t0,11fe4 <_vfprintf_r+0x2a4>
   11fb4:	0c812703          	lw	a4,200(sp)
   11fb8:	00200593          	li	a1,2
   11fbc:	00260613          	addi	a2,a2,2
   11fc0:	00170713          	addi	a4,a4,1
   11fc4:	0a810793          	addi	a5,sp,168
   11fc8:	00ba2223          	sw	a1,4(s4)
   11fcc:	00fa2023          	sw	a5,0(s4)
   11fd0:	0cc12623          	sw	a2,204(sp)
   11fd4:	0ce12423          	sw	a4,200(sp)
   11fd8:	00700593          	li	a1,7
   11fdc:	008a0a13          	addi	s4,s4,8
   11fe0:	50e5cae3          	blt	a1,a4,12cf4 <_vfprintf_r+0xfb4>
   11fe4:	08000713          	li	a4,128
   11fe8:	1eef86e3          	beq	t6,a4,129d4 <_vfprintf_r+0xc94>
   11fec:	412b0b33          	sub	s6,s6,s2
   11ff0:	2f6040e3          	bgtz	s6,12ad0 <_vfprintf_r+0xd90>
   11ff4:	100cf713          	andi	a4,s9,256
   11ff8:	040712e3          	bnez	a4,1283c <_vfprintf_r+0xafc>
   11ffc:	0c812783          	lw	a5,200(sp)
   12000:	01260633          	add	a2,a2,s2
   12004:	01aa2023          	sw	s10,0(s4)
   12008:	00178793          	addi	a5,a5,1
   1200c:	012a2223          	sw	s2,4(s4)
   12010:	0cc12623          	sw	a2,204(sp)
   12014:	0cf12423          	sw	a5,200(sp)
   12018:	00700713          	li	a4,7
   1201c:	4af74e63          	blt	a4,a5,124d8 <_vfprintf_r+0x798>
   12020:	008a0a13          	addi	s4,s4,8
   12024:	004cfe13          	andi	t3,s9,4
   12028:	000e0663          	beqz	t3,12034 <_vfprintf_r+0x2f4>
   1202c:	41bb8933          	sub	s2,s7,s11
   12030:	4f204ae3          	bgtz	s2,12d24 <_vfprintf_r+0xfe4>
   12034:	000b8313          	mv	t1,s7
   12038:	01bbd463          	bge	s7,s11,12040 <_vfprintf_r+0x300>
   1203c:	000d8313          	mv	t1,s11
   12040:	00c12783          	lw	a5,12(sp)
   12044:	006787b3          	add	a5,a5,t1
   12048:	00f12623          	sw	a5,12(sp)
   1204c:	360614e3          	bnez	a2,12bb4 <_vfprintf_r+0xe74>
   12050:	01012783          	lw	a5,16(sp)
   12054:	0c012423          	sw	zero,200(sp)
   12058:	00078863          	beqz	a5,12068 <_vfprintf_r+0x328>
   1205c:	01012583          	lw	a1,16(sp)
   12060:	00812503          	lw	a0,8(sp)
   12064:	a08ff0ef          	jal	1126c <_free_r>
   12068:	00098a13          	mv	s4,s3
   1206c:	000a8d13          	mv	s10,s5
   12070:	000d4703          	lbu	a4,0(s10)
   12074:	de0710e3          	bnez	a4,11e54 <_vfprintf_r+0x114>
   12078:	0cc12783          	lw	a5,204(sp)
   1207c:	00078463          	beqz	a5,12084 <_vfprintf_r+0x344>
   12080:	3a10106f          	j	13c20 <_vfprintf_r+0x1ee0>
   12084:	00c4d783          	lhu	a5,12(s1)
   12088:	1bc12983          	lw	s3,444(sp)
   1208c:	1b812a03          	lw	s4,440(sp)
   12090:	0407f793          	andi	a5,a5,64
   12094:	1b412a83          	lw	s5,436(sp)
   12098:	1b012b03          	lw	s6,432(sp)
   1209c:	1ac12b83          	lw	s7,428(sp)
   120a0:	1a812c03          	lw	s8,424(sp)
   120a4:	1a412c83          	lw	s9,420(sp)
   120a8:	1a012d03          	lw	s10,416(sp)
   120ac:	19c12d83          	lw	s11,412(sp)
   120b0:	00078463          	beqz	a5,120b8 <_vfprintf_r+0x378>
   120b4:	29c0206f          	j	14350 <_vfprintf_r+0x2610>
   120b8:	1cc12083          	lw	ra,460(sp)
   120bc:	1c812403          	lw	s0,456(sp)
   120c0:	00c12503          	lw	a0,12(sp)
   120c4:	1c412483          	lw	s1,452(sp)
   120c8:	1c012903          	lw	s2,448(sp)
   120cc:	1d010113          	addi	sp,sp,464
   120d0:	00008067          	ret
   120d4:	00812503          	lw	a0,8(sp)
   120d8:	00048593          	mv	a1,s1
   120dc:	4f8040ef          	jal	165d4 <__swsetup_r>
   120e0:	00050463          	beqz	a0,120e8 <_vfprintf_r+0x3a8>
   120e4:	26c0206f          	j	14350 <_vfprintf_r+0x2610>
   120e8:	00c49703          	lh	a4,12(s1)
   120ec:	00a00693          	li	a3,10
   120f0:	01a77713          	andi	a4,a4,26
   120f4:	ced71ee3          	bne	a4,a3,11df0 <_vfprintf_r+0xb0>
   120f8:	00e49703          	lh	a4,14(s1)
   120fc:	ce074ae3          	bltz	a4,11df0 <_vfprintf_r+0xb0>
   12100:	01412683          	lw	a3,20(sp)
   12104:	00812503          	lw	a0,8(sp)
   12108:	00090613          	mv	a2,s2
   1210c:	00048593          	mv	a1,s1
   12110:	52c020ef          	jal	1463c <__sbprintf>
   12114:	00a12623          	sw	a0,12(sp)
   12118:	fa1ff06f          	j	120b8 <_vfprintf_r+0x378>
   1211c:	00812c03          	lw	s8,8(sp)
   12120:	000c0513          	mv	a0,s8
   12124:	205040ef          	jal	16b28 <_localeconv_r>
   12128:	00452783          	lw	a5,4(a0)
   1212c:	00078513          	mv	a0,a5
   12130:	04f12023          	sw	a5,64(sp)
   12134:	e29fe0ef          	jal	10f5c <strlen>
   12138:	00050793          	mv	a5,a0
   1213c:	000c0513          	mv	a0,s8
   12140:	04f12223          	sw	a5,68(sp)
   12144:	00078c13          	mv	s8,a5
   12148:	1e1040ef          	jal	16b28 <_localeconv_r>
   1214c:	00852703          	lw	a4,8(a0)
   12150:	02e12c23          	sw	a4,56(sp)
   12154:	740c1ce3          	bnez	s8,130ac <_vfprintf_r+0x136c>
   12158:	000ac883          	lbu	a7,0(s5)
   1215c:	d75ff06f          	j	11ed0 <_vfprintf_r+0x190>
   12160:	000ac883          	lbu	a7,0(s5)
   12164:	020cec93          	ori	s9,s9,32
   12168:	d69ff06f          	j	11ed0 <_vfprintf_r+0x190>
   1216c:	010cec93          	ori	s9,s9,16
   12170:	020cf793          	andi	a5,s9,32
   12174:	3a078a63          	beqz	a5,12528 <_vfprintf_r+0x7e8>
   12178:	01412783          	lw	a5,20(sp)
   1217c:	00778c13          	addi	s8,a5,7
   12180:	ff8c7c13          	andi	s8,s8,-8
   12184:	004c2783          	lw	a5,4(s8)
   12188:	000c2903          	lw	s2,0(s8)
   1218c:	008c0713          	addi	a4,s8,8
   12190:	00e12a23          	sw	a4,20(sp)
   12194:	00078d93          	mv	s11,a5
   12198:	3c07c263          	bltz	a5,1255c <_vfprintf_r+0x81c>
   1219c:	000c8e93          	mv	t4,s9
   121a0:	4e0b4663          	bltz	s6,1268c <_vfprintf_r+0x94c>
   121a4:	01b967b3          	or	a5,s2,s11
   121a8:	f7fcfe93          	andi	t4,s9,-129
   121ac:	4e079063          	bnez	a5,1268c <_vfprintf_r+0x94c>
   121b0:	4e0b1463          	bnez	s6,12698 <_vfprintf_r+0x958>
   121b4:	00000913          	li	s2,0
   121b8:	000e8c93          	mv	s9,t4
   121bc:	19010d13          	addi	s10,sp,400
   121c0:	0a714703          	lbu	a4,167(sp)
   121c4:	000b0d93          	mv	s11,s6
   121c8:	012b5463          	bge	s6,s2,121d0 <_vfprintf_r+0x490>
   121cc:	00090d93          	mv	s11,s2
   121d0:	00012823          	sw	zero,16(sp)
   121d4:	02012223          	sw	zero,36(sp)
   121d8:	02012023          	sw	zero,32(sp)
   121dc:	00012c23          	sw	zero,24(sp)
   121e0:	d6070ce3          	beqz	a4,11f58 <_vfprintf_r+0x218>
   121e4:	001d8d93          	addi	s11,s11,1
   121e8:	d71ff06f          	j	11f58 <_vfprintf_r+0x218>
   121ec:	010cec93          	ori	s9,s9,16
   121f0:	020cf793          	andi	a5,s9,32
   121f4:	30078263          	beqz	a5,124f8 <_vfprintf_r+0x7b8>
   121f8:	01412783          	lw	a5,20(sp)
   121fc:	00778c13          	addi	s8,a5,7
   12200:	ff8c7c13          	andi	s8,s8,-8
   12204:	000c2903          	lw	s2,0(s8)
   12208:	004c2d83          	lw	s11,4(s8)
   1220c:	008c0793          	addi	a5,s8,8
   12210:	00f12a23          	sw	a5,20(sp)
   12214:	bffcfe93          	andi	t4,s9,-1025
   12218:	00000793          	li	a5,0
   1221c:	00000713          	li	a4,0
   12220:	0ae103a3          	sb	a4,167(sp)
   12224:	340b4e63          	bltz	s6,12580 <_vfprintf_r+0x840>
   12228:	01b96733          	or	a4,s2,s11
   1222c:	f7fefc93          	andi	s9,t4,-129
   12230:	1a0718e3          	bnez	a4,12be0 <_vfprintf_r+0xea0>
   12234:	740b1263          	bnez	s6,12978 <_vfprintf_r+0xc38>
   12238:	5e0796e3          	bnez	a5,13024 <_vfprintf_r+0x12e4>
   1223c:	001ef913          	andi	s2,t4,1
   12240:	19010d13          	addi	s10,sp,400
   12244:	f6090ee3          	beqz	s2,121c0 <_vfprintf_r+0x480>
   12248:	03000793          	li	a5,48
   1224c:	18f107a3          	sb	a5,399(sp)
   12250:	18f10d13          	addi	s10,sp,399
   12254:	f6dff06f          	j	121c0 <_vfprintf_r+0x480>
   12258:	41aa8933          	sub	s2,s5,s10
   1225c:	c1aa9ce3          	bne	s5,s10,11e74 <_vfprintf_r+0x134>
   12260:	000ac703          	lbu	a4,0(s5)
   12264:	c4dff06f          	j	11eb0 <_vfprintf_r+0x170>
   12268:	00812503          	lw	a0,8(sp)
   1226c:	0c410613          	addi	a2,sp,196
   12270:	00048593          	mv	a1,s1
   12274:	5b8020ef          	jal	1482c <__sprint_r>
   12278:	e00516e3          	bnez	a0,12084 <_vfprintf_r+0x344>
   1227c:	00098a13          	mv	s4,s3
   12280:	c21ff06f          	j	11ea0 <_vfprintf_r+0x160>
   12284:	008cf713          	andi	a4,s9,8
   12288:	600710e3          	bnez	a4,13088 <_vfprintf_r+0x1348>
   1228c:	01412783          	lw	a5,20(sp)
   12290:	09010513          	addi	a0,sp,144
   12294:	01112823          	sw	a7,16(sp)
   12298:	00778c13          	addi	s8,a5,7
   1229c:	ff8c7c13          	andi	s8,s8,-8
   122a0:	000c2583          	lw	a1,0(s8)
   122a4:	004c2603          	lw	a2,4(s8)
   122a8:	008c0793          	addi	a5,s8,8
   122ac:	00f12a23          	sw	a5,20(sp)
   122b0:	3f80e0ef          	jal	206a8 <__extenddftf2>
   122b4:	09012583          	lw	a1,144(sp)
   122b8:	09412603          	lw	a2,148(sp)
   122bc:	09812683          	lw	a3,152(sp)
   122c0:	09c12703          	lw	a4,156(sp)
   122c4:	01012883          	lw	a7,16(sp)
   122c8:	0d010513          	addi	a0,sp,208
   122cc:	01112823          	sw	a7,16(sp)
   122d0:	0ce12e23          	sw	a4,220(sp)
   122d4:	0cb12823          	sw	a1,208(sp)
   122d8:	0cc12a23          	sw	a2,212(sp)
   122dc:	0cd12c23          	sw	a3,216(sp)
   122e0:	240050ef          	jal	17520 <_ldcheck>
   122e4:	0aa12623          	sw	a0,172(sp)
   122e8:	00200713          	li	a4,2
   122ec:	01012883          	lw	a7,16(sp)
   122f0:	00e51463          	bne	a0,a4,122f8 <_vfprintf_r+0x5b8>
   122f4:	3f40106f          	j	136e8 <_vfprintf_r+0x19a8>
   122f8:	00100713          	li	a4,1
   122fc:	00e51463          	bne	a0,a4,12304 <_vfprintf_r+0x5c4>
   12300:	5540106f          	j	13854 <_vfprintf_r+0x1b14>
   12304:	06100713          	li	a4,97
   12308:	00e89463          	bne	a7,a4,12310 <_vfprintf_r+0x5d0>
   1230c:	7a50006f          	j	132b0 <_vfprintf_r+0x1570>
   12310:	04100713          	li	a4,65
   12314:	05800793          	li	a5,88
   12318:	78e88ee3          	beq	a7,a4,132b4 <_vfprintf_r+0x1574>
   1231c:	fff00713          	li	a4,-1
   12320:	00eb1463          	bne	s6,a4,12328 <_vfprintf_r+0x5e8>
   12324:	16c0206f          	j	14490 <_vfprintf_r+0x2750>
   12328:	fdf8f713          	andi	a4,a7,-33
   1232c:	04700693          	li	a3,71
   12330:	00012823          	sw	zero,16(sp)
   12334:	00d71663          	bne	a4,a3,12340 <_vfprintf_r+0x600>
   12338:	000b1463          	bnez	s6,12340 <_vfprintf_r+0x600>
   1233c:	00100b13          	li	s6,1
   12340:	0dc12c03          	lw	s8,220(sp)
   12344:	100ce793          	ori	a5,s9,256
   12348:	04f12423          	sw	a5,72(sp)
   1234c:	02012a23          	sw	zero,52(sp)
   12350:	0d012f83          	lw	t6,208(sp)
   12354:	0d412f03          	lw	t5,212(sp)
   12358:	0d812e83          	lw	t4,216(sp)
   1235c:	000c5a63          	bgez	s8,12370 <_vfprintf_r+0x630>
   12360:	80000737          	lui	a4,0x80000
   12364:	02d00793          	li	a5,45
   12368:	01874c33          	xor	s8,a4,s8
   1236c:	02f12a23          	sw	a5,52(sp)
   12370:	fbf88713          	addi	a4,a7,-65
   12374:	02500693          	li	a3,37
   12378:	78e6e2e3          	bltu	a3,a4,132fc <_vfprintf_r+0x15bc>
   1237c:	0000f697          	auipc	a3,0xf
   12380:	a7868693          	addi	a3,a3,-1416 # 20df4 <_exit+0x444>
   12384:	00271713          	slli	a4,a4,0x2
   12388:	00d70733          	add	a4,a4,a3
   1238c:	00072703          	lw	a4,0(a4) # 80000000 <__BSS_END__+0x7ffdd2d0>
   12390:	00d70733          	add	a4,a4,a3
   12394:	00070067          	jr	a4
   12398:	00812503          	lw	a0,8(sp)
   1239c:	0c410613          	addi	a2,sp,196
   123a0:	00048593          	mv	a1,s1
   123a4:	05112623          	sw	a7,76(sp)
   123a8:	05f12423          	sw	t6,72(sp)
   123ac:	02512a23          	sw	t0,52(sp)
   123b0:	47c020ef          	jal	1482c <__sprint_r>
   123b4:	00051ae3          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   123b8:	0cc12603          	lw	a2,204(sp)
   123bc:	04c12883          	lw	a7,76(sp)
   123c0:	04812f83          	lw	t6,72(sp)
   123c4:	03412283          	lw	t0,52(sp)
   123c8:	00098a13          	mv	s4,s3
   123cc:	be5ff06f          	j	11fb0 <_vfprintf_r+0x270>
   123d0:	0c812903          	lw	s2,200(sp)
   123d4:	01c12783          	lw	a5,28(sp)
   123d8:	00100713          	li	a4,1
   123dc:	01aa2023          	sw	s10,0(s4)
   123e0:	00160c13          	addi	s8,a2,1
   123e4:	00190913          	addi	s2,s2,1
   123e8:	008a0b13          	addi	s6,s4,8
   123ec:	32f75ae3          	bge	a4,a5,12f20 <_vfprintf_r+0x11e0>
   123f0:	00100713          	li	a4,1
   123f4:	00ea2223          	sw	a4,4(s4)
   123f8:	0d812623          	sw	s8,204(sp)
   123fc:	0d212423          	sw	s2,200(sp)
   12400:	00700713          	li	a4,7
   12404:	01275463          	bge	a4,s2,1240c <_vfprintf_r+0x6cc>
   12408:	0bc0106f          	j	134c4 <_vfprintf_r+0x1784>
   1240c:	02812783          	lw	a5,40(sp)
   12410:	02c12703          	lw	a4,44(sp)
   12414:	00190913          	addi	s2,s2,1
   12418:	00fc0c33          	add	s8,s8,a5
   1241c:	00eb2023          	sw	a4,0(s6)
   12420:	00fb2223          	sw	a5,4(s6)
   12424:	0d812623          	sw	s8,204(sp)
   12428:	0d212423          	sw	s2,200(sp)
   1242c:	00700713          	li	a4,7
   12430:	008b0b13          	addi	s6,s6,8
   12434:	01275463          	bge	a4,s2,1243c <_vfprintf_r+0x6fc>
   12438:	0b00106f          	j	134e8 <_vfprintf_r+0x17a8>
   1243c:	0d012703          	lw	a4,208(sp)
   12440:	01c12783          	lw	a5,28(sp)
   12444:	08010593          	addi	a1,sp,128
   12448:	08e12823          	sw	a4,144(sp)
   1244c:	0d412703          	lw	a4,212(sp)
   12450:	09010513          	addi	a0,sp,144
   12454:	08012023          	sw	zero,128(sp)
   12458:	08e12a23          	sw	a4,148(sp)
   1245c:	0d812703          	lw	a4,216(sp)
   12460:	08012223          	sw	zero,132(sp)
   12464:	08012423          	sw	zero,136(sp)
   12468:	08e12c23          	sw	a4,152(sp)
   1246c:	0dc12703          	lw	a4,220(sp)
   12470:	08012623          	sw	zero,140(sp)
   12474:	fff78a13          	addi	s4,a5,-1
   12478:	08e12e23          	sw	a4,156(sp)
   1247c:	7940b0ef          	jal	1dc10 <__eqtf2>
   12480:	2e0500e3          	beqz	a0,12f60 <_vfprintf_r+0x1220>
   12484:	001d0793          	addi	a5,s10,1
   12488:	00190913          	addi	s2,s2,1
   1248c:	014c0c33          	add	s8,s8,s4
   12490:	00fb2023          	sw	a5,0(s6)
   12494:	014b2223          	sw	s4,4(s6)
   12498:	0d812623          	sw	s8,204(sp)
   1249c:	0d212423          	sw	s2,200(sp)
   124a0:	00700793          	li	a5,7
   124a4:	008b0b13          	addi	s6,s6,8
   124a8:	2927cae3          	blt	a5,s2,12f3c <_vfprintf_r+0x11fc>
   124ac:	03c12683          	lw	a3,60(sp)
   124b0:	0b410713          	addi	a4,sp,180
   124b4:	00190793          	addi	a5,s2,1
   124b8:	01868633          	add	a2,a3,s8
   124bc:	00eb2023          	sw	a4,0(s6)
   124c0:	00db2223          	sw	a3,4(s6)
   124c4:	0cc12623          	sw	a2,204(sp)
   124c8:	0cf12423          	sw	a5,200(sp)
   124cc:	00700713          	li	a4,7
   124d0:	008b0a13          	addi	s4,s6,8
   124d4:	b4f758e3          	bge	a4,a5,12024 <_vfprintf_r+0x2e4>
   124d8:	00812503          	lw	a0,8(sp)
   124dc:	0c410613          	addi	a2,sp,196
   124e0:	00048593          	mv	a1,s1
   124e4:	348020ef          	jal	1482c <__sprint_r>
   124e8:	6e051063          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   124ec:	0cc12603          	lw	a2,204(sp)
   124f0:	00098a13          	mv	s4,s3
   124f4:	b31ff06f          	j	12024 <_vfprintf_r+0x2e4>
   124f8:	01412703          	lw	a4,20(sp)
   124fc:	010cf793          	andi	a5,s9,16
   12500:	00072903          	lw	s2,0(a4)
   12504:	00470713          	addi	a4,a4,4
   12508:	00e12a23          	sw	a4,20(sp)
   1250c:	36079ae3          	bnez	a5,13080 <_vfprintf_r+0x1340>
   12510:	040cf793          	andi	a5,s9,64
   12514:	360782e3          	beqz	a5,13078 <_vfprintf_r+0x1338>
   12518:	01091913          	slli	s2,s2,0x10
   1251c:	01095913          	srli	s2,s2,0x10
   12520:	00000d93          	li	s11,0
   12524:	cf1ff06f          	j	12214 <_vfprintf_r+0x4d4>
   12528:	01412703          	lw	a4,20(sp)
   1252c:	010cf793          	andi	a5,s9,16
   12530:	00072903          	lw	s2,0(a4)
   12534:	00470713          	addi	a4,a4,4
   12538:	00e12a23          	sw	a4,20(sp)
   1253c:	320798e3          	bnez	a5,1306c <_vfprintf_r+0x132c>
   12540:	040cf793          	andi	a5,s9,64
   12544:	320780e3          	beqz	a5,13064 <_vfprintf_r+0x1324>
   12548:	01091913          	slli	s2,s2,0x10
   1254c:	41095913          	srai	s2,s2,0x10
   12550:	41f95d93          	srai	s11,s2,0x1f
   12554:	000d8793          	mv	a5,s11
   12558:	c407d2e3          	bgez	a5,1219c <_vfprintf_r+0x45c>
   1255c:	02d00713          	li	a4,45
   12560:	012037b3          	snez	a5,s2
   12564:	41b00db3          	neg	s11,s11
   12568:	0ae103a3          	sb	a4,167(sp)
   1256c:	40fd8db3          	sub	s11,s11,a5
   12570:	41200933          	neg	s2,s2
   12574:	000c8e93          	mv	t4,s9
   12578:	00100793          	li	a5,1
   1257c:	ca0b56e3          	bgez	s6,12228 <_vfprintf_r+0x4e8>
   12580:	00100713          	li	a4,1
   12584:	10e78463          	beq	a5,a4,1268c <_vfprintf_r+0x94c>
   12588:	00200713          	li	a4,2
   1258c:	40e78263          	beq	a5,a4,12990 <_vfprintf_r+0xc50>
   12590:	19010d13          	addi	s10,sp,400
   12594:	01dd9793          	slli	a5,s11,0x1d
   12598:	00797713          	andi	a4,s2,7
   1259c:	00395913          	srli	s2,s2,0x3
   125a0:	03070713          	addi	a4,a4,48
   125a4:	0127e933          	or	s2,a5,s2
   125a8:	003ddd93          	srli	s11,s11,0x3
   125ac:	feed0fa3          	sb	a4,-1(s10)
   125b0:	01b967b3          	or	a5,s2,s11
   125b4:	000d0613          	mv	a2,s10
   125b8:	fffd0d13          	addi	s10,s10,-1
   125bc:	fc079ce3          	bnez	a5,12594 <_vfprintf_r+0x854>
   125c0:	001ef693          	andi	a3,t4,1
   125c4:	40068063          	beqz	a3,129c4 <_vfprintf_r+0xc84>
   125c8:	03000693          	li	a3,48
   125cc:	3ed70c63          	beq	a4,a3,129c4 <_vfprintf_r+0xc84>
   125d0:	ffe60613          	addi	a2,a2,-2
   125d4:	19010793          	addi	a5,sp,400
   125d8:	fedd0fa3          	sb	a3,-1(s10)
   125dc:	40c78933          	sub	s2,a5,a2
   125e0:	000e8c93          	mv	s9,t4
   125e4:	00060d13          	mv	s10,a2
   125e8:	bd9ff06f          	j	121c0 <_vfprintf_r+0x480>
   125ec:	01412703          	lw	a4,20(sp)
   125f0:	0a0103a3          	sb	zero,167(sp)
   125f4:	00100d93          	li	s11,1
   125f8:	00072783          	lw	a5,0(a4)
   125fc:	00470713          	addi	a4,a4,4
   12600:	00e12a23          	sw	a4,20(sp)
   12604:	12f10623          	sb	a5,300(sp)
   12608:	00100913          	li	s2,1
   1260c:	12c10d13          	addi	s10,sp,300
   12610:	935ff06f          	j	11f44 <_vfprintf_r+0x204>
   12614:	01412783          	lw	a5,20(sp)
   12618:	0a0103a3          	sb	zero,167(sp)
   1261c:	0007ad03          	lw	s10,0(a5)
   12620:	00478c13          	addi	s8,a5,4
   12624:	3a0d02e3          	beqz	s10,131c8 <_vfprintf_r+0x1488>
   12628:	000b5463          	bgez	s6,12630 <_vfprintf_r+0x8f0>
   1262c:	1bc0106f          	j	137e8 <_vfprintf_r+0x1aa8>
   12630:	000b0613          	mv	a2,s6
   12634:	00000593          	li	a1,0
   12638:	000d0513          	mv	a0,s10
   1263c:	01112a23          	sw	a7,20(sp)
   12640:	270040ef          	jal	168b0 <memchr>
   12644:	00a12823          	sw	a0,16(sp)
   12648:	01412883          	lw	a7,20(sp)
   1264c:	00051463          	bnez	a0,12654 <_vfprintf_r+0x914>
   12650:	3d90106f          	j	14228 <_vfprintf_r+0x24e8>
   12654:	01012783          	lw	a5,16(sp)
   12658:	0a714703          	lbu	a4,167(sp)
   1265c:	01812a23          	sw	s8,20(sp)
   12660:	41a78933          	sub	s2,a5,s10
   12664:	fff94693          	not	a3,s2
   12668:	41f6d693          	srai	a3,a3,0x1f
   1266c:	00012823          	sw	zero,16(sp)
   12670:	02012223          	sw	zero,36(sp)
   12674:	02012023          	sw	zero,32(sp)
   12678:	00012c23          	sw	zero,24(sp)
   1267c:	00d97db3          	and	s11,s2,a3
   12680:	00000b13          	li	s6,0
   12684:	b60710e3          	bnez	a4,121e4 <_vfprintf_r+0x4a4>
   12688:	8d1ff06f          	j	11f58 <_vfprintf_r+0x218>
   1268c:	680d98e3          	bnez	s11,1351c <_vfprintf_r+0x17dc>
   12690:	00900793          	li	a5,9
   12694:	6927e4e3          	bltu	a5,s2,1351c <_vfprintf_r+0x17dc>
   12698:	03090913          	addi	s2,s2,48
   1269c:	192107a3          	sb	s2,399(sp)
   126a0:	000e8c93          	mv	s9,t4
   126a4:	00100913          	li	s2,1
   126a8:	18f10d13          	addi	s10,sp,399
   126ac:	b15ff06f          	j	121c0 <_vfprintf_r+0x480>
   126b0:	01412783          	lw	a5,20(sp)
   126b4:	0007ab83          	lw	s7,0(a5)
   126b8:	00478793          	addi	a5,a5,4
   126bc:	180bd2e3          	bgez	s7,13040 <_vfprintf_r+0x1300>
   126c0:	41700bb3          	neg	s7,s7
   126c4:	00f12a23          	sw	a5,20(sp)
   126c8:	000ac883          	lbu	a7,0(s5)
   126cc:	004cec93          	ori	s9,s9,4
   126d0:	801ff06f          	j	11ed0 <_vfprintf_r+0x190>
   126d4:	010cee93          	ori	t4,s9,16
   126d8:	020ef793          	andi	a5,t4,32
   126dc:	10078ae3          	beqz	a5,12ff0 <_vfprintf_r+0x12b0>
   126e0:	01412783          	lw	a5,20(sp)
   126e4:	00778c13          	addi	s8,a5,7
   126e8:	ff8c7c13          	andi	s8,s8,-8
   126ec:	008c0793          	addi	a5,s8,8
   126f0:	00f12a23          	sw	a5,20(sp)
   126f4:	000c2903          	lw	s2,0(s8)
   126f8:	004c2d83          	lw	s11,4(s8)
   126fc:	00100793          	li	a5,1
   12700:	b1dff06f          	j	1221c <_vfprintf_r+0x4dc>
   12704:	02b00793          	li	a5,43
   12708:	000ac883          	lbu	a7,0(s5)
   1270c:	0af103a3          	sb	a5,167(sp)
   12710:	fc0ff06f          	j	11ed0 <_vfprintf_r+0x190>
   12714:	01412703          	lw	a4,20(sp)
   12718:	000087b7          	lui	a5,0x8
   1271c:	83078793          	addi	a5,a5,-2000 # 7830 <exit-0x8884>
   12720:	0af11423          	sh	a5,168(sp)
   12724:	00470793          	addi	a5,a4,4
   12728:	00f12a23          	sw	a5,20(sp)
   1272c:	0000e797          	auipc	a5,0xe
   12730:	41c78793          	addi	a5,a5,1052 # 20b48 <_exit+0x198>
   12734:	02f12823          	sw	a5,48(sp)
   12738:	00072903          	lw	s2,0(a4)
   1273c:	00000d93          	li	s11,0
   12740:	002cee93          	ori	t4,s9,2
   12744:	00200793          	li	a5,2
   12748:	07800893          	li	a7,120
   1274c:	ad1ff06f          	j	1221c <_vfprintf_r+0x4dc>
   12750:	020cf793          	andi	a5,s9,32
   12754:	16078ee3          	beqz	a5,130d0 <_vfprintf_r+0x1390>
   12758:	01412783          	lw	a5,20(sp)
   1275c:	00c12683          	lw	a3,12(sp)
   12760:	0007a783          	lw	a5,0(a5)
   12764:	41f6d713          	srai	a4,a3,0x1f
   12768:	00d7a023          	sw	a3,0(a5)
   1276c:	00e7a223          	sw	a4,4(a5)
   12770:	01412783          	lw	a5,20(sp)
   12774:	000a8d13          	mv	s10,s5
   12778:	00478793          	addi	a5,a5,4
   1277c:	00f12a23          	sw	a5,20(sp)
   12780:	8f1ff06f          	j	12070 <_vfprintf_r+0x330>
   12784:	000ac883          	lbu	a7,0(s5)
   12788:	06c00793          	li	a5,108
   1278c:	22f886e3          	beq	a7,a5,131b8 <_vfprintf_r+0x1478>
   12790:	010cec93          	ori	s9,s9,16
   12794:	f3cff06f          	j	11ed0 <_vfprintf_r+0x190>
   12798:	000ac883          	lbu	a7,0(s5)
   1279c:	06800793          	li	a5,104
   127a0:	20f884e3          	beq	a7,a5,131a8 <_vfprintf_r+0x1468>
   127a4:	040cec93          	ori	s9,s9,64
   127a8:	f28ff06f          	j	11ed0 <_vfprintf_r+0x190>
   127ac:	000ac883          	lbu	a7,0(s5)
   127b0:	008cec93          	ori	s9,s9,8
   127b4:	f1cff06f          	j	11ed0 <_vfprintf_r+0x190>
   127b8:	000ac883          	lbu	a7,0(s5)
   127bc:	001cec93          	ori	s9,s9,1
   127c0:	f10ff06f          	j	11ed0 <_vfprintf_r+0x190>
   127c4:	0a714783          	lbu	a5,167(sp)
   127c8:	000ac883          	lbu	a7,0(s5)
   127cc:	f0079263          	bnez	a5,11ed0 <_vfprintf_r+0x190>
   127d0:	02000793          	li	a5,32
   127d4:	0af103a3          	sb	a5,167(sp)
   127d8:	ef8ff06f          	j	11ed0 <_vfprintf_r+0x190>
   127dc:	000ac883          	lbu	a7,0(s5)
   127e0:	080cec93          	ori	s9,s9,128
   127e4:	eecff06f          	j	11ed0 <_vfprintf_r+0x190>
   127e8:	000ac883          	lbu	a7,0(s5)
   127ec:	02a00793          	li	a5,42
   127f0:	001a8693          	addi	a3,s5,1
   127f4:	00f89463          	bne	a7,a5,127fc <_vfprintf_r+0xabc>
   127f8:	5e50106f          	j	145dc <_vfprintf_r+0x289c>
   127fc:	fd088793          	addi	a5,a7,-48
   12800:	00900713          	li	a4,9
   12804:	00000b13          	li	s6,0
   12808:	00900613          	li	a2,9
   1280c:	02f76463          	bltu	a4,a5,12834 <_vfprintf_r+0xaf4>
   12810:	0006c883          	lbu	a7,0(a3)
   12814:	002b1713          	slli	a4,s6,0x2
   12818:	01670b33          	add	s6,a4,s6
   1281c:	001b1b13          	slli	s6,s6,0x1
   12820:	00fb0b33          	add	s6,s6,a5
   12824:	fd088793          	addi	a5,a7,-48
   12828:	00168693          	addi	a3,a3,1
   1282c:	fef672e3          	bgeu	a2,a5,12810 <_vfprintf_r+0xad0>
   12830:	0c0b4ce3          	bltz	s6,13108 <_vfprintf_r+0x13c8>
   12834:	00068a93          	mv	s5,a3
   12838:	e9cff06f          	j	11ed4 <_vfprintf_r+0x194>
   1283c:	06500713          	li	a4,101
   12840:	b91758e3          	bge	a4,a7,123d0 <_vfprintf_r+0x690>
   12844:	0d012703          	lw	a4,208(sp)
   12848:	08010593          	addi	a1,sp,128
   1284c:	09010513          	addi	a0,sp,144
   12850:	08e12823          	sw	a4,144(sp)
   12854:	0d412703          	lw	a4,212(sp)
   12858:	02c12a23          	sw	a2,52(sp)
   1285c:	08012023          	sw	zero,128(sp)
   12860:	08e12a23          	sw	a4,148(sp)
   12864:	0d812703          	lw	a4,216(sp)
   12868:	08012223          	sw	zero,132(sp)
   1286c:	08012423          	sw	zero,136(sp)
   12870:	08e12c23          	sw	a4,152(sp)
   12874:	0dc12703          	lw	a4,220(sp)
   12878:	08012623          	sw	zero,140(sp)
   1287c:	08e12e23          	sw	a4,156(sp)
   12880:	3900b0ef          	jal	1dc10 <__eqtf2>
   12884:	03412603          	lw	a2,52(sp)
   12888:	54051663          	bnez	a0,12dd4 <_vfprintf_r+0x1094>
   1288c:	0c812783          	lw	a5,200(sp)
   12890:	0000e717          	auipc	a4,0xe
   12894:	2e870713          	addi	a4,a4,744 # 20b78 <_exit+0x1c8>
   12898:	00ea2023          	sw	a4,0(s4)
   1289c:	00160613          	addi	a2,a2,1
   128a0:	00100713          	li	a4,1
   128a4:	00178793          	addi	a5,a5,1
   128a8:	00ea2223          	sw	a4,4(s4)
   128ac:	0cc12623          	sw	a2,204(sp)
   128b0:	0cf12423          	sw	a5,200(sp)
   128b4:	00700713          	li	a4,7
   128b8:	008a0a13          	addi	s4,s4,8
   128bc:	5ef74ce3          	blt	a4,a5,136b4 <_vfprintf_r+0x1974>
   128c0:	0ac12783          	lw	a5,172(sp)
   128c4:	01c12703          	lw	a4,28(sp)
   128c8:	76e7d463          	bge	a5,a4,13030 <_vfprintf_r+0x12f0>
   128cc:	02c12783          	lw	a5,44(sp)
   128d0:	02812703          	lw	a4,40(sp)
   128d4:	008a0a13          	addi	s4,s4,8
   128d8:	fefa2c23          	sw	a5,-8(s4)
   128dc:	0c812783          	lw	a5,200(sp)
   128e0:	00e60633          	add	a2,a2,a4
   128e4:	feea2e23          	sw	a4,-4(s4)
   128e8:	00178793          	addi	a5,a5,1
   128ec:	0cc12623          	sw	a2,204(sp)
   128f0:	0cf12423          	sw	a5,200(sp)
   128f4:	00700713          	li	a4,7
   128f8:	08f748e3          	blt	a4,a5,13188 <_vfprintf_r+0x1448>
   128fc:	01c12783          	lw	a5,28(sp)
   12900:	fff78913          	addi	s2,a5,-1
   12904:	f3205063          	blez	s2,12024 <_vfprintf_r+0x2e4>
   12908:	0000e817          	auipc	a6,0xe
   1290c:	58480813          	addi	a6,a6,1412 # 20e8c <zeroes.0>
   12910:	01000713          	li	a4,16
   12914:	0c812783          	lw	a5,200(sp)
   12918:	01000b13          	li	s6,16
   1291c:	00700c13          	li	s8,7
   12920:	00080d13          	mv	s10,a6
   12924:	01274863          	blt	a4,s2,12934 <_vfprintf_r+0xbf4>
   12928:	5b10006f          	j	136d8 <_vfprintf_r+0x1998>
   1292c:	ff090913          	addi	s2,s2,-16
   12930:	5b2b52e3          	bge	s6,s2,136d4 <_vfprintf_r+0x1994>
   12934:	01060613          	addi	a2,a2,16
   12938:	00178793          	addi	a5,a5,1
   1293c:	01aa2023          	sw	s10,0(s4)
   12940:	016a2223          	sw	s6,4(s4)
   12944:	0cc12623          	sw	a2,204(sp)
   12948:	0cf12423          	sw	a5,200(sp)
   1294c:	008a0a13          	addi	s4,s4,8
   12950:	fcfc5ee3          	bge	s8,a5,1292c <_vfprintf_r+0xbec>
   12954:	00812503          	lw	a0,8(sp)
   12958:	0c410613          	addi	a2,sp,196
   1295c:	00048593          	mv	a1,s1
   12960:	6cd010ef          	jal	1482c <__sprint_r>
   12964:	26051263          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   12968:	0cc12603          	lw	a2,204(sp)
   1296c:	0c812783          	lw	a5,200(sp)
   12970:	00098a13          	mv	s4,s3
   12974:	fb9ff06f          	j	1292c <_vfprintf_r+0xbec>
   12978:	00100713          	li	a4,1
   1297c:	00e79463          	bne	a5,a4,12984 <_vfprintf_r+0xc44>
   12980:	1510106f          	j	142d0 <_vfprintf_r+0x2590>
   12984:	00200713          	li	a4,2
   12988:	000c8e93          	mv	t4,s9
   1298c:	c0e792e3          	bne	a5,a4,12590 <_vfprintf_r+0x850>
   12990:	03012683          	lw	a3,48(sp)
   12994:	19010d13          	addi	s10,sp,400
   12998:	00f97793          	andi	a5,s2,15
   1299c:	00f687b3          	add	a5,a3,a5
   129a0:	0007c703          	lbu	a4,0(a5)
   129a4:	00495913          	srli	s2,s2,0x4
   129a8:	01cd9793          	slli	a5,s11,0x1c
   129ac:	0127e933          	or	s2,a5,s2
   129b0:	004ddd93          	srli	s11,s11,0x4
   129b4:	feed0fa3          	sb	a4,-1(s10)
   129b8:	01b967b3          	or	a5,s2,s11
   129bc:	fffd0d13          	addi	s10,s10,-1
   129c0:	fc079ce3          	bnez	a5,12998 <_vfprintf_r+0xc58>
   129c4:	19010793          	addi	a5,sp,400
   129c8:	41a78933          	sub	s2,a5,s10
   129cc:	000e8c93          	mv	s9,t4
   129d0:	ff0ff06f          	j	121c0 <_vfprintf_r+0x480>
   129d4:	41bb8c33          	sub	s8,s7,s11
   129d8:	e1805a63          	blez	s8,11fec <_vfprintf_r+0x2ac>
   129dc:	01000513          	li	a0,16
   129e0:	0c812583          	lw	a1,200(sp)
   129e4:	0000e817          	auipc	a6,0xe
   129e8:	4a880813          	addi	a6,a6,1192 # 20e8c <zeroes.0>
   129ec:	09855c63          	bge	a0,s8,12a84 <_vfprintf_r+0xd44>
   129f0:	00090713          	mv	a4,s2
   129f4:	000a0793          	mv	a5,s4
   129f8:	000c0913          	mv	s2,s8
   129fc:	01000e93          	li	t4,16
   12a00:	00700f93          	li	t6,7
   12a04:	03112a23          	sw	a7,52(sp)
   12a08:	00080a13          	mv	s4,a6
   12a0c:	00070c13          	mv	s8,a4
   12a10:	00c0006f          	j	12a1c <_vfprintf_r+0xcdc>
   12a14:	ff090913          	addi	s2,s2,-16
   12a18:	052eda63          	bge	t4,s2,12a6c <_vfprintf_r+0xd2c>
   12a1c:	01060613          	addi	a2,a2,16
   12a20:	00158593          	addi	a1,a1,1 # 2001 <exit-0xe0b3>
   12a24:	0147a023          	sw	s4,0(a5)
   12a28:	01d7a223          	sw	t4,4(a5)
   12a2c:	0cc12623          	sw	a2,204(sp)
   12a30:	0cb12423          	sw	a1,200(sp)
   12a34:	00878793          	addi	a5,a5,8
   12a38:	fcbfdee3          	bge	t6,a1,12a14 <_vfprintf_r+0xcd4>
   12a3c:	00812503          	lw	a0,8(sp)
   12a40:	0c410613          	addi	a2,sp,196
   12a44:	00048593          	mv	a1,s1
   12a48:	5e5010ef          	jal	1482c <__sprint_r>
   12a4c:	16051e63          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   12a50:	01000e93          	li	t4,16
   12a54:	ff090913          	addi	s2,s2,-16
   12a58:	0cc12603          	lw	a2,204(sp)
   12a5c:	0c812583          	lw	a1,200(sp)
   12a60:	00098793          	mv	a5,s3
   12a64:	00700f93          	li	t6,7
   12a68:	fb2ecae3          	blt	t4,s2,12a1c <_vfprintf_r+0xcdc>
   12a6c:	03412883          	lw	a7,52(sp)
   12a70:	000c0713          	mv	a4,s8
   12a74:	000a0813          	mv	a6,s4
   12a78:	00090c13          	mv	s8,s2
   12a7c:	00078a13          	mv	s4,a5
   12a80:	00070913          	mv	s2,a4
   12a84:	01860633          	add	a2,a2,s8
   12a88:	00158593          	addi	a1,a1,1
   12a8c:	010a2023          	sw	a6,0(s4)
   12a90:	018a2223          	sw	s8,4(s4)
   12a94:	0cc12623          	sw	a2,204(sp)
   12a98:	0cb12423          	sw	a1,200(sp)
   12a9c:	00700713          	li	a4,7
   12aa0:	008a0a13          	addi	s4,s4,8
   12aa4:	d4b75463          	bge	a4,a1,11fec <_vfprintf_r+0x2ac>
   12aa8:	00812503          	lw	a0,8(sp)
   12aac:	0c410613          	addi	a2,sp,196
   12ab0:	00048593          	mv	a1,s1
   12ab4:	03112a23          	sw	a7,52(sp)
   12ab8:	575010ef          	jal	1482c <__sprint_r>
   12abc:	10051663          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   12ac0:	0cc12603          	lw	a2,204(sp)
   12ac4:	03412883          	lw	a7,52(sp)
   12ac8:	00098a13          	mv	s4,s3
   12acc:	d20ff06f          	j	11fec <_vfprintf_r+0x2ac>
   12ad0:	0c812583          	lw	a1,200(sp)
   12ad4:	0000e817          	auipc	a6,0xe
   12ad8:	3b880813          	addi	a6,a6,952 # 20e8c <zeroes.0>
   12adc:	09645663          	bge	s0,s6,12b68 <_vfprintf_r+0xe28>
   12ae0:	000a0793          	mv	a5,s4
   12ae4:	00700c13          	li	s8,7
   12ae8:	00090a13          	mv	s4,s2
   12aec:	03112a23          	sw	a7,52(sp)
   12af0:	000b0913          	mv	s2,s6
   12af4:	00080b13          	mv	s6,a6
   12af8:	00c0006f          	j	12b04 <_vfprintf_r+0xdc4>
   12afc:	ff090913          	addi	s2,s2,-16
   12b00:	05245a63          	bge	s0,s2,12b54 <_vfprintf_r+0xe14>
   12b04:	01060613          	addi	a2,a2,16
   12b08:	00158593          	addi	a1,a1,1
   12b0c:	0000e717          	auipc	a4,0xe
   12b10:	38070713          	addi	a4,a4,896 # 20e8c <zeroes.0>
   12b14:	00e7a023          	sw	a4,0(a5)
   12b18:	0087a223          	sw	s0,4(a5)
   12b1c:	0cc12623          	sw	a2,204(sp)
   12b20:	0cb12423          	sw	a1,200(sp)
   12b24:	00878793          	addi	a5,a5,8
   12b28:	fcbc5ae3          	bge	s8,a1,12afc <_vfprintf_r+0xdbc>
   12b2c:	00812503          	lw	a0,8(sp)
   12b30:	0c410613          	addi	a2,sp,196
   12b34:	00048593          	mv	a1,s1
   12b38:	4f5010ef          	jal	1482c <__sprint_r>
   12b3c:	08051663          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   12b40:	ff090913          	addi	s2,s2,-16
   12b44:	0cc12603          	lw	a2,204(sp)
   12b48:	0c812583          	lw	a1,200(sp)
   12b4c:	00098793          	mv	a5,s3
   12b50:	fb244ae3          	blt	s0,s2,12b04 <_vfprintf_r+0xdc4>
   12b54:	03412883          	lw	a7,52(sp)
   12b58:	000b0813          	mv	a6,s6
   12b5c:	00090b13          	mv	s6,s2
   12b60:	000a0913          	mv	s2,s4
   12b64:	00078a13          	mv	s4,a5
   12b68:	01660633          	add	a2,a2,s6
   12b6c:	00158593          	addi	a1,a1,1
   12b70:	010a2023          	sw	a6,0(s4)
   12b74:	016a2223          	sw	s6,4(s4)
   12b78:	0cc12623          	sw	a2,204(sp)
   12b7c:	0cb12423          	sw	a1,200(sp)
   12b80:	00700713          	li	a4,7
   12b84:	008a0a13          	addi	s4,s4,8
   12b88:	c6b75663          	bge	a4,a1,11ff4 <_vfprintf_r+0x2b4>
   12b8c:	00812503          	lw	a0,8(sp)
   12b90:	0c410613          	addi	a2,sp,196
   12b94:	00048593          	mv	a1,s1
   12b98:	03112a23          	sw	a7,52(sp)
   12b9c:	491010ef          	jal	1482c <__sprint_r>
   12ba0:	02051463          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   12ba4:	0cc12603          	lw	a2,204(sp)
   12ba8:	03412883          	lw	a7,52(sp)
   12bac:	00098a13          	mv	s4,s3
   12bb0:	c44ff06f          	j	11ff4 <_vfprintf_r+0x2b4>
   12bb4:	00812503          	lw	a0,8(sp)
   12bb8:	0c410613          	addi	a2,sp,196
   12bbc:	00048593          	mv	a1,s1
   12bc0:	46d010ef          	jal	1482c <__sprint_r>
   12bc4:	c8050663          	beqz	a0,12050 <_vfprintf_r+0x310>
   12bc8:	01012383          	lw	t2,16(sp)
   12bcc:	ca038c63          	beqz	t2,12084 <_vfprintf_r+0x344>
   12bd0:	00812503          	lw	a0,8(sp)
   12bd4:	00038593          	mv	a1,t2
   12bd8:	e94fe0ef          	jal	1126c <_free_r>
   12bdc:	ca8ff06f          	j	12084 <_vfprintf_r+0x344>
   12be0:	000c8e93          	mv	t4,s9
   12be4:	99dff06f          	j	12580 <_vfprintf_r+0x840>
   12be8:	01000513          	li	a0,16
   12bec:	0c812583          	lw	a1,200(sp)
   12bf0:	0000ec17          	auipc	s8,0xe
   12bf4:	2acc0c13          	addi	s8,s8,684 # 20e9c <blanks.1>
   12bf8:	0ae55063          	bge	a0,a4,12c98 <_vfprintf_r+0xf58>
   12bfc:	000a0793          	mv	a5,s4
   12c00:	01000813          	li	a6,16
   12c04:	000c0a13          	mv	s4,s8
   12c08:	00700393          	li	t2,7
   12c0c:	00090c13          	mv	s8,s2
   12c10:	02512a23          	sw	t0,52(sp)
   12c14:	05f12423          	sw	t6,72(sp)
   12c18:	05112623          	sw	a7,76(sp)
   12c1c:	00070913          	mv	s2,a4
   12c20:	00c0006f          	j	12c2c <_vfprintf_r+0xeec>
   12c24:	ff090913          	addi	s2,s2,-16
   12c28:	05285a63          	bge	a6,s2,12c7c <_vfprintf_r+0xf3c>
   12c2c:	01060613          	addi	a2,a2,16
   12c30:	00158593          	addi	a1,a1,1
   12c34:	0147a023          	sw	s4,0(a5)
   12c38:	0107a223          	sw	a6,4(a5)
   12c3c:	0cc12623          	sw	a2,204(sp)
   12c40:	0cb12423          	sw	a1,200(sp)
   12c44:	00878793          	addi	a5,a5,8
   12c48:	fcb3dee3          	bge	t2,a1,12c24 <_vfprintf_r+0xee4>
   12c4c:	00812503          	lw	a0,8(sp)
   12c50:	0c410613          	addi	a2,sp,196
   12c54:	00048593          	mv	a1,s1
   12c58:	3d5010ef          	jal	1482c <__sprint_r>
   12c5c:	f60516e3          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   12c60:	01000813          	li	a6,16
   12c64:	ff090913          	addi	s2,s2,-16
   12c68:	0cc12603          	lw	a2,204(sp)
   12c6c:	0c812583          	lw	a1,200(sp)
   12c70:	00098793          	mv	a5,s3
   12c74:	00700393          	li	t2,7
   12c78:	fb284ae3          	blt	a6,s2,12c2c <_vfprintf_r+0xeec>
   12c7c:	03412283          	lw	t0,52(sp)
   12c80:	04812f83          	lw	t6,72(sp)
   12c84:	04c12883          	lw	a7,76(sp)
   12c88:	00090713          	mv	a4,s2
   12c8c:	000c0913          	mv	s2,s8
   12c90:	000a0c13          	mv	s8,s4
   12c94:	00078a13          	mv	s4,a5
   12c98:	00e60633          	add	a2,a2,a4
   12c9c:	00158593          	addi	a1,a1,1
   12ca0:	00ea2223          	sw	a4,4(s4)
   12ca4:	018a2023          	sw	s8,0(s4)
   12ca8:	0cc12623          	sw	a2,204(sp)
   12cac:	0cb12423          	sw	a1,200(sp)
   12cb0:	00700713          	li	a4,7
   12cb4:	008a0a13          	addi	s4,s4,8
   12cb8:	acb75063          	bge	a4,a1,11f78 <_vfprintf_r+0x238>
   12cbc:	00812503          	lw	a0,8(sp)
   12cc0:	0c410613          	addi	a2,sp,196
   12cc4:	00048593          	mv	a1,s1
   12cc8:	05112623          	sw	a7,76(sp)
   12ccc:	05f12423          	sw	t6,72(sp)
   12cd0:	02512a23          	sw	t0,52(sp)
   12cd4:	359010ef          	jal	1482c <__sprint_r>
   12cd8:	ee0518e3          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   12cdc:	0cc12603          	lw	a2,204(sp)
   12ce0:	04c12883          	lw	a7,76(sp)
   12ce4:	04812f83          	lw	t6,72(sp)
   12ce8:	03412283          	lw	t0,52(sp)
   12cec:	00098a13          	mv	s4,s3
   12cf0:	a88ff06f          	j	11f78 <_vfprintf_r+0x238>
   12cf4:	00812503          	lw	a0,8(sp)
   12cf8:	0c410613          	addi	a2,sp,196
   12cfc:	00048593          	mv	a1,s1
   12d00:	05112423          	sw	a7,72(sp)
   12d04:	03f12a23          	sw	t6,52(sp)
   12d08:	325010ef          	jal	1482c <__sprint_r>
   12d0c:	ea051ee3          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   12d10:	0cc12603          	lw	a2,204(sp)
   12d14:	04812883          	lw	a7,72(sp)
   12d18:	03412f83          	lw	t6,52(sp)
   12d1c:	00098a13          	mv	s4,s3
   12d20:	ac4ff06f          	j	11fe4 <_vfprintf_r+0x2a4>
   12d24:	01000713          	li	a4,16
   12d28:	0c812783          	lw	a5,200(sp)
   12d2c:	0000ec17          	auipc	s8,0xe
   12d30:	170c0c13          	addi	s8,s8,368 # 20e9c <blanks.1>
   12d34:	07275263          	bge	a4,s2,12d98 <_vfprintf_r+0x1058>
   12d38:	00812d03          	lw	s10,8(sp)
   12d3c:	01000b13          	li	s6,16
   12d40:	00700c93          	li	s9,7
   12d44:	00c0006f          	j	12d50 <_vfprintf_r+0x1010>
   12d48:	ff090913          	addi	s2,s2,-16
   12d4c:	052b5663          	bge	s6,s2,12d98 <_vfprintf_r+0x1058>
   12d50:	01060613          	addi	a2,a2,16
   12d54:	00178793          	addi	a5,a5,1
   12d58:	018a2023          	sw	s8,0(s4)
   12d5c:	016a2223          	sw	s6,4(s4)
   12d60:	0cc12623          	sw	a2,204(sp)
   12d64:	0cf12423          	sw	a5,200(sp)
   12d68:	008a0a13          	addi	s4,s4,8
   12d6c:	fcfcdee3          	bge	s9,a5,12d48 <_vfprintf_r+0x1008>
   12d70:	0c410613          	addi	a2,sp,196
   12d74:	00048593          	mv	a1,s1
   12d78:	000d0513          	mv	a0,s10
   12d7c:	2b1010ef          	jal	1482c <__sprint_r>
   12d80:	e40514e3          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   12d84:	ff090913          	addi	s2,s2,-16
   12d88:	0cc12603          	lw	a2,204(sp)
   12d8c:	0c812783          	lw	a5,200(sp)
   12d90:	00098a13          	mv	s4,s3
   12d94:	fb2b4ee3          	blt	s6,s2,12d50 <_vfprintf_r+0x1010>
   12d98:	01260633          	add	a2,a2,s2
   12d9c:	00178793          	addi	a5,a5,1
   12da0:	018a2023          	sw	s8,0(s4)
   12da4:	012a2223          	sw	s2,4(s4)
   12da8:	0cc12623          	sw	a2,204(sp)
   12dac:	0cf12423          	sw	a5,200(sp)
   12db0:	00700713          	li	a4,7
   12db4:	a8f75063          	bge	a4,a5,12034 <_vfprintf_r+0x2f4>
   12db8:	00812503          	lw	a0,8(sp)
   12dbc:	0c410613          	addi	a2,sp,196
   12dc0:	00048593          	mv	a1,s1
   12dc4:	269010ef          	jal	1482c <__sprint_r>
   12dc8:	e00510e3          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   12dcc:	0cc12603          	lw	a2,204(sp)
   12dd0:	a64ff06f          	j	12034 <_vfprintf_r+0x2f4>
   12dd4:	0ac12583          	lw	a1,172(sp)
   12dd8:	04b058e3          	blez	a1,13628 <_vfprintf_r+0x18e8>
   12ddc:	01812783          	lw	a5,24(sp)
   12de0:	01c12703          	lw	a4,28(sp)
   12de4:	00078b13          	mv	s6,a5
   12de8:	30f74a63          	blt	a4,a5,130fc <_vfprintf_r+0x13bc>
   12dec:	03605663          	blez	s6,12e18 <_vfprintf_r+0x10d8>
   12df0:	0c812703          	lw	a4,200(sp)
   12df4:	01660633          	add	a2,a2,s6
   12df8:	01aa2023          	sw	s10,0(s4)
   12dfc:	00170713          	addi	a4,a4,1
   12e00:	016a2223          	sw	s6,4(s4)
   12e04:	0cc12623          	sw	a2,204(sp)
   12e08:	0ce12423          	sw	a4,200(sp)
   12e0c:	00700593          	li	a1,7
   12e10:	008a0a13          	addi	s4,s4,8
   12e14:	5ae5c0e3          	blt	a1,a4,13bb4 <_vfprintf_r+0x1e74>
   12e18:	fffb4713          	not	a4,s6
   12e1c:	01812783          	lw	a5,24(sp)
   12e20:	41f75713          	srai	a4,a4,0x1f
   12e24:	00eb7b33          	and	s6,s6,a4
   12e28:	41678b33          	sub	s6,a5,s6
   12e2c:	3b604e63          	bgtz	s6,131e8 <_vfprintf_r+0x14a8>
   12e30:	01812783          	lw	a5,24(sp)
   12e34:	400cf713          	andi	a4,s9,1024
   12e38:	00fd0c33          	add	s8,s10,a5
   12e3c:	280716e3          	bnez	a4,138c8 <_vfprintf_r+0x1b88>
   12e40:	0ac12583          	lw	a1,172(sp)
   12e44:	01c12783          	lw	a5,28(sp)
   12e48:	40f5ca63          	blt	a1,a5,1325c <_vfprintf_r+0x151c>
   12e4c:	001cf713          	andi	a4,s9,1
   12e50:	40071663          	bnez	a4,1325c <_vfprintf_r+0x151c>
   12e54:	01c12703          	lw	a4,28(sp)
   12e58:	00ed07b3          	add	a5,s10,a4
   12e5c:	40b705b3          	sub	a1,a4,a1
   12e60:	41878b33          	sub	s6,a5,s8
   12e64:	0165d463          	bge	a1,s6,12e6c <_vfprintf_r+0x112c>
   12e68:	00058b13          	mv	s6,a1
   12e6c:	03605863          	blez	s6,12e9c <_vfprintf_r+0x115c>
   12e70:	0c812703          	lw	a4,200(sp)
   12e74:	01660633          	add	a2,a2,s6
   12e78:	018a2023          	sw	s8,0(s4)
   12e7c:	00170713          	addi	a4,a4,1
   12e80:	016a2223          	sw	s6,4(s4)
   12e84:	0cc12623          	sw	a2,204(sp)
   12e88:	0ce12423          	sw	a4,200(sp)
   12e8c:	00700793          	li	a5,7
   12e90:	008a0a13          	addi	s4,s4,8
   12e94:	00e7d463          	bge	a5,a4,12e9c <_vfprintf_r+0x115c>
   12e98:	3cc0106f          	j	14264 <_vfprintf_r+0x2524>
   12e9c:	fffb4713          	not	a4,s6
   12ea0:	41f75713          	srai	a4,a4,0x1f
   12ea4:	00eb77b3          	and	a5,s6,a4
   12ea8:	40f58933          	sub	s2,a1,a5
   12eac:	97205c63          	blez	s2,12024 <_vfprintf_r+0x2e4>
   12eb0:	0000e817          	auipc	a6,0xe
   12eb4:	fdc80813          	addi	a6,a6,-36 # 20e8c <zeroes.0>
   12eb8:	01000713          	li	a4,16
   12ebc:	0c812783          	lw	a5,200(sp)
   12ec0:	01000b13          	li	s6,16
   12ec4:	00700c13          	li	s8,7
   12ec8:	00080d13          	mv	s10,a6
   12ecc:	01274863          	blt	a4,s2,12edc <_vfprintf_r+0x119c>
   12ed0:	0090006f          	j	136d8 <_vfprintf_r+0x1998>
   12ed4:	ff090913          	addi	s2,s2,-16
   12ed8:	7f2b5e63          	bge	s6,s2,136d4 <_vfprintf_r+0x1994>
   12edc:	01060613          	addi	a2,a2,16
   12ee0:	00178793          	addi	a5,a5,1
   12ee4:	01aa2023          	sw	s10,0(s4)
   12ee8:	016a2223          	sw	s6,4(s4)
   12eec:	0cc12623          	sw	a2,204(sp)
   12ef0:	0cf12423          	sw	a5,200(sp)
   12ef4:	008a0a13          	addi	s4,s4,8
   12ef8:	fcfc5ee3          	bge	s8,a5,12ed4 <_vfprintf_r+0x1194>
   12efc:	00812503          	lw	a0,8(sp)
   12f00:	0c410613          	addi	a2,sp,196
   12f04:	00048593          	mv	a1,s1
   12f08:	125010ef          	jal	1482c <__sprint_r>
   12f0c:	ca051ee3          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   12f10:	0cc12603          	lw	a2,204(sp)
   12f14:	0c812783          	lw	a5,200(sp)
   12f18:	00098a13          	mv	s4,s3
   12f1c:	fb9ff06f          	j	12ed4 <_vfprintf_r+0x1194>
   12f20:	001cf593          	andi	a1,s9,1
   12f24:	cc059663          	bnez	a1,123f0 <_vfprintf_r+0x6b0>
   12f28:	00ea2223          	sw	a4,4(s4)
   12f2c:	0d812623          	sw	s8,204(sp)
   12f30:	0d212423          	sw	s2,200(sp)
   12f34:	00700793          	li	a5,7
   12f38:	d727da63          	bge	a5,s2,124ac <_vfprintf_r+0x76c>
   12f3c:	00812503          	lw	a0,8(sp)
   12f40:	0c410613          	addi	a2,sp,196
   12f44:	00048593          	mv	a1,s1
   12f48:	0e5010ef          	jal	1482c <__sprint_r>
   12f4c:	c6051ee3          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   12f50:	0cc12c03          	lw	s8,204(sp)
   12f54:	0c812903          	lw	s2,200(sp)
   12f58:	00098b13          	mv	s6,s3
   12f5c:	d50ff06f          	j	124ac <_vfprintf_r+0x76c>
   12f60:	01c12703          	lw	a4,28(sp)
   12f64:	00100793          	li	a5,1
   12f68:	d4e7d263          	bge	a5,a4,124ac <_vfprintf_r+0x76c>
   12f6c:	01100793          	li	a5,17
   12f70:	0000e817          	auipc	a6,0xe
   12f74:	f1c80813          	addi	a6,a6,-228 # 20e8c <zeroes.0>
   12f78:	0ce7d6e3          	bge	a5,a4,13844 <_vfprintf_r+0x1b04>
   12f7c:	00048713          	mv	a4,s1
   12f80:	01512c23          	sw	s5,24(sp)
   12f84:	000a0493          	mv	s1,s4
   12f88:	01000793          	li	a5,16
   12f8c:	00700d13          	li	s10,7
   12f90:	00080a93          	mv	s5,a6
   12f94:	00070a13          	mv	s4,a4
   12f98:	00c0006f          	j	12fa4 <_vfprintf_r+0x1264>
   12f9c:	ff048493          	addi	s1,s1,-16
   12fa0:	0897d8e3          	bge	a5,s1,13830 <_vfprintf_r+0x1af0>
   12fa4:	010c0c13          	addi	s8,s8,16
   12fa8:	00190913          	addi	s2,s2,1
   12fac:	015b2023          	sw	s5,0(s6)
   12fb0:	00fb2223          	sw	a5,4(s6)
   12fb4:	0d812623          	sw	s8,204(sp)
   12fb8:	0d212423          	sw	s2,200(sp)
   12fbc:	008b0b13          	addi	s6,s6,8
   12fc0:	fd2d5ee3          	bge	s10,s2,12f9c <_vfprintf_r+0x125c>
   12fc4:	00812503          	lw	a0,8(sp)
   12fc8:	0c410613          	addi	a2,sp,196
   12fcc:	000a0593          	mv	a1,s4
   12fd0:	05d010ef          	jal	1482c <__sprint_r>
   12fd4:	00050463          	beqz	a0,12fdc <_vfprintf_r+0x129c>
   12fd8:	2ec0106f          	j	142c4 <_vfprintf_r+0x2584>
   12fdc:	0cc12c03          	lw	s8,204(sp)
   12fe0:	0c812903          	lw	s2,200(sp)
   12fe4:	00098b13          	mv	s6,s3
   12fe8:	01000793          	li	a5,16
   12fec:	fb1ff06f          	j	12f9c <_vfprintf_r+0x125c>
   12ff0:	01412703          	lw	a4,20(sp)
   12ff4:	010ef793          	andi	a5,t4,16
   12ff8:	00072903          	lw	s2,0(a4)
   12ffc:	00470713          	addi	a4,a4,4
   13000:	00e12a23          	sw	a4,20(sp)
   13004:	04079a63          	bnez	a5,13058 <_vfprintf_r+0x1318>
   13008:	040ef793          	andi	a5,t4,64
   1300c:	04078063          	beqz	a5,1304c <_vfprintf_r+0x130c>
   13010:	01091913          	slli	s2,s2,0x10
   13014:	01095913          	srli	s2,s2,0x10
   13018:	00000d93          	li	s11,0
   1301c:	00100793          	li	a5,1
   13020:	9fcff06f          	j	1221c <_vfprintf_r+0x4dc>
   13024:	00000913          	li	s2,0
   13028:	19010d13          	addi	s10,sp,400
   1302c:	994ff06f          	j	121c0 <_vfprintf_r+0x480>
   13030:	001cf793          	andi	a5,s9,1
   13034:	00079463          	bnez	a5,1303c <_vfprintf_r+0x12fc>
   13038:	fedfe06f          	j	12024 <_vfprintf_r+0x2e4>
   1303c:	891ff06f          	j	128cc <_vfprintf_r+0xb8c>
   13040:	000ac883          	lbu	a7,0(s5)
   13044:	00f12a23          	sw	a5,20(sp)
   13048:	e89fe06f          	j	11ed0 <_vfprintf_r+0x190>
   1304c:	200ef793          	andi	a5,t4,512
   13050:	00078463          	beqz	a5,13058 <_vfprintf_r+0x1318>
   13054:	0ff97913          	zext.b	s2,s2
   13058:	00000d93          	li	s11,0
   1305c:	00100793          	li	a5,1
   13060:	9bcff06f          	j	1221c <_vfprintf_r+0x4dc>
   13064:	200cf793          	andi	a5,s9,512
   13068:	380792e3          	bnez	a5,13bec <_vfprintf_r+0x1eac>
   1306c:	41f95d93          	srai	s11,s2,0x1f
   13070:	000d8793          	mv	a5,s11
   13074:	924ff06f          	j	12198 <_vfprintf_r+0x458>
   13078:	200cf793          	andi	a5,s9,512
   1307c:	360792e3          	bnez	a5,13be0 <_vfprintf_r+0x1ea0>
   13080:	00000d93          	li	s11,0
   13084:	990ff06f          	j	12214 <_vfprintf_r+0x4d4>
   13088:	01412783          	lw	a5,20(sp)
   1308c:	0007a703          	lw	a4,0(a5)
   13090:	00478793          	addi	a5,a5,4
   13094:	00f12a23          	sw	a5,20(sp)
   13098:	00072583          	lw	a1,0(a4)
   1309c:	00472603          	lw	a2,4(a4)
   130a0:	00872683          	lw	a3,8(a4)
   130a4:	00c72703          	lw	a4,12(a4)
   130a8:	a20ff06f          	j	122c8 <_vfprintf_r+0x588>
   130ac:	03812783          	lw	a5,56(sp)
   130b0:	000ac883          	lbu	a7,0(s5)
   130b4:	00079463          	bnez	a5,130bc <_vfprintf_r+0x137c>
   130b8:	e19fe06f          	j	11ed0 <_vfprintf_r+0x190>
   130bc:	0007c783          	lbu	a5,0(a5)
   130c0:	00079463          	bnez	a5,130c8 <_vfprintf_r+0x1388>
   130c4:	e0dfe06f          	j	11ed0 <_vfprintf_r+0x190>
   130c8:	400cec93          	ori	s9,s9,1024
   130cc:	e05fe06f          	j	11ed0 <_vfprintf_r+0x190>
   130d0:	010cf793          	andi	a5,s9,16
   130d4:	6a079063          	bnez	a5,13774 <_vfprintf_r+0x1a34>
   130d8:	040cf793          	andi	a5,s9,64
   130dc:	320798e3          	bnez	a5,13c0c <_vfprintf_r+0x1ecc>
   130e0:	200cfe13          	andi	t3,s9,512
   130e4:	680e0863          	beqz	t3,13774 <_vfprintf_r+0x1a34>
   130e8:	01412783          	lw	a5,20(sp)
   130ec:	00c12703          	lw	a4,12(sp)
   130f0:	0007a783          	lw	a5,0(a5)
   130f4:	00e78023          	sb	a4,0(a5)
   130f8:	e78ff06f          	j	12770 <_vfprintf_r+0xa30>
   130fc:	00070b13          	mv	s6,a4
   13100:	cf6048e3          	bgtz	s6,12df0 <_vfprintf_r+0x10b0>
   13104:	d15ff06f          	j	12e18 <_vfprintf_r+0x10d8>
   13108:	fff00b13          	li	s6,-1
   1310c:	00068a93          	mv	s5,a3
   13110:	dc5fe06f          	j	11ed4 <_vfprintf_r+0x194>
   13114:	0000e797          	auipc	a5,0xe
   13118:	a4878793          	addi	a5,a5,-1464 # 20b5c <_exit+0x1ac>
   1311c:	02f12823          	sw	a5,48(sp)
   13120:	020cf793          	andi	a5,s9,32
   13124:	36078863          	beqz	a5,13494 <_vfprintf_r+0x1754>
   13128:	01412783          	lw	a5,20(sp)
   1312c:	00778c13          	addi	s8,a5,7
   13130:	ff8c7c13          	andi	s8,s8,-8
   13134:	000c2903          	lw	s2,0(s8)
   13138:	004c2d83          	lw	s11,4(s8)
   1313c:	008c0793          	addi	a5,s8,8
   13140:	00f12a23          	sw	a5,20(sp)
   13144:	001cf793          	andi	a5,s9,1
   13148:	00078e63          	beqz	a5,13164 <_vfprintf_r+0x1424>
   1314c:	01b967b3          	or	a5,s2,s11
   13150:	00078a63          	beqz	a5,13164 <_vfprintf_r+0x1424>
   13154:	03000793          	li	a5,48
   13158:	0af10423          	sb	a5,168(sp)
   1315c:	0b1104a3          	sb	a7,169(sp)
   13160:	002cec93          	ori	s9,s9,2
   13164:	bffcfe93          	andi	t4,s9,-1025
   13168:	00200793          	li	a5,2
   1316c:	8b0ff06f          	j	1221c <_vfprintf_r+0x4dc>
   13170:	000c8e93          	mv	t4,s9
   13174:	d64ff06f          	j	126d8 <_vfprintf_r+0x998>
   13178:	0000e797          	auipc	a5,0xe
   1317c:	9d078793          	addi	a5,a5,-1584 # 20b48 <_exit+0x198>
   13180:	02f12823          	sw	a5,48(sp)
   13184:	f9dff06f          	j	13120 <_vfprintf_r+0x13e0>
   13188:	00812503          	lw	a0,8(sp)
   1318c:	0c410613          	addi	a2,sp,196
   13190:	00048593          	mv	a1,s1
   13194:	698010ef          	jal	1482c <__sprint_r>
   13198:	a20518e3          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   1319c:	0cc12603          	lw	a2,204(sp)
   131a0:	00098a13          	mv	s4,s3
   131a4:	f58ff06f          	j	128fc <_vfprintf_r+0xbbc>
   131a8:	001ac883          	lbu	a7,1(s5)
   131ac:	200cec93          	ori	s9,s9,512
   131b0:	001a8a93          	addi	s5,s5,1
   131b4:	d1dfe06f          	j	11ed0 <_vfprintf_r+0x190>
   131b8:	001ac883          	lbu	a7,1(s5)
   131bc:	020cec93          	ori	s9,s9,32
   131c0:	001a8a93          	addi	s5,s5,1
   131c4:	d0dfe06f          	j	11ed0 <_vfprintf_r+0x190>
   131c8:	00600793          	li	a5,6
   131cc:	000b0913          	mv	s2,s6
   131d0:	2167e4e3          	bltu	a5,s6,13bd8 <_vfprintf_r+0x1e98>
   131d4:	00090d93          	mv	s11,s2
   131d8:	01812a23          	sw	s8,20(sp)
   131dc:	0000ed17          	auipc	s10,0xe
   131e0:	994d0d13          	addi	s10,s10,-1644 # 20b70 <_exit+0x1c0>
   131e4:	d61fe06f          	j	11f44 <_vfprintf_r+0x204>
   131e8:	01000593          	li	a1,16
   131ec:	0c812703          	lw	a4,200(sp)
   131f0:	0000e817          	auipc	a6,0xe
   131f4:	c9c80813          	addi	a6,a6,-868 # 20e8c <zeroes.0>
   131f8:	6965d663          	bge	a1,s6,13884 <_vfprintf_r+0x1b44>
   131fc:	000a0793          	mv	a5,s4
   13200:	01000c13          	li	s8,16
   13204:	00700913          	li	s2,7
   13208:	00080a13          	mv	s4,a6
   1320c:	00c0006f          	j	13218 <_vfprintf_r+0x14d8>
   13210:	ff0b0b13          	addi	s6,s6,-16
   13214:	676c5463          	bge	s8,s6,1387c <_vfprintf_r+0x1b3c>
   13218:	01060613          	addi	a2,a2,16
   1321c:	00170713          	addi	a4,a4,1
   13220:	0147a023          	sw	s4,0(a5)
   13224:	0187a223          	sw	s8,4(a5)
   13228:	0cc12623          	sw	a2,204(sp)
   1322c:	0ce12423          	sw	a4,200(sp)
   13230:	00878793          	addi	a5,a5,8
   13234:	fce95ee3          	bge	s2,a4,13210 <_vfprintf_r+0x14d0>
   13238:	00812503          	lw	a0,8(sp)
   1323c:	0c410613          	addi	a2,sp,196
   13240:	00048593          	mv	a1,s1
   13244:	5e8010ef          	jal	1482c <__sprint_r>
   13248:	980510e3          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   1324c:	0cc12603          	lw	a2,204(sp)
   13250:	0c812703          	lw	a4,200(sp)
   13254:	00098793          	mv	a5,s3
   13258:	fb9ff06f          	j	13210 <_vfprintf_r+0x14d0>
   1325c:	02c12703          	lw	a4,44(sp)
   13260:	02812783          	lw	a5,40(sp)
   13264:	00700513          	li	a0,7
   13268:	00ea2023          	sw	a4,0(s4)
   1326c:	0c812703          	lw	a4,200(sp)
   13270:	00f60633          	add	a2,a2,a5
   13274:	00fa2223          	sw	a5,4(s4)
   13278:	00170713          	addi	a4,a4,1
   1327c:	0cc12623          	sw	a2,204(sp)
   13280:	0ce12423          	sw	a4,200(sp)
   13284:	008a0a13          	addi	s4,s4,8
   13288:	bce556e3          	bge	a0,a4,12e54 <_vfprintf_r+0x1114>
   1328c:	00812503          	lw	a0,8(sp)
   13290:	0c410613          	addi	a2,sp,196
   13294:	00048593          	mv	a1,s1
   13298:	594010ef          	jal	1482c <__sprint_r>
   1329c:	920516e3          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   132a0:	0ac12583          	lw	a1,172(sp)
   132a4:	0cc12603          	lw	a2,204(sp)
   132a8:	00098a13          	mv	s4,s3
   132ac:	ba9ff06f          	j	12e54 <_vfprintf_r+0x1114>
   132b0:	07800793          	li	a5,120
   132b4:	03000713          	li	a4,48
   132b8:	0ae10423          	sb	a4,168(sp)
   132bc:	0af104a3          	sb	a5,169(sp)
   132c0:	06300713          	li	a4,99
   132c4:	00012823          	sw	zero,16(sp)
   132c8:	002cec93          	ori	s9,s9,2
   132cc:	12c10d13          	addi	s10,sp,300
   132d0:	87675863          	bge	a4,s6,12340 <_vfprintf_r+0x600>
   132d4:	00812503          	lw	a0,8(sp)
   132d8:	001b0593          	addi	a1,s6,1
   132dc:	01112823          	sw	a7,16(sp)
   132e0:	a90fe0ef          	jal	11570 <_malloc_r>
   132e4:	01012883          	lw	a7,16(sp)
   132e8:	00050d13          	mv	s10,a0
   132ec:	00051463          	bnez	a0,132f4 <_vfprintf_r+0x15b4>
   132f0:	3280106f          	j	14618 <_vfprintf_r+0x28d8>
   132f4:	00a12823          	sw	a0,16(sp)
   132f8:	848ff06f          	j	12340 <_vfprintf_r+0x600>
   132fc:	00812503          	lw	a0,8(sp)
   13300:	09010913          	addi	s2,sp,144
   13304:	0ac10713          	addi	a4,sp,172
   13308:	0bc10813          	addi	a6,sp,188
   1330c:	0b010793          	addi	a5,sp,176
   13310:	000b0693          	mv	a3,s6
   13314:	00200613          	li	a2,2
   13318:	00090593          	mv	a1,s2
   1331c:	03112223          	sw	a7,36(sp)
   13320:	09f12823          	sw	t6,144(sp)
   13324:	03f12023          	sw	t6,32(sp)
   13328:	09e12a23          	sw	t5,148(sp)
   1332c:	01e12e23          	sw	t5,28(sp)
   13330:	09d12c23          	sw	t4,152(sp)
   13334:	01d12c23          	sw	t4,24(sp)
   13338:	09812e23          	sw	s8,156(sp)
   1333c:	705030ef          	jal	17240 <_ldtoa_r>
   13340:	001cf713          	andi	a4,s9,1
   13344:	01812e83          	lw	t4,24(sp)
   13348:	01c12f03          	lw	t5,28(sp)
   1334c:	02012f83          	lw	t6,32(sp)
   13350:	02412883          	lw	a7,36(sp)
   13354:	00050d13          	mv	s10,a0
   13358:	10071ce3          	bnez	a4,13c70 <_vfprintf_r+0x1f30>
   1335c:	0ac12783          	lw	a5,172(sp)
   13360:	00f12c23          	sw	a5,24(sp)
   13364:	0bc12783          	lw	a5,188(sp)
   13368:	40a787b3          	sub	a5,a5,a0
   1336c:	00f12e23          	sw	a5,28(sp)
   13370:	01812783          	lw	a5,24(sp)
   13374:	ffd00713          	li	a4,-3
   13378:	00e7c463          	blt	a5,a4,13380 <_vfprintf_r+0x1640>
   1337c:	60fb56e3          	bge	s6,a5,14188 <_vfprintf_r+0x2448>
   13380:	01812783          	lw	a5,24(sp)
   13384:	ffe88893          	addi	a7,a7,-2
   13388:	fff78713          	addi	a4,a5,-1
   1338c:	0ae12623          	sw	a4,172(sp)
   13390:	0ff8f693          	zext.b	a3,a7
   13394:	00000613          	li	a2,0
   13398:	0ad10a23          	sb	a3,180(sp)
   1339c:	02b00693          	li	a3,43
   133a0:	00075a63          	bgez	a4,133b4 <_vfprintf_r+0x1674>
   133a4:	01812783          	lw	a5,24(sp)
   133a8:	00100713          	li	a4,1
   133ac:	02d00693          	li	a3,45
   133b0:	40f70733          	sub	a4,a4,a5
   133b4:	0ad10aa3          	sb	a3,181(sp)
   133b8:	00900693          	li	a3,9
   133bc:	00e6c463          	blt	a3,a4,133c4 <_vfprintf_r+0x1684>
   133c0:	0900106f          	j	14450 <_vfprintf_r+0x2710>
   133c4:	0c310813          	addi	a6,sp,195
   133c8:	00080e93          	mv	t4,a6
   133cc:	00a00613          	li	a2,10
   133d0:	06300f13          	li	t5,99
   133d4:	02c767b3          	rem	a5,a4,a2
   133d8:	000e8513          	mv	a0,t4
   133dc:	00070693          	mv	a3,a4
   133e0:	fffe8e93          	addi	t4,t4,-1
   133e4:	03078793          	addi	a5,a5,48
   133e8:	fef50fa3          	sb	a5,-1(a0)
   133ec:	02c74733          	div	a4,a4,a2
   133f0:	fedf42e3          	blt	t5,a3,133d4 <_vfprintf_r+0x1694>
   133f4:	03070713          	addi	a4,a4,48
   133f8:	ffe50693          	addi	a3,a0,-2
   133fc:	feee8fa3          	sb	a4,-1(t4)
   13400:	0106e463          	bltu	a3,a6,13408 <_vfprintf_r+0x16c8>
   13404:	1cc0106f          	j	145d0 <_vfprintf_r+0x2890>
   13408:	0b610613          	addi	a2,sp,182
   1340c:	0006c783          	lbu	a5,0(a3)
   13410:	00168693          	addi	a3,a3,1
   13414:	00160613          	addi	a2,a2,1
   13418:	fef60fa3          	sb	a5,-1(a2)
   1341c:	ff0698e3          	bne	a3,a6,1340c <_vfprintf_r+0x16cc>
   13420:	19010793          	addi	a5,sp,400
   13424:	40a78733          	sub	a4,a5,a0
   13428:	f3770793          	addi	a5,a4,-201
   1342c:	02f12e23          	sw	a5,60(sp)
   13430:	01c12783          	lw	a5,28(sp)
   13434:	03c12683          	lw	a3,60(sp)
   13438:	00100713          	li	a4,1
   1343c:	00d78933          	add	s2,a5,a3
   13440:	00f74463          	blt	a4,a5,13448 <_vfprintf_r+0x1708>
   13444:	03c0106f          	j	14480 <_vfprintf_r+0x2740>
   13448:	02812783          	lw	a5,40(sp)
   1344c:	00f90933          	add	s2,s2,a5
   13450:	fff94693          	not	a3,s2
   13454:	bffcfe13          	andi	t3,s9,-1025
   13458:	41f6d693          	srai	a3,a3,0x1f
   1345c:	100e6793          	ori	a5,t3,256
   13460:	04f12423          	sw	a5,72(sp)
   13464:	00d97db3          	and	s11,s2,a3
   13468:	02012223          	sw	zero,36(sp)
   1346c:	02012023          	sw	zero,32(sp)
   13470:	00012c23          	sw	zero,24(sp)
   13474:	03412783          	lw	a5,52(sp)
   13478:	4e078ce3          	beqz	a5,14170 <_vfprintf_r+0x2430>
   1347c:	02d00713          	li	a4,45
   13480:	04812c83          	lw	s9,72(sp)
   13484:	0ae103a3          	sb	a4,167(sp)
   13488:	00000b13          	li	s6,0
   1348c:	001d8d93          	addi	s11,s11,1
   13490:	ac9fe06f          	j	11f58 <_vfprintf_r+0x218>
   13494:	01412703          	lw	a4,20(sp)
   13498:	010cf793          	andi	a5,s9,16
   1349c:	00072903          	lw	s2,0(a4)
   134a0:	00470713          	addi	a4,a4,4
   134a4:	00e12a23          	sw	a4,20(sp)
   134a8:	06079663          	bnez	a5,13514 <_vfprintf_r+0x17d4>
   134ac:	040cf793          	andi	a5,s9,64
   134b0:	04078e63          	beqz	a5,1350c <_vfprintf_r+0x17cc>
   134b4:	01091913          	slli	s2,s2,0x10
   134b8:	01095913          	srli	s2,s2,0x10
   134bc:	00000d93          	li	s11,0
   134c0:	c85ff06f          	j	13144 <_vfprintf_r+0x1404>
   134c4:	00812503          	lw	a0,8(sp)
   134c8:	0c410613          	addi	a2,sp,196
   134cc:	00048593          	mv	a1,s1
   134d0:	35c010ef          	jal	1482c <__sprint_r>
   134d4:	ee051a63          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   134d8:	0cc12c03          	lw	s8,204(sp)
   134dc:	0c812903          	lw	s2,200(sp)
   134e0:	00098b13          	mv	s6,s3
   134e4:	f29fe06f          	j	1240c <_vfprintf_r+0x6cc>
   134e8:	00812503          	lw	a0,8(sp)
   134ec:	0c410613          	addi	a2,sp,196
   134f0:	00048593          	mv	a1,s1
   134f4:	338010ef          	jal	1482c <__sprint_r>
   134f8:	ec051863          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   134fc:	0cc12c03          	lw	s8,204(sp)
   13500:	0c812903          	lw	s2,200(sp)
   13504:	00098b13          	mv	s6,s3
   13508:	f35fe06f          	j	1243c <_vfprintf_r+0x6fc>
   1350c:	200cf793          	andi	a5,s9,512
   13510:	6e079863          	bnez	a5,13c00 <_vfprintf_r+0x1ec0>
   13514:	00000d93          	li	s11,0
   13518:	c2dff06f          	j	13144 <_vfprintf_r+0x1404>
   1351c:	ccccd837          	lui	a6,0xccccd
   13520:	ccccdcb7          	lui	s9,0xccccd
   13524:	03812703          	lw	a4,56(sp)
   13528:	400eff13          	andi	t5,t4,1024
   1352c:	00000613          	li	a2,0
   13530:	19010593          	addi	a1,sp,400
   13534:	00500e13          	li	t3,5
   13538:	ccd80813          	addi	a6,a6,-819 # cccccccd <__BSS_END__+0xccca9f9d>
   1353c:	cccc8c93          	addi	s9,s9,-820 # cccccccc <__BSS_END__+0xccca9f9c>
   13540:	0ff00c13          	li	s8,255
   13544:	0540006f          	j	13598 <_vfprintf_r+0x1858>
   13548:	012d37b3          	sltu	a5,s10,s2
   1354c:	00fd07b3          	add	a5,s10,a5
   13550:	03c7f7b3          	remu	a5,a5,t3
   13554:	40f907b3          	sub	a5,s2,a5
   13558:	00f935b3          	sltu	a1,s2,a5
   1355c:	40bd85b3          	sub	a1,s11,a1
   13560:	03978333          	mul	t1,a5,s9
   13564:	030585b3          	mul	a1,a1,a6
   13568:	0307b533          	mulhu	a0,a5,a6
   1356c:	006585b3          	add	a1,a1,t1
   13570:	030787b3          	mul	a5,a5,a6
   13574:	00a585b3          	add	a1,a1,a0
   13578:	01f59513          	slli	a0,a1,0x1f
   1357c:	0015d593          	srli	a1,a1,0x1
   13580:	0017d793          	srli	a5,a5,0x1
   13584:	00f567b3          	or	a5,a0,a5
   13588:	480d82e3          	beqz	s11,1420c <_vfprintf_r+0x24cc>
   1358c:	00058d93          	mv	s11,a1
   13590:	00078913          	mv	s2,a5
   13594:	00068593          	mv	a1,a3
   13598:	01b90d33          	add	s10,s2,s11
   1359c:	012d37b3          	sltu	a5,s10,s2
   135a0:	00fd07b3          	add	a5,s10,a5
   135a4:	03c7f7b3          	remu	a5,a5,t3
   135a8:	fff58693          	addi	a3,a1,-1
   135ac:	00160613          	addi	a2,a2,1
   135b0:	40f907b3          	sub	a5,s2,a5
   135b4:	00f93533          	sltu	a0,s2,a5
   135b8:	40ad8533          	sub	a0,s11,a0
   135bc:	0307b333          	mulhu	t1,a5,a6
   135c0:	03050533          	mul	a0,a0,a6
   135c4:	030787b3          	mul	a5,a5,a6
   135c8:	00650533          	add	a0,a0,t1
   135cc:	01f51513          	slli	a0,a0,0x1f
   135d0:	0017d793          	srli	a5,a5,0x1
   135d4:	00f567b3          	or	a5,a0,a5
   135d8:	00279513          	slli	a0,a5,0x2
   135dc:	00f507b3          	add	a5,a0,a5
   135e0:	00179793          	slli	a5,a5,0x1
   135e4:	40f907b3          	sub	a5,s2,a5
   135e8:	03078793          	addi	a5,a5,48
   135ec:	fef58fa3          	sb	a5,-1(a1)
   135f0:	f40f0ce3          	beqz	t5,13548 <_vfprintf_r+0x1808>
   135f4:	00074783          	lbu	a5,0(a4)
   135f8:	f4f618e3          	bne	a2,a5,13548 <_vfprintf_r+0x1808>
   135fc:	f58606e3          	beq	a2,s8,13548 <_vfprintf_r+0x1808>
   13600:	4c0d9c63          	bnez	s11,13ad8 <_vfprintf_r+0x1d98>
   13604:	00900793          	li	a5,9
   13608:	4d27e863          	bltu	a5,s2,13ad8 <_vfprintf_r+0x1d98>
   1360c:	00068d13          	mv	s10,a3
   13610:	19010793          	addi	a5,sp,400
   13614:	00c12e23          	sw	a2,28(sp)
   13618:	02e12c23          	sw	a4,56(sp)
   1361c:	41a78933          	sub	s2,a5,s10
   13620:	000e8c93          	mv	s9,t4
   13624:	b9dfe06f          	j	121c0 <_vfprintf_r+0x480>
   13628:	0c812703          	lw	a4,200(sp)
   1362c:	0000d517          	auipc	a0,0xd
   13630:	54c50513          	addi	a0,a0,1356 # 20b78 <_exit+0x1c8>
   13634:	00aa2023          	sw	a0,0(s4)
   13638:	00160613          	addi	a2,a2,1
   1363c:	00100513          	li	a0,1
   13640:	00170713          	addi	a4,a4,1
   13644:	00aa2223          	sw	a0,4(s4)
   13648:	0cc12623          	sw	a2,204(sp)
   1364c:	0ce12423          	sw	a4,200(sp)
   13650:	00700513          	li	a0,7
   13654:	008a0a13          	addi	s4,s4,8
   13658:	52e54a63          	blt	a0,a4,13b8c <_vfprintf_r+0x1e4c>
   1365c:	12059663          	bnez	a1,13788 <_vfprintf_r+0x1a48>
   13660:	01c12783          	lw	a5,28(sp)
   13664:	001cf713          	andi	a4,s9,1
   13668:	00f76733          	or	a4,a4,a5
   1366c:	00071463          	bnez	a4,13674 <_vfprintf_r+0x1934>
   13670:	9b5fe06f          	j	12024 <_vfprintf_r+0x2e4>
   13674:	02c12703          	lw	a4,44(sp)
   13678:	02812783          	lw	a5,40(sp)
   1367c:	00700593          	li	a1,7
   13680:	00ea2023          	sw	a4,0(s4)
   13684:	0c812703          	lw	a4,200(sp)
   13688:	00c78633          	add	a2,a5,a2
   1368c:	00fa2223          	sw	a5,4(s4)
   13690:	00170713          	addi	a4,a4,1
   13694:	0cc12623          	sw	a2,204(sp)
   13698:	0ce12423          	sw	a4,200(sp)
   1369c:	5ae5c463          	blt	a1,a4,13c44 <_vfprintf_r+0x1f04>
   136a0:	008a0a13          	addi	s4,s4,8
   136a4:	1180006f          	j	137bc <_vfprintf_r+0x1a7c>
   136a8:	00812503          	lw	a0,8(sp)
   136ac:	93cfd0ef          	jal	107e8 <__sinit>
   136b0:	ef0fe06f          	j	11da0 <_vfprintf_r+0x60>
   136b4:	00812503          	lw	a0,8(sp)
   136b8:	0c410613          	addi	a2,sp,196
   136bc:	00048593          	mv	a1,s1
   136c0:	16c010ef          	jal	1482c <__sprint_r>
   136c4:	d0051263          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   136c8:	0cc12603          	lw	a2,204(sp)
   136cc:	00098a13          	mv	s4,s3
   136d0:	9f0ff06f          	j	128c0 <_vfprintf_r+0xb80>
   136d4:	000d0813          	mv	a6,s10
   136d8:	01260633          	add	a2,a2,s2
   136dc:	00178793          	addi	a5,a5,1
   136e0:	010a2023          	sw	a6,0(s4)
   136e4:	929fe06f          	j	1200c <_vfprintf_r+0x2cc>
   136e8:	0d012783          	lw	a5,208(sp)
   136ec:	08010593          	addi	a1,sp,128
   136f0:	09010513          	addi	a0,sp,144
   136f4:	08f12823          	sw	a5,144(sp)
   136f8:	0d412783          	lw	a5,212(sp)
   136fc:	08012023          	sw	zero,128(sp)
   13700:	08012223          	sw	zero,132(sp)
   13704:	08f12a23          	sw	a5,148(sp)
   13708:	0d812783          	lw	a5,216(sp)
   1370c:	08012423          	sw	zero,136(sp)
   13710:	08012623          	sw	zero,140(sp)
   13714:	08f12c23          	sw	a5,152(sp)
   13718:	0dc12783          	lw	a5,220(sp)
   1371c:	08f12e23          	sw	a5,156(sp)
   13720:	6ec0a0ef          	jal	1de0c <__letf2>
   13724:	01012883          	lw	a7,16(sp)
   13728:	2e0548e3          	bltz	a0,14218 <_vfprintf_r+0x24d8>
   1372c:	0a714703          	lbu	a4,167(sp)
   13730:	04700693          	li	a3,71
   13734:	0000dd17          	auipc	s10,0xd
   13738:	408d0d13          	addi	s10,s10,1032 # 20b3c <_exit+0x18c>
   1373c:	0116c663          	blt	a3,a7,13748 <_vfprintf_r+0x1a08>
   13740:	0000dd17          	auipc	s10,0xd
   13744:	3f8d0d13          	addi	s10,s10,1016 # 20b38 <_exit+0x188>
   13748:	00012823          	sw	zero,16(sp)
   1374c:	02012223          	sw	zero,36(sp)
   13750:	02012023          	sw	zero,32(sp)
   13754:	00012c23          	sw	zero,24(sp)
   13758:	f7fcfc93          	andi	s9,s9,-129
   1375c:	00300d93          	li	s11,3
   13760:	00300913          	li	s2,3
   13764:	00000b13          	li	s6,0
   13768:	00070463          	beqz	a4,13770 <_vfprintf_r+0x1a30>
   1376c:	a79fe06f          	j	121e4 <_vfprintf_r+0x4a4>
   13770:	fe8fe06f          	j	11f58 <_vfprintf_r+0x218>
   13774:	01412783          	lw	a5,20(sp)
   13778:	00c12703          	lw	a4,12(sp)
   1377c:	0007a783          	lw	a5,0(a5)
   13780:	00e7a023          	sw	a4,0(a5)
   13784:	fedfe06f          	j	12770 <_vfprintf_r+0xa30>
   13788:	02c12703          	lw	a4,44(sp)
   1378c:	02812783          	lw	a5,40(sp)
   13790:	00700513          	li	a0,7
   13794:	00ea2023          	sw	a4,0(s4)
   13798:	0c812703          	lw	a4,200(sp)
   1379c:	00c78633          	add	a2,a5,a2
   137a0:	00fa2223          	sw	a5,4(s4)
   137a4:	00170713          	addi	a4,a4,1
   137a8:	0cc12623          	sw	a2,204(sp)
   137ac:	0ce12423          	sw	a4,200(sp)
   137b0:	008a0a13          	addi	s4,s4,8
   137b4:	48e54863          	blt	a0,a4,13c44 <_vfprintf_r+0x1f04>
   137b8:	3205c0e3          	bltz	a1,142d8 <_vfprintf_r+0x2598>
   137bc:	01c12783          	lw	a5,28(sp)
   137c0:	00170713          	addi	a4,a4,1
   137c4:	01aa2023          	sw	s10,0(s4)
   137c8:	00c78633          	add	a2,a5,a2
   137cc:	00fa2223          	sw	a5,4(s4)
   137d0:	0cc12623          	sw	a2,204(sp)
   137d4:	0ce12423          	sw	a4,200(sp)
   137d8:	00700793          	li	a5,7
   137dc:	00e7c463          	blt	a5,a4,137e4 <_vfprintf_r+0x1aa4>
   137e0:	841fe06f          	j	12020 <_vfprintf_r+0x2e0>
   137e4:	cf5fe06f          	j	124d8 <_vfprintf_r+0x798>
   137e8:	000d0513          	mv	a0,s10
   137ec:	03112a23          	sw	a7,52(sp)
   137f0:	f6cfd0ef          	jal	10f5c <strlen>
   137f4:	0a714703          	lbu	a4,167(sp)
   137f8:	fff54693          	not	a3,a0
   137fc:	41f6d693          	srai	a3,a3,0x1f
   13800:	01812a23          	sw	s8,20(sp)
   13804:	00012823          	sw	zero,16(sp)
   13808:	02012223          	sw	zero,36(sp)
   1380c:	02012023          	sw	zero,32(sp)
   13810:	00012c23          	sw	zero,24(sp)
   13814:	03412883          	lw	a7,52(sp)
   13818:	00050913          	mv	s2,a0
   1381c:	00d57db3          	and	s11,a0,a3
   13820:	00000b13          	li	s6,0
   13824:	00070463          	beqz	a4,1382c <_vfprintf_r+0x1aec>
   13828:	9bdfe06f          	j	121e4 <_vfprintf_r+0x4a4>
   1382c:	f2cfe06f          	j	11f58 <_vfprintf_r+0x218>
   13830:	000a8813          	mv	a6,s5
   13834:	01812a83          	lw	s5,24(sp)
   13838:	000a0793          	mv	a5,s4
   1383c:	00048a13          	mv	s4,s1
   13840:	00078493          	mv	s1,a5
   13844:	014c0c33          	add	s8,s8,s4
   13848:	00190913          	addi	s2,s2,1
   1384c:	010b2023          	sw	a6,0(s6)
   13850:	c45fe06f          	j	12494 <_vfprintf_r+0x754>
   13854:	0dc12783          	lw	a5,220(sp)
   13858:	3c07ce63          	bltz	a5,13c34 <_vfprintf_r+0x1ef4>
   1385c:	0a714703          	lbu	a4,167(sp)
   13860:	04700693          	li	a3,71
   13864:	0000dd17          	auipc	s10,0xd
   13868:	2e0d0d13          	addi	s10,s10,736 # 20b44 <_exit+0x194>
   1386c:	ed16cee3          	blt	a3,a7,13748 <_vfprintf_r+0x1a08>
   13870:	0000dd17          	auipc	s10,0xd
   13874:	2d0d0d13          	addi	s10,s10,720 # 20b40 <_exit+0x190>
   13878:	ed1ff06f          	j	13748 <_vfprintf_r+0x1a08>
   1387c:	000a0813          	mv	a6,s4
   13880:	00078a13          	mv	s4,a5
   13884:	01660633          	add	a2,a2,s6
   13888:	00170713          	addi	a4,a4,1
   1388c:	010a2023          	sw	a6,0(s4)
   13890:	016a2223          	sw	s6,4(s4)
   13894:	0cc12623          	sw	a2,204(sp)
   13898:	0ce12423          	sw	a4,200(sp)
   1389c:	00700593          	li	a1,7
   138a0:	008a0a13          	addi	s4,s4,8
   138a4:	d8e5d663          	bge	a1,a4,12e30 <_vfprintf_r+0x10f0>
   138a8:	00812503          	lw	a0,8(sp)
   138ac:	0c410613          	addi	a2,sp,196
   138b0:	00048593          	mv	a1,s1
   138b4:	779000ef          	jal	1482c <__sprint_r>
   138b8:	b0051863          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   138bc:	0cc12603          	lw	a2,204(sp)
   138c0:	00098a13          	mv	s4,s3
   138c4:	d6cff06f          	j	12e30 <_vfprintf_r+0x10f0>
   138c8:	01c12783          	lw	a5,28(sp)
   138cc:	03512a23          	sw	s5,52(sp)
   138d0:	02012a83          	lw	s5,32(sp)
   138d4:	00fd07b3          	add	a5,s10,a5
   138d8:	05912423          	sw	s9,72(sp)
   138dc:	05712623          	sw	s7,76(sp)
   138e0:	03b12023          	sw	s11,32(sp)
   138e4:	02412d83          	lw	s11,36(sp)
   138e8:	03a12223          	sw	s10,36(sp)
   138ec:	03812c83          	lw	s9,56(sp)
   138f0:	000c0d13          	mv	s10,s8
   138f4:	00812903          	lw	s2,8(sp)
   138f8:	04412c03          	lw	s8,68(sp)
   138fc:	00700813          	li	a6,7
   13900:	01000713          	li	a4,16
   13904:	0000db17          	auipc	s6,0xd
   13908:	588b0b13          	addi	s6,s6,1416 # 20e8c <zeroes.0>
   1390c:	000a0593          	mv	a1,s4
   13910:	00078b93          	mv	s7,a5
   13914:	09505663          	blez	s5,139a0 <_vfprintf_r+0x1c60>
   13918:	17b05063          	blez	s11,13a78 <_vfprintf_r+0x1d38>
   1391c:	fffd8d93          	addi	s11,s11,-1
   13920:	04012783          	lw	a5,64(sp)
   13924:	01860633          	add	a2,a2,s8
   13928:	0185a223          	sw	s8,4(a1)
   1392c:	00f5a023          	sw	a5,0(a1)
   13930:	0c812783          	lw	a5,200(sp)
   13934:	0cc12623          	sw	a2,204(sp)
   13938:	00858593          	addi	a1,a1,8
   1393c:	00178793          	addi	a5,a5,1
   13940:	0cf12423          	sw	a5,200(sp)
   13944:	14f84063          	blt	a6,a5,13a84 <_vfprintf_r+0x1d44>
   13948:	000cc683          	lbu	a3,0(s9)
   1394c:	41ab8a33          	sub	s4,s7,s10
   13950:	0146d463          	bge	a3,s4,13958 <_vfprintf_r+0x1c18>
   13954:	00068a13          	mv	s4,a3
   13958:	03405663          	blez	s4,13984 <_vfprintf_r+0x1c44>
   1395c:	0c812683          	lw	a3,200(sp)
   13960:	01460633          	add	a2,a2,s4
   13964:	01a5a023          	sw	s10,0(a1)
   13968:	00168693          	addi	a3,a3,1
   1396c:	0145a223          	sw	s4,4(a1)
   13970:	0cc12623          	sw	a2,204(sp)
   13974:	0cd12423          	sw	a3,200(sp)
   13978:	12d84a63          	blt	a6,a3,13aac <_vfprintf_r+0x1d6c>
   1397c:	000cc683          	lbu	a3,0(s9)
   13980:	00858593          	addi	a1,a1,8
   13984:	fffa4513          	not	a0,s4
   13988:	41f55513          	srai	a0,a0,0x1f
   1398c:	00aa77b3          	and	a5,s4,a0
   13990:	40f68a33          	sub	s4,a3,a5
   13994:	05404263          	bgtz	s4,139d8 <_vfprintf_r+0x1c98>
   13998:	00dd0d33          	add	s10,s10,a3
   1399c:	f7504ee3          	bgtz	s5,13918 <_vfprintf_r+0x1bd8>
   139a0:	f7b04ee3          	bgtz	s11,1391c <_vfprintf_r+0x1bdc>
   139a4:	01c12783          	lw	a5,28(sp)
   139a8:	000d0c13          	mv	s8,s10
   139ac:	02412d03          	lw	s10,36(sp)
   139b0:	03912c23          	sw	s9,56(sp)
   139b4:	03412a83          	lw	s5,52(sp)
   139b8:	00fd0733          	add	a4,s10,a5
   139bc:	04812c83          	lw	s9,72(sp)
   139c0:	04c12b83          	lw	s7,76(sp)
   139c4:	02012d83          	lw	s11,32(sp)
   139c8:	00058a13          	mv	s4,a1
   139cc:	c7877a63          	bgeu	a4,s8,12e40 <_vfprintf_r+0x1100>
   139d0:	00070c13          	mv	s8,a4
   139d4:	c6cff06f          	j	12e40 <_vfprintf_r+0x1100>
   139d8:	0c812683          	lw	a3,200(sp)
   139dc:	0000df17          	auipc	t5,0xd
   139e0:	4b0f0f13          	addi	t5,t5,1200 # 20e8c <zeroes.0>
   139e4:	07475463          	bge	a4,s4,13a4c <_vfprintf_r+0x1d0c>
   139e8:	01612c23          	sw	s6,24(sp)
   139ec:	00c0006f          	j	139f8 <_vfprintf_r+0x1cb8>
   139f0:	ff0a0a13          	addi	s4,s4,-16
   139f4:	05475a63          	bge	a4,s4,13a48 <_vfprintf_r+0x1d08>
   139f8:	01060613          	addi	a2,a2,16
   139fc:	00168693          	addi	a3,a3,1
   13a00:	0165a023          	sw	s6,0(a1)
   13a04:	00e5a223          	sw	a4,4(a1)
   13a08:	0cc12623          	sw	a2,204(sp)
   13a0c:	0cd12423          	sw	a3,200(sp)
   13a10:	00858593          	addi	a1,a1,8
   13a14:	fcd85ee3          	bge	a6,a3,139f0 <_vfprintf_r+0x1cb0>
   13a18:	0c410613          	addi	a2,sp,196
   13a1c:	00048593          	mv	a1,s1
   13a20:	00090513          	mv	a0,s2
   13a24:	609000ef          	jal	1482c <__sprint_r>
   13a28:	9a051063          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   13a2c:	01000713          	li	a4,16
   13a30:	ff0a0a13          	addi	s4,s4,-16
   13a34:	0cc12603          	lw	a2,204(sp)
   13a38:	0c812683          	lw	a3,200(sp)
   13a3c:	00098593          	mv	a1,s3
   13a40:	00700813          	li	a6,7
   13a44:	fb474ae3          	blt	a4,s4,139f8 <_vfprintf_r+0x1cb8>
   13a48:	01812f03          	lw	t5,24(sp)
   13a4c:	01460633          	add	a2,a2,s4
   13a50:	00168693          	addi	a3,a3,1
   13a54:	01e5a023          	sw	t5,0(a1)
   13a58:	0145a223          	sw	s4,4(a1)
   13a5c:	0cc12623          	sw	a2,204(sp)
   13a60:	0cd12423          	sw	a3,200(sp)
   13a64:	76d84a63          	blt	a6,a3,141d8 <_vfprintf_r+0x2498>
   13a68:	000cc683          	lbu	a3,0(s9)
   13a6c:	00858593          	addi	a1,a1,8
   13a70:	00dd0d33          	add	s10,s10,a3
   13a74:	f29ff06f          	j	1399c <_vfprintf_r+0x1c5c>
   13a78:	fffc8c93          	addi	s9,s9,-1
   13a7c:	fffa8a93          	addi	s5,s5,-1
   13a80:	ea1ff06f          	j	13920 <_vfprintf_r+0x1be0>
   13a84:	0c410613          	addi	a2,sp,196
   13a88:	00048593          	mv	a1,s1
   13a8c:	00090513          	mv	a0,s2
   13a90:	59d000ef          	jal	1482c <__sprint_r>
   13a94:	92051a63          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   13a98:	0cc12603          	lw	a2,204(sp)
   13a9c:	00098593          	mv	a1,s3
   13aa0:	01000713          	li	a4,16
   13aa4:	00700813          	li	a6,7
   13aa8:	ea1ff06f          	j	13948 <_vfprintf_r+0x1c08>
   13aac:	0c410613          	addi	a2,sp,196
   13ab0:	00048593          	mv	a1,s1
   13ab4:	00090513          	mv	a0,s2
   13ab8:	575000ef          	jal	1482c <__sprint_r>
   13abc:	90051663          	bnez	a0,12bc8 <_vfprintf_r+0xe88>
   13ac0:	000cc683          	lbu	a3,0(s9)
   13ac4:	0cc12603          	lw	a2,204(sp)
   13ac8:	00098593          	mv	a1,s3
   13acc:	01000713          	li	a4,16
   13ad0:	00700813          	li	a6,7
   13ad4:	eb1ff06f          	j	13984 <_vfprintf_r+0x1c44>
   13ad8:	04412783          	lw	a5,68(sp)
   13adc:	04012583          	lw	a1,64(sp)
   13ae0:	03012a23          	sw	a6,52(sp)
   13ae4:	40f686b3          	sub	a3,a3,a5
   13ae8:	00078613          	mv	a2,a5
   13aec:	00068513          	mv	a0,a3
   13af0:	03e12223          	sw	t5,36(sp)
   13af4:	03d12023          	sw	t4,32(sp)
   13af8:	01112e23          	sw	a7,28(sp)
   13afc:	00e12c23          	sw	a4,24(sp)
   13b00:	00d12823          	sw	a3,16(sp)
   13b04:	671020ef          	jal	16974 <strncpy>
   13b08:	012d37b3          	sltu	a5,s10,s2
   13b0c:	00500613          	li	a2,5
   13b10:	00fd07b3          	add	a5,s10,a5
   13b14:	02c7f7b3          	remu	a5,a5,a2
   13b18:	01812703          	lw	a4,24(sp)
   13b1c:	ccccd337          	lui	t1,0xccccd
   13b20:	ccccd537          	lui	a0,0xccccd
   13b24:	00174583          	lbu	a1,1(a4)
   13b28:	ccd30313          	addi	t1,t1,-819 # cccccccd <__BSS_END__+0xccca9f9d>
   13b2c:	ccc50513          	addi	a0,a0,-820 # cccccccc <__BSS_END__+0xccca9f9c>
   13b30:	00b035b3          	snez	a1,a1
   13b34:	00b70733          	add	a4,a4,a1
   13b38:	01012683          	lw	a3,16(sp)
   13b3c:	01c12883          	lw	a7,28(sp)
   13b40:	02012e83          	lw	t4,32(sp)
   13b44:	02412f03          	lw	t5,36(sp)
   13b48:	03412803          	lw	a6,52(sp)
   13b4c:	00000613          	li	a2,0
   13b50:	00500e13          	li	t3,5
   13b54:	40f907b3          	sub	a5,s2,a5
   13b58:	00f935b3          	sltu	a1,s2,a5
   13b5c:	40bd85b3          	sub	a1,s11,a1
   13b60:	02a78533          	mul	a0,a5,a0
   13b64:	026585b3          	mul	a1,a1,t1
   13b68:	0267bfb3          	mulhu	t6,a5,t1
   13b6c:	00a585b3          	add	a1,a1,a0
   13b70:	02678533          	mul	a0,a5,t1
   13b74:	01f585b3          	add	a1,a1,t6
   13b78:	01f59793          	slli	a5,a1,0x1f
   13b7c:	0015d593          	srli	a1,a1,0x1
   13b80:	00155513          	srli	a0,a0,0x1
   13b84:	00a7e7b3          	or	a5,a5,a0
   13b88:	a05ff06f          	j	1358c <_vfprintf_r+0x184c>
   13b8c:	00812503          	lw	a0,8(sp)
   13b90:	0c410613          	addi	a2,sp,196
   13b94:	00048593          	mv	a1,s1
   13b98:	495000ef          	jal	1482c <__sprint_r>
   13b9c:	00050463          	beqz	a0,13ba4 <_vfprintf_r+0x1e64>
   13ba0:	828ff06f          	j	12bc8 <_vfprintf_r+0xe88>
   13ba4:	0ac12583          	lw	a1,172(sp)
   13ba8:	0cc12603          	lw	a2,204(sp)
   13bac:	00098a13          	mv	s4,s3
   13bb0:	aadff06f          	j	1365c <_vfprintf_r+0x191c>
   13bb4:	00812503          	lw	a0,8(sp)
   13bb8:	0c410613          	addi	a2,sp,196
   13bbc:	00048593          	mv	a1,s1
   13bc0:	46d000ef          	jal	1482c <__sprint_r>
   13bc4:	00050463          	beqz	a0,13bcc <_vfprintf_r+0x1e8c>
   13bc8:	800ff06f          	j	12bc8 <_vfprintf_r+0xe88>
   13bcc:	0cc12603          	lw	a2,204(sp)
   13bd0:	00098a13          	mv	s4,s3
   13bd4:	a44ff06f          	j	12e18 <_vfprintf_r+0x10d8>
   13bd8:	00600913          	li	s2,6
   13bdc:	df8ff06f          	j	131d4 <_vfprintf_r+0x1494>
   13be0:	0ff97913          	zext.b	s2,s2
   13be4:	00000d93          	li	s11,0
   13be8:	e2cfe06f          	j	12214 <_vfprintf_r+0x4d4>
   13bec:	01891913          	slli	s2,s2,0x18
   13bf0:	41895913          	srai	s2,s2,0x18
   13bf4:	41f95d93          	srai	s11,s2,0x1f
   13bf8:	000d8793          	mv	a5,s11
   13bfc:	d9cfe06f          	j	12198 <_vfprintf_r+0x458>
   13c00:	0ff97913          	zext.b	s2,s2
   13c04:	00000d93          	li	s11,0
   13c08:	d3cff06f          	j	13144 <_vfprintf_r+0x1404>
   13c0c:	01412783          	lw	a5,20(sp)
   13c10:	00c12703          	lw	a4,12(sp)
   13c14:	0007a783          	lw	a5,0(a5)
   13c18:	00e79023          	sh	a4,0(a5)
   13c1c:	b55fe06f          	j	12770 <_vfprintf_r+0xa30>
   13c20:	00812503          	lw	a0,8(sp)
   13c24:	0c410613          	addi	a2,sp,196
   13c28:	00048593          	mv	a1,s1
   13c2c:	401000ef          	jal	1482c <__sprint_r>
   13c30:	c54fe06f          	j	12084 <_vfprintf_r+0x344>
   13c34:	02d00793          	li	a5,45
   13c38:	0af103a3          	sb	a5,167(sp)
   13c3c:	02d00713          	li	a4,45
   13c40:	c21ff06f          	j	13860 <_vfprintf_r+0x1b20>
   13c44:	00812503          	lw	a0,8(sp)
   13c48:	0c410613          	addi	a2,sp,196
   13c4c:	00048593          	mv	a1,s1
   13c50:	3dd000ef          	jal	1482c <__sprint_r>
   13c54:	00050463          	beqz	a0,13c5c <_vfprintf_r+0x1f1c>
   13c58:	f71fe06f          	j	12bc8 <_vfprintf_r+0xe88>
   13c5c:	0ac12583          	lw	a1,172(sp)
   13c60:	0cc12603          	lw	a2,204(sp)
   13c64:	0c812703          	lw	a4,200(sp)
   13c68:	00098a13          	mv	s4,s3
   13c6c:	b4dff06f          	j	137b8 <_vfprintf_r+0x1a78>
   13c70:	01650733          	add	a4,a0,s6
   13c74:	04700613          	li	a2,71
   13c78:	08010593          	addi	a1,sp,128
   13c7c:	00090513          	mv	a0,s2
   13c80:	02e12023          	sw	a4,32(sp)
   13c84:	00c12e23          	sw	a2,28(sp)
   13c88:	01112c23          	sw	a7,24(sp)
   13c8c:	09f12823          	sw	t6,144(sp)
   13c90:	09e12a23          	sw	t5,148(sp)
   13c94:	09d12c23          	sw	t4,152(sp)
   13c98:	09812e23          	sw	s8,156(sp)
   13c9c:	08012023          	sw	zero,128(sp)
   13ca0:	08012223          	sw	zero,132(sp)
   13ca4:	08012423          	sw	zero,136(sp)
   13ca8:	08012623          	sw	zero,140(sp)
   13cac:	765090ef          	jal	1dc10 <__eqtf2>
   13cb0:	01812883          	lw	a7,24(sp)
   13cb4:	01c12603          	lw	a2,28(sp)
   13cb8:	02012703          	lw	a4,32(sp)
   13cbc:	58050c63          	beqz	a0,14254 <_vfprintf_r+0x2514>
   13cc0:	0bc12783          	lw	a5,188(sp)
   13cc4:	00e7fe63          	bgeu	a5,a4,13ce0 <_vfprintf_r+0x1fa0>
   13cc8:	03000593          	li	a1,48
   13ccc:	00178693          	addi	a3,a5,1
   13cd0:	0ad12e23          	sw	a3,188(sp)
   13cd4:	00b78023          	sb	a1,0(a5)
   13cd8:	0bc12783          	lw	a5,188(sp)
   13cdc:	fee7e8e3          	bltu	a5,a4,13ccc <_vfprintf_r+0x1f8c>
   13ce0:	0ac12703          	lw	a4,172(sp)
   13ce4:	00e12c23          	sw	a4,24(sp)
   13ce8:	41a787b3          	sub	a5,a5,s10
   13cec:	04700713          	li	a4,71
   13cf0:	00f12e23          	sw	a5,28(sp)
   13cf4:	e6e60e63          	beq	a2,a4,13370 <_vfprintf_r+0x1630>
   13cf8:	04600713          	li	a4,70
   13cfc:	6ee60263          	beq	a2,a4,143e0 <_vfprintf_r+0x26a0>
   13d00:	01812783          	lw	a5,24(sp)
   13d04:	fff78713          	addi	a4,a5,-1
   13d08:	e84ff06f          	j	1338c <_vfprintf_r+0x164c>
   13d0c:	001b0693          	addi	a3,s6,1
   13d10:	00200613          	li	a2,2
   13d14:	00812503          	lw	a0,8(sp)
   13d18:	09010913          	addi	s2,sp,144
   13d1c:	0ac10713          	addi	a4,sp,172
   13d20:	00090593          	mv	a1,s2
   13d24:	0bc10813          	addi	a6,sp,188
   13d28:	0b010793          	addi	a5,sp,176
   13d2c:	05112623          	sw	a7,76(sp)
   13d30:	02d12223          	sw	a3,36(sp)
   13d34:	09f12823          	sw	t6,144(sp)
   13d38:	03f12023          	sw	t6,32(sp)
   13d3c:	09e12a23          	sw	t5,148(sp)
   13d40:	01e12e23          	sw	t5,28(sp)
   13d44:	09d12c23          	sw	t4,152(sp)
   13d48:	01d12c23          	sw	t4,24(sp)
   13d4c:	09812e23          	sw	s8,156(sp)
   13d50:	4f0030ef          	jal	17240 <_ldtoa_r>
   13d54:	04c12883          	lw	a7,76(sp)
   13d58:	02412683          	lw	a3,36(sp)
   13d5c:	04600593          	li	a1,70
   13d60:	fdf8f613          	andi	a2,a7,-33
   13d64:	01812e83          	lw	t4,24(sp)
   13d68:	01c12f03          	lw	t5,28(sp)
   13d6c:	02012f83          	lw	t6,32(sp)
   13d70:	00050d13          	mv	s10,a0
   13d74:	00d50733          	add	a4,a0,a3
   13d78:	08b61ae3          	bne	a2,a1,1460c <_vfprintf_r+0x28cc>
   13d7c:	000d4503          	lbu	a0,0(s10)
   13d80:	03000593          	li	a1,48
   13d84:	5cb50c63          	beq	a0,a1,1435c <_vfprintf_r+0x261c>
   13d88:	0ac12683          	lw	a3,172(sp)
   13d8c:	08010593          	addi	a1,sp,128
   13d90:	00d70733          	add	a4,a4,a3
   13d94:	ee9ff06f          	j	13c7c <_vfprintf_r+0x1f3c>
   13d98:	09010913          	addi	s2,sp,144
   13d9c:	08010593          	addi	a1,sp,128
   13da0:	0ac10613          	addi	a2,sp,172
   13da4:	00090513          	mv	a0,s2
   13da8:	03112e23          	sw	a7,60(sp)
   13dac:	09f12023          	sw	t6,128(sp)
   13db0:	09e12223          	sw	t5,132(sp)
   13db4:	09d12423          	sw	t4,136(sp)
   13db8:	00b12c23          	sw	a1,24(sp)
   13dbc:	09812623          	sw	s8,140(sp)
   13dc0:	250030ef          	jal	17010 <frexpl>
   13dc4:	09012803          	lw	a6,144(sp)
   13dc8:	0000d717          	auipc	a4,0xd
   13dcc:	0e870713          	addi	a4,a4,232 # 20eb0 <blanks.1+0x14>
   13dd0:	00072503          	lw	a0,0(a4)
   13dd4:	09012023          	sw	a6,128(sp)
   13dd8:	09412803          	lw	a6,148(sp)
   13ddc:	00472603          	lw	a2,4(a4)
   13de0:	00872683          	lw	a3,8(a4)
   13de4:	09012223          	sw	a6,132(sp)
   13de8:	09812803          	lw	a6,152(sp)
   13dec:	00c72703          	lw	a4,12(a4)
   13df0:	01812583          	lw	a1,24(sp)
   13df4:	09012423          	sw	a6,136(sp)
   13df8:	09c12803          	lw	a6,156(sp)
   13dfc:	06a12823          	sw	a0,112(sp)
   13e00:	06c12a23          	sw	a2,116(sp)
   13e04:	00090513          	mv	a0,s2
   13e08:	07010613          	addi	a2,sp,112
   13e0c:	09012623          	sw	a6,140(sp)
   13e10:	06d12c23          	sw	a3,120(sp)
   13e14:	06e12e23          	sw	a4,124(sp)
   13e18:	1240a0ef          	jal	1df3c <__multf3>
   13e1c:	01812583          	lw	a1,24(sp)
   13e20:	09012f03          	lw	t5,144(sp)
   13e24:	09412e83          	lw	t4,148(sp)
   13e28:	09812803          	lw	a6,152(sp)
   13e2c:	00090513          	mv	a0,s2
   13e30:	02b12223          	sw	a1,36(sp)
   13e34:	03e12023          	sw	t5,32(sp)
   13e38:	01d12e23          	sw	t4,28(sp)
   13e3c:	01012c23          	sw	a6,24(sp)
   13e40:	08012023          	sw	zero,128(sp)
   13e44:	08012223          	sw	zero,132(sp)
   13e48:	08012423          	sw	zero,136(sp)
   13e4c:	08012623          	sw	zero,140(sp)
   13e50:	5c1090ef          	jal	1dc10 <__eqtf2>
   13e54:	09c12d83          	lw	s11,156(sp)
   13e58:	01812803          	lw	a6,24(sp)
   13e5c:	01c12e83          	lw	t4,28(sp)
   13e60:	02012f03          	lw	t5,32(sp)
   13e64:	02412583          	lw	a1,36(sp)
   13e68:	03c12883          	lw	a7,60(sp)
   13e6c:	00051663          	bnez	a0,13e78 <_vfprintf_r+0x2138>
   13e70:	00100713          	li	a4,1
   13e74:	0ae12623          	sw	a4,172(sp)
   13e78:	0000d797          	auipc	a5,0xd
   13e7c:	ce478793          	addi	a5,a5,-796 # 20b5c <_exit+0x1ac>
   13e80:	06100713          	li	a4,97
   13e84:	00f12c23          	sw	a5,24(sp)
   13e88:	00e89863          	bne	a7,a4,13e98 <_vfprintf_r+0x2158>
   13e8c:	0000d797          	auipc	a5,0xd
   13e90:	cbc78793          	addi	a5,a5,-836 # 20b48 <_exit+0x198>
   13e94:	00f12c23          	sw	a5,24(sp)
   13e98:	0000d717          	auipc	a4,0xd
   13e9c:	02870713          	addi	a4,a4,40 # 20ec0 <blanks.1+0x24>
   13ea0:	00072783          	lw	a5,0(a4)
   13ea4:	03512e23          	sw	s5,60(sp)
   13ea8:	05712823          	sw	s7,80(sp)
   13eac:	00f12e23          	sw	a5,28(sp)
   13eb0:	00472783          	lw	a5,4(a4)
   13eb4:	05412a23          	sw	s4,84(sp)
   13eb8:	000d0a93          	mv	s5,s10
   13ebc:	02f12023          	sw	a5,32(sp)
   13ec0:	00872783          	lw	a5,8(a4)
   13ec4:	04912c23          	sw	s1,88(sp)
   13ec8:	05a12e23          	sw	s10,92(sp)
   13ecc:	02f12223          	sw	a5,36(sp)
   13ed0:	00c72783          	lw	a5,12(a4)
   13ed4:	000d8c13          	mv	s8,s11
   13ed8:	fffb0b13          	addi	s6,s6,-1
   13edc:	05112423          	sw	a7,72(sp)
   13ee0:	05912623          	sw	s9,76(sp)
   13ee4:	00078d13          	mv	s10,a5
   13ee8:	000e8a13          	mv	s4,t4
   13eec:	00080b93          	mv	s7,a6
   13ef0:	000f0d93          	mv	s11,t5
   13ef4:	00058493          	mv	s1,a1
   13ef8:	03c0006f          	j	13f34 <_vfprintf_r+0x21f4>
   13efc:	00048593          	mv	a1,s1
   13f00:	00090513          	mv	a0,s2
   13f04:	09b12823          	sw	s11,144(sp)
   13f08:	09412a23          	sw	s4,148(sp)
   13f0c:	09712c23          	sw	s7,152(sp)
   13f10:	09812e23          	sw	s8,156(sp)
   13f14:	08012023          	sw	zero,128(sp)
   13f18:	08012223          	sw	zero,132(sp)
   13f1c:	08012423          	sw	zero,136(sp)
   13f20:	08012623          	sw	zero,140(sp)
   13f24:	fffb0c93          	addi	s9,s6,-1
   13f28:	4e9090ef          	jal	1dc10 <__eqtf2>
   13f2c:	5e050863          	beqz	a0,1451c <_vfprintf_r+0x27dc>
   13f30:	000c8b13          	mv	s6,s9
   13f34:	01c12783          	lw	a5,28(sp)
   13f38:	07010613          	addi	a2,sp,112
   13f3c:	00048593          	mv	a1,s1
   13f40:	06f12823          	sw	a5,112(sp)
   13f44:	02012783          	lw	a5,32(sp)
   13f48:	00090513          	mv	a0,s2
   13f4c:	09b12023          	sw	s11,128(sp)
   13f50:	06f12a23          	sw	a5,116(sp)
   13f54:	02412783          	lw	a5,36(sp)
   13f58:	09412223          	sw	s4,132(sp)
   13f5c:	09712423          	sw	s7,136(sp)
   13f60:	06f12c23          	sw	a5,120(sp)
   13f64:	09812623          	sw	s8,140(sp)
   13f68:	07a12e23          	sw	s10,124(sp)
   13f6c:	7d1090ef          	jal	1df3c <__multf3>
   13f70:	00090513          	mv	a0,s2
   13f74:	4e80c0ef          	jal	2045c <__fixtfsi>
   13f78:	00050593          	mv	a1,a0
   13f7c:	00050c93          	mv	s9,a0
   13f80:	00090513          	mv	a0,s2
   13f84:	09012d83          	lw	s11,144(sp)
   13f88:	09412c03          	lw	s8,148(sp)
   13f8c:	09812b83          	lw	s7,152(sp)
   13f90:	09c12a03          	lw	s4,156(sp)
   13f94:	5c00c0ef          	jal	20554 <__floatsitf>
   13f98:	09012683          	lw	a3,144(sp)
   13f9c:	06010613          	addi	a2,sp,96
   13fa0:	07010593          	addi	a1,sp,112
   13fa4:	06d12023          	sw	a3,96(sp)
   13fa8:	09412683          	lw	a3,148(sp)
   13fac:	00048513          	mv	a0,s1
   13fb0:	07b12823          	sw	s11,112(sp)
   13fb4:	06d12223          	sw	a3,100(sp)
   13fb8:	09812683          	lw	a3,152(sp)
   13fbc:	07812a23          	sw	s8,116(sp)
   13fc0:	07712c23          	sw	s7,120(sp)
   13fc4:	06d12423          	sw	a3,104(sp)
   13fc8:	09c12683          	lw	a3,156(sp)
   13fcc:	07412e23          	sw	s4,124(sp)
   13fd0:	06d12623          	sw	a3,108(sp)
   13fd4:	76d0a0ef          	jal	1ef40 <__subtf3>
   13fd8:	01812783          	lw	a5,24(sp)
   13fdc:	000a8f93          	mv	t6,s5
   13fe0:	001a8a93          	addi	s5,s5,1
   13fe4:	019786b3          	add	a3,a5,s9
   13fe8:	0006c603          	lbu	a2,0(a3)
   13fec:	08012d83          	lw	s11,128(sp)
   13ff0:	08412a03          	lw	s4,132(sp)
   13ff4:	08812b83          	lw	s7,136(sp)
   13ff8:	08c12c03          	lw	s8,140(sp)
   13ffc:	fff00793          	li	a5,-1
   14000:	feca8fa3          	sb	a2,-1(s5)
   14004:	eefb1ce3          	bne	s6,a5,13efc <_vfprintf_r+0x21bc>
   14008:	0000d517          	auipc	a0,0xd
   1400c:	ec850513          	addi	a0,a0,-312 # 20ed0 <blanks.1+0x34>
   14010:	04812883          	lw	a7,72(sp)
   14014:	00452283          	lw	t0,4(a0)
   14018:	00852383          	lw	t2,8(a0)
   1401c:	00c52783          	lw	a5,12(a0)
   14020:	00052b03          	lw	s6,0(a0)
   14024:	000d8f13          	mv	t5,s11
   14028:	000a0e93          	mv	t4,s4
   1402c:	000b8813          	mv	a6,s7
   14030:	00048593          	mv	a1,s1
   14034:	000c0d93          	mv	s11,s8
   14038:	05c12d03          	lw	s10,92(sp)
   1403c:	05412a03          	lw	s4,84(sp)
   14040:	05012b83          	lw	s7,80(sp)
   14044:	05812483          	lw	s1,88(sp)
   14048:	03f12023          	sw	t6,32(sp)
   1404c:	01112e23          	sw	a7,28(sp)
   14050:	05912e23          	sw	s9,92(sp)
   14054:	000a8c13          	mv	s8,s5
   14058:	04c12c83          	lw	s9,76(sp)
   1405c:	03c12a83          	lw	s5,60(sp)
   14060:	09e12823          	sw	t5,144(sp)
   14064:	05e12c23          	sw	t5,88(sp)
   14068:	09d12a23          	sw	t4,148(sp)
   1406c:	05d12a23          	sw	t4,84(sp)
   14070:	09012c23          	sw	a6,152(sp)
   14074:	05012823          	sw	a6,80(sp)
   14078:	09b12e23          	sw	s11,156(sp)
   1407c:	09612023          	sw	s6,128(sp)
   14080:	08512223          	sw	t0,132(sp)
   14084:	04512623          	sw	t0,76(sp)
   14088:	08712423          	sw	t2,136(sp)
   1408c:	04712423          	sw	t2,72(sp)
   14090:	08f12623          	sw	a5,140(sp)
   14094:	02f12e23          	sw	a5,60(sp)
   14098:	02b12223          	sw	a1,36(sp)
   1409c:	00090513          	mv	a0,s2
   140a0:	43d090ef          	jal	1dcdc <__getf2>
   140a4:	01c12883          	lw	a7,28(sp)
   140a8:	02012f83          	lw	t6,32(sp)
   140ac:	02a04863          	bgtz	a0,140dc <_vfprintf_r+0x239c>
   140b0:	02412583          	lw	a1,36(sp)
   140b4:	00090513          	mv	a0,s2
   140b8:	03112023          	sw	a7,32(sp)
   140bc:	01f12e23          	sw	t6,28(sp)
   140c0:	351090ef          	jal	1dc10 <__eqtf2>
   140c4:	02012883          	lw	a7,32(sp)
   140c8:	04051e63          	bnez	a0,14124 <_vfprintf_r+0x23e4>
   140cc:	05c12703          	lw	a4,92(sp)
   140d0:	01c12f83          	lw	t6,28(sp)
   140d4:	00177693          	andi	a3,a4,1
   140d8:	04068663          	beqz	a3,14124 <_vfprintf_r+0x23e4>
   140dc:	01812783          	lw	a5,24(sp)
   140e0:	0bf12e23          	sw	t6,188(sp)
   140e4:	fffc4603          	lbu	a2,-1(s8)
   140e8:	00f7c583          	lbu	a1,15(a5)
   140ec:	000c0693          	mv	a3,s8
   140f0:	02b61063          	bne	a2,a1,14110 <_vfprintf_r+0x23d0>
   140f4:	03000513          	li	a0,48
   140f8:	fea68fa3          	sb	a0,-1(a3)
   140fc:	0bc12683          	lw	a3,188(sp)
   14100:	fff68793          	addi	a5,a3,-1
   14104:	0af12e23          	sw	a5,188(sp)
   14108:	fff6c603          	lbu	a2,-1(a3)
   1410c:	fec586e3          	beq	a1,a2,140f8 <_vfprintf_r+0x23b8>
   14110:	00160593          	addi	a1,a2,1
   14114:	03900513          	li	a0,57
   14118:	0ff5f593          	zext.b	a1,a1
   1411c:	04a60463          	beq	a2,a0,14164 <_vfprintf_r+0x2424>
   14120:	feb68fa3          	sb	a1,-1(a3)
   14124:	0ac12783          	lw	a5,172(sp)
   14128:	41ac0733          	sub	a4,s8,s10
   1412c:	00e12e23          	sw	a4,28(sp)
   14130:	fff78713          	addi	a4,a5,-1
   14134:	00f12c23          	sw	a5,24(sp)
   14138:	06100613          	li	a2,97
   1413c:	0ae12623          	sw	a4,172(sp)
   14140:	07000693          	li	a3,112
   14144:	00c88663          	beq	a7,a2,14150 <_vfprintf_r+0x2410>
   14148:	05000693          	li	a3,80
   1414c:	04100893          	li	a7,65
   14150:	00100613          	li	a2,1
   14154:	a44ff06f          	j	13398 <_vfprintf_r+0x1658>
   14158:	000b0693          	mv	a3,s6
   1415c:	00300613          	li	a2,3
   14160:	bb5ff06f          	j	13d14 <_vfprintf_r+0x1fd4>
   14164:	01812783          	lw	a5,24(sp)
   14168:	00a7c583          	lbu	a1,10(a5)
   1416c:	fb5ff06f          	j	14120 <_vfprintf_r+0x23e0>
   14170:	0a714703          	lbu	a4,167(sp)
   14174:	04812c83          	lw	s9,72(sp)
   14178:	00000b13          	li	s6,0
   1417c:	00070463          	beqz	a4,14184 <_vfprintf_r+0x2444>
   14180:	864fe06f          	j	121e4 <_vfprintf_r+0x4a4>
   14184:	dd5fd06f          	j	11f58 <_vfprintf_r+0x218>
   14188:	01c12783          	lw	a5,28(sp)
   1418c:	01812703          	lw	a4,24(sp)
   14190:	10f74263          	blt	a4,a5,14294 <_vfprintf_r+0x2554>
   14194:	01812783          	lw	a5,24(sp)
   14198:	001cf713          	andi	a4,s9,1
   1419c:	00078913          	mv	s2,a5
   141a0:	00070663          	beqz	a4,141ac <_vfprintf_r+0x246c>
   141a4:	02812703          	lw	a4,40(sp)
   141a8:	00e78933          	add	s2,a5,a4
   141ac:	400cfe13          	andi	t3,s9,1024
   141b0:	000e0663          	beqz	t3,141bc <_vfprintf_r+0x247c>
   141b4:	01812783          	lw	a5,24(sp)
   141b8:	2ef04263          	bgtz	a5,1449c <_vfprintf_r+0x275c>
   141bc:	fff94693          	not	a3,s2
   141c0:	41f6d693          	srai	a3,a3,0x1f
   141c4:	00d97db3          	and	s11,s2,a3
   141c8:	06700893          	li	a7,103
   141cc:	02012223          	sw	zero,36(sp)
   141d0:	02012023          	sw	zero,32(sp)
   141d4:	aa0ff06f          	j	13474 <_vfprintf_r+0x1734>
   141d8:	0c410613          	addi	a2,sp,196
   141dc:	00048593          	mv	a1,s1
   141e0:	00090513          	mv	a0,s2
   141e4:	648000ef          	jal	1482c <__sprint_r>
   141e8:	00050463          	beqz	a0,141f0 <_vfprintf_r+0x24b0>
   141ec:	9ddfe06f          	j	12bc8 <_vfprintf_r+0xe88>
   141f0:	000cc683          	lbu	a3,0(s9)
   141f4:	0cc12603          	lw	a2,204(sp)
   141f8:	00098593          	mv	a1,s3
   141fc:	01000713          	li	a4,16
   14200:	00700813          	li	a6,7
   14204:	00dd0d33          	add	s10,s10,a3
   14208:	f94ff06f          	j	1399c <_vfprintf_r+0x1c5c>
   1420c:	00900513          	li	a0,9
   14210:	b7256e63          	bltu	a0,s2,1358c <_vfprintf_r+0x184c>
   14214:	bf8ff06f          	j	1360c <_vfprintf_r+0x18cc>
   14218:	02d00793          	li	a5,45
   1421c:	0af103a3          	sb	a5,167(sp)
   14220:	02d00713          	li	a4,45
   14224:	d0cff06f          	j	13730 <_vfprintf_r+0x19f0>
   14228:	0a714703          	lbu	a4,167(sp)
   1422c:	01812a23          	sw	s8,20(sp)
   14230:	02012223          	sw	zero,36(sp)
   14234:	02012023          	sw	zero,32(sp)
   14238:	00012c23          	sw	zero,24(sp)
   1423c:	000b0d93          	mv	s11,s6
   14240:	000b0913          	mv	s2,s6
   14244:	00000b13          	li	s6,0
   14248:	00070463          	beqz	a4,14250 <_vfprintf_r+0x2510>
   1424c:	f99fd06f          	j	121e4 <_vfprintf_r+0x4a4>
   14250:	d09fd06f          	j	11f58 <_vfprintf_r+0x218>
   14254:	0ac12783          	lw	a5,172(sp)
   14258:	00f12c23          	sw	a5,24(sp)
   1425c:	00070793          	mv	a5,a4
   14260:	a89ff06f          	j	13ce8 <_vfprintf_r+0x1fa8>
   14264:	00812503          	lw	a0,8(sp)
   14268:	0c410613          	addi	a2,sp,196
   1426c:	00048593          	mv	a1,s1
   14270:	5bc000ef          	jal	1482c <__sprint_r>
   14274:	00050463          	beqz	a0,1427c <_vfprintf_r+0x253c>
   14278:	951fe06f          	j	12bc8 <_vfprintf_r+0xe88>
   1427c:	0ac12583          	lw	a1,172(sp)
   14280:	01c12783          	lw	a5,28(sp)
   14284:	0cc12603          	lw	a2,204(sp)
   14288:	00098a13          	mv	s4,s3
   1428c:	40b785b3          	sub	a1,a5,a1
   14290:	c0dfe06f          	j	12e9c <_vfprintf_r+0x115c>
   14294:	01c12783          	lw	a5,28(sp)
   14298:	02812703          	lw	a4,40(sp)
   1429c:	06700893          	li	a7,103
   142a0:	00e78933          	add	s2,a5,a4
   142a4:	01812783          	lw	a5,24(sp)
   142a8:	2af05a63          	blez	a5,1455c <_vfprintf_r+0x281c>
   142ac:	400cfe13          	andi	t3,s9,1024
   142b0:	1e0e1863          	bnez	t3,144a0 <_vfprintf_r+0x2760>
   142b4:	fff94693          	not	a3,s2
   142b8:	41f6d693          	srai	a3,a3,0x1f
   142bc:	00d97db3          	and	s11,s2,a3
   142c0:	f0dff06f          	j	141cc <_vfprintf_r+0x248c>
   142c4:	01012383          	lw	t2,16(sp)
   142c8:	000a0493          	mv	s1,s4
   142cc:	901fe06f          	j	12bcc <_vfprintf_r+0xe8c>
   142d0:	000c8e93          	mv	t4,s9
   142d4:	bc4fe06f          	j	12698 <_vfprintf_r+0x958>
   142d8:	ff000513          	li	a0,-16
   142dc:	40b00933          	neg	s2,a1
   142e0:	0000d817          	auipc	a6,0xd
   142e4:	bac80813          	addi	a6,a6,-1108 # 20e8c <zeroes.0>
   142e8:	12a5d063          	bge	a1,a0,14408 <_vfprintf_r+0x26c8>
   142ec:	01512c23          	sw	s5,24(sp)
   142f0:	01000b13          	li	s6,16
   142f4:	00700c13          	li	s8,7
   142f8:	00080a93          	mv	s5,a6
   142fc:	00c0006f          	j	14308 <_vfprintf_r+0x25c8>
   14300:	ff090913          	addi	s2,s2,-16
   14304:	0f2b5e63          	bge	s6,s2,14400 <_vfprintf_r+0x26c0>
   14308:	01060613          	addi	a2,a2,16
   1430c:	00170713          	addi	a4,a4,1
   14310:	015a2023          	sw	s5,0(s4)
   14314:	016a2223          	sw	s6,4(s4)
   14318:	0cc12623          	sw	a2,204(sp)
   1431c:	0ce12423          	sw	a4,200(sp)
   14320:	008a0a13          	addi	s4,s4,8
   14324:	fcec5ee3          	bge	s8,a4,14300 <_vfprintf_r+0x25c0>
   14328:	00812503          	lw	a0,8(sp)
   1432c:	0c410613          	addi	a2,sp,196
   14330:	00048593          	mv	a1,s1
   14334:	4f8000ef          	jal	1482c <__sprint_r>
   14338:	00050463          	beqz	a0,14340 <_vfprintf_r+0x2600>
   1433c:	88dfe06f          	j	12bc8 <_vfprintf_r+0xe88>
   14340:	0cc12603          	lw	a2,204(sp)
   14344:	0c812703          	lw	a4,200(sp)
   14348:	00098a13          	mv	s4,s3
   1434c:	fb5ff06f          	j	14300 <_vfprintf_r+0x25c0>
   14350:	fff00793          	li	a5,-1
   14354:	00f12623          	sw	a5,12(sp)
   14358:	d61fd06f          	j	120b8 <_vfprintf_r+0x378>
   1435c:	08010593          	addi	a1,sp,128
   14360:	00090513          	mv	a0,s2
   14364:	04e12c23          	sw	a4,88(sp)
   14368:	04c12a23          	sw	a2,84(sp)
   1436c:	05112823          	sw	a7,80(sp)
   14370:	04d12623          	sw	a3,76(sp)
   14374:	09f12823          	sw	t6,144(sp)
   14378:	03f12223          	sw	t6,36(sp)
   1437c:	09e12a23          	sw	t5,148(sp)
   14380:	03e12023          	sw	t5,32(sp)
   14384:	09d12c23          	sw	t4,152(sp)
   14388:	01d12e23          	sw	t4,28(sp)
   1438c:	00b12c23          	sw	a1,24(sp)
   14390:	09812e23          	sw	s8,156(sp)
   14394:	08012023          	sw	zero,128(sp)
   14398:	08012223          	sw	zero,132(sp)
   1439c:	08012423          	sw	zero,136(sp)
   143a0:	08012623          	sw	zero,140(sp)
   143a4:	06d090ef          	jal	1dc10 <__eqtf2>
   143a8:	01812583          	lw	a1,24(sp)
   143ac:	01c12e83          	lw	t4,28(sp)
   143b0:	02012f03          	lw	t5,32(sp)
   143b4:	02412f83          	lw	t6,36(sp)
   143b8:	04c12683          	lw	a3,76(sp)
   143bc:	05012883          	lw	a7,80(sp)
   143c0:	05412603          	lw	a2,84(sp)
   143c4:	05812703          	lw	a4,88(sp)
   143c8:	1c051063          	bnez	a0,14588 <_vfprintf_r+0x2848>
   143cc:	0ac12783          	lw	a5,172(sp)
   143d0:	00f70733          	add	a4,a4,a5
   143d4:	00f12c23          	sw	a5,24(sp)
   143d8:	41a707b3          	sub	a5,a4,s10
   143dc:	00f12e23          	sw	a5,28(sp)
   143e0:	01812783          	lw	a5,24(sp)
   143e4:	001cf713          	andi	a4,s9,1
   143e8:	01676733          	or	a4,a4,s6
   143ec:	1af05863          	blez	a5,1459c <_vfprintf_r+0x285c>
   143f0:	18071263          	bnez	a4,14574 <_vfprintf_r+0x2834>
   143f4:	01812903          	lw	s2,24(sp)
   143f8:	06600893          	li	a7,102
   143fc:	eb1ff06f          	j	142ac <_vfprintf_r+0x256c>
   14400:	000a8813          	mv	a6,s5
   14404:	01812a83          	lw	s5,24(sp)
   14408:	01260633          	add	a2,a2,s2
   1440c:	00170713          	addi	a4,a4,1
   14410:	010a2023          	sw	a6,0(s4)
   14414:	012a2223          	sw	s2,4(s4)
   14418:	0cc12623          	sw	a2,204(sp)
   1441c:	0ce12423          	sw	a4,200(sp)
   14420:	00700593          	li	a1,7
   14424:	a6e5de63          	bge	a1,a4,136a0 <_vfprintf_r+0x1960>
   14428:	00812503          	lw	a0,8(sp)
   1442c:	0c410613          	addi	a2,sp,196
   14430:	00048593          	mv	a1,s1
   14434:	3f8000ef          	jal	1482c <__sprint_r>
   14438:	00050463          	beqz	a0,14440 <_vfprintf_r+0x2700>
   1443c:	f8cfe06f          	j	12bc8 <_vfprintf_r+0xe88>
   14440:	0cc12603          	lw	a2,204(sp)
   14444:	0c812703          	lw	a4,200(sp)
   14448:	00098a13          	mv	s4,s3
   1444c:	b70ff06f          	j	137bc <_vfprintf_r+0x1a7c>
   14450:	0b610693          	addi	a3,sp,182
   14454:	00061863          	bnez	a2,14464 <_vfprintf_r+0x2724>
   14458:	03000693          	li	a3,48
   1445c:	0ad10b23          	sb	a3,182(sp)
   14460:	0b710693          	addi	a3,sp,183
   14464:	19010793          	addi	a5,sp,400
   14468:	40f68633          	sub	a2,a3,a5
   1446c:	03070713          	addi	a4,a4,48
   14470:	0dd60793          	addi	a5,a2,221
   14474:	00e68023          	sb	a4,0(a3)
   14478:	02f12e23          	sw	a5,60(sp)
   1447c:	fb5fe06f          	j	13430 <_vfprintf_r+0x16f0>
   14480:	001cf713          	andi	a4,s9,1
   14484:	00071463          	bnez	a4,1448c <_vfprintf_r+0x274c>
   14488:	fc9fe06f          	j	13450 <_vfprintf_r+0x1710>
   1448c:	fbdfe06f          	j	13448 <_vfprintf_r+0x1708>
   14490:	00012823          	sw	zero,16(sp)
   14494:	00600b13          	li	s6,6
   14498:	ea9fd06f          	j	12340 <_vfprintf_r+0x600>
   1449c:	06700893          	li	a7,103
   144a0:	03812603          	lw	a2,56(sp)
   144a4:	0ff00693          	li	a3,255
   144a8:	00064703          	lbu	a4,0(a2)
   144ac:	14d70a63          	beq	a4,a3,14600 <_vfprintf_r+0x28c0>
   144b0:	01812783          	lw	a5,24(sp)
   144b4:	00000513          	li	a0,0
   144b8:	00000593          	li	a1,0
   144bc:	00f75e63          	bge	a4,a5,144d8 <_vfprintf_r+0x2798>
   144c0:	40e787b3          	sub	a5,a5,a4
   144c4:	00164703          	lbu	a4,1(a2)
   144c8:	04070463          	beqz	a4,14510 <_vfprintf_r+0x27d0>
   144cc:	00158593          	addi	a1,a1,1
   144d0:	00160613          	addi	a2,a2,1
   144d4:	fed714e3          	bne	a4,a3,144bc <_vfprintf_r+0x277c>
   144d8:	02c12c23          	sw	a2,56(sp)
   144dc:	02b12023          	sw	a1,32(sp)
   144e0:	02a12223          	sw	a0,36(sp)
   144e4:	00f12c23          	sw	a5,24(sp)
   144e8:	02412783          	lw	a5,36(sp)
   144ec:	02012703          	lw	a4,32(sp)
   144f0:	00e78733          	add	a4,a5,a4
   144f4:	04412783          	lw	a5,68(sp)
   144f8:	02f70733          	mul	a4,a4,a5
   144fc:	01270933          	add	s2,a4,s2
   14500:	fff94693          	not	a3,s2
   14504:	41f6d693          	srai	a3,a3,0x1f
   14508:	00d97db3          	and	s11,s2,a3
   1450c:	f69fe06f          	j	13474 <_vfprintf_r+0x1734>
   14510:	00064703          	lbu	a4,0(a2)
   14514:	00150513          	addi	a0,a0,1
   14518:	fbdff06f          	j	144d4 <_vfprintf_r+0x2794>
   1451c:	000a8c13          	mv	s8,s5
   14520:	001b0693          	addi	a3,s6,1
   14524:	04812883          	lw	a7,72(sp)
   14528:	04c12c83          	lw	s9,76(sp)
   1452c:	05012b83          	lw	s7,80(sp)
   14530:	05412a03          	lw	s4,84(sp)
   14534:	03c12a83          	lw	s5,60(sp)
   14538:	05812483          	lw	s1,88(sp)
   1453c:	05c12d03          	lw	s10,92(sp)
   14540:	00dc06b3          	add	a3,s8,a3
   14544:	03000613          	li	a2,48
   14548:	bc0b4ee3          	bltz	s6,14124 <_vfprintf_r+0x23e4>
   1454c:	001c0c13          	addi	s8,s8,1
   14550:	fecc0fa3          	sb	a2,-1(s8)
   14554:	ff869ce3          	bne	a3,s8,1454c <_vfprintf_r+0x280c>
   14558:	bcdff06f          	j	14124 <_vfprintf_r+0x23e4>
   1455c:	40f90f33          	sub	t5,s2,a5
   14560:	001f0913          	addi	s2,t5,1
   14564:	fff94693          	not	a3,s2
   14568:	41f6d693          	srai	a3,a3,0x1f
   1456c:	00d97db3          	and	s11,s2,a3
   14570:	c5dff06f          	j	141cc <_vfprintf_r+0x248c>
   14574:	02812703          	lw	a4,40(sp)
   14578:	06600893          	li	a7,102
   1457c:	00eb0f33          	add	t5,s6,a4
   14580:	00ff0933          	add	s2,t5,a5
   14584:	d29ff06f          	j	142ac <_vfprintf_r+0x256c>
   14588:	00100513          	li	a0,1
   1458c:	40d506b3          	sub	a3,a0,a3
   14590:	0ad12623          	sw	a3,172(sp)
   14594:	00d70733          	add	a4,a4,a3
   14598:	ee4ff06f          	j	13c7c <_vfprintf_r+0x1f3c>
   1459c:	00071a63          	bnez	a4,145b0 <_vfprintf_r+0x2870>
   145a0:	00100d93          	li	s11,1
   145a4:	06600893          	li	a7,102
   145a8:	00100913          	li	s2,1
   145ac:	c21ff06f          	j	141cc <_vfprintf_r+0x248c>
   145b0:	02812783          	lw	a5,40(sp)
   145b4:	06600893          	li	a7,102
   145b8:	00178f13          	addi	t5,a5,1
   145bc:	016f0933          	add	s2,t5,s6
   145c0:	fff94693          	not	a3,s2
   145c4:	41f6d693          	srai	a3,a3,0x1f
   145c8:	00d97db3          	and	s11,s2,a3
   145cc:	c01ff06f          	j	141cc <_vfprintf_r+0x248c>
   145d0:	00200793          	li	a5,2
   145d4:	02f12e23          	sw	a5,60(sp)
   145d8:	e59fe06f          	j	13430 <_vfprintf_r+0x16f0>
   145dc:	01412783          	lw	a5,20(sp)
   145e0:	0007ab03          	lw	s6,0(a5)
   145e4:	00478793          	addi	a5,a5,4
   145e8:	000b5463          	bgez	s6,145f0 <_vfprintf_r+0x28b0>
   145ec:	fff00b13          	li	s6,-1
   145f0:	001ac883          	lbu	a7,1(s5)
   145f4:	00f12a23          	sw	a5,20(sp)
   145f8:	00068a93          	mv	s5,a3
   145fc:	8d5fd06f          	j	11ed0 <_vfprintf_r+0x190>
   14600:	02012223          	sw	zero,36(sp)
   14604:	02012023          	sw	zero,32(sp)
   14608:	ee1ff06f          	j	144e8 <_vfprintf_r+0x27a8>
   1460c:	04500613          	li	a2,69
   14610:	08010593          	addi	a1,sp,128
   14614:	e68ff06f          	j	13c7c <_vfprintf_r+0x1f3c>
   14618:	00c4d783          	lhu	a5,12(s1)
   1461c:	0407e793          	ori	a5,a5,64
   14620:	00f49623          	sh	a5,12(s1)
   14624:	a61fd06f          	j	12084 <_vfprintf_r+0x344>

00014628 <vfprintf>:
   14628:	00060693          	mv	a3,a2
   1462c:	00058613          	mv	a2,a1
   14630:	00050593          	mv	a1,a0
   14634:	08c1a503          	lw	a0,140(gp) # 2290c <_impure_ptr>
   14638:	f08fd06f          	j	11d40 <_vfprintf_r>

0001463c <__sbprintf>:
   1463c:	b8010113          	addi	sp,sp,-1152
   14640:	00c59783          	lh	a5,12(a1)
   14644:	00e5d703          	lhu	a4,14(a1)
   14648:	46812c23          	sw	s0,1144(sp)
   1464c:	00058413          	mv	s0,a1
   14650:	000105b7          	lui	a1,0x10
   14654:	ffd58593          	addi	a1,a1,-3 # fffd <exit-0xb7>
   14658:	06442e03          	lw	t3,100(s0)
   1465c:	01c42303          	lw	t1,28(s0)
   14660:	02442883          	lw	a7,36(s0)
   14664:	01071713          	slli	a4,a4,0x10
   14668:	00b7f7b3          	and	a5,a5,a1
   1466c:	00e7e7b3          	or	a5,a5,a4
   14670:	40000813          	li	a6,1024
   14674:	00f12a23          	sw	a5,20(sp)
   14678:	00810593          	addi	a1,sp,8
   1467c:	07010793          	addi	a5,sp,112
   14680:	46912a23          	sw	s1,1140(sp)
   14684:	47212823          	sw	s2,1136(sp)
   14688:	46112e23          	sw	ra,1148(sp)
   1468c:	00050913          	mv	s2,a0
   14690:	07c12623          	sw	t3,108(sp)
   14694:	02612223          	sw	t1,36(sp)
   14698:	03112623          	sw	a7,44(sp)
   1469c:	00f12423          	sw	a5,8(sp)
   146a0:	00f12c23          	sw	a5,24(sp)
   146a4:	01012823          	sw	a6,16(sp)
   146a8:	01012e23          	sw	a6,28(sp)
   146ac:	02012023          	sw	zero,32(sp)
   146b0:	e90fd0ef          	jal	11d40 <_vfprintf_r>
   146b4:	00050493          	mv	s1,a0
   146b8:	02055c63          	bgez	a0,146f0 <__sbprintf+0xb4>
   146bc:	01415783          	lhu	a5,20(sp)
   146c0:	0407f793          	andi	a5,a5,64
   146c4:	00078863          	beqz	a5,146d4 <__sbprintf+0x98>
   146c8:	00c45783          	lhu	a5,12(s0)
   146cc:	0407e793          	ori	a5,a5,64
   146d0:	00f41623          	sh	a5,12(s0)
   146d4:	47c12083          	lw	ra,1148(sp)
   146d8:	47812403          	lw	s0,1144(sp)
   146dc:	47012903          	lw	s2,1136(sp)
   146e0:	00048513          	mv	a0,s1
   146e4:	47412483          	lw	s1,1140(sp)
   146e8:	48010113          	addi	sp,sp,1152
   146ec:	00008067          	ret
   146f0:	00810593          	addi	a1,sp,8
   146f4:	00090513          	mv	a0,s2
   146f8:	0ed010ef          	jal	15fe4 <_fflush_r>
   146fc:	fc0500e3          	beqz	a0,146bc <__sbprintf+0x80>
   14700:	fff00493          	li	s1,-1
   14704:	fb9ff06f          	j	146bc <__sbprintf+0x80>

00014708 <__sprint_r.part.0>:
   14708:	0645a783          	lw	a5,100(a1)
   1470c:	fd010113          	addi	sp,sp,-48
   14710:	01612823          	sw	s6,16(sp)
   14714:	02112623          	sw	ra,44(sp)
   14718:	01279713          	slli	a4,a5,0x12
   1471c:	00060b13          	mv	s6,a2
   14720:	0e075863          	bgez	a4,14810 <__sprint_r.part.0+0x108>
   14724:	00862783          	lw	a5,8(a2)
   14728:	03212023          	sw	s2,32(sp)
   1472c:	01312e23          	sw	s3,28(sp)
   14730:	01512a23          	sw	s5,20(sp)
   14734:	01712623          	sw	s7,12(sp)
   14738:	00058913          	mv	s2,a1
   1473c:	00062b83          	lw	s7,0(a2)
   14740:	00050993          	mv	s3,a0
   14744:	fff00a93          	li	s5,-1
   14748:	0a078863          	beqz	a5,147f8 <__sprint_r.part.0+0xf0>
   1474c:	02812423          	sw	s0,40(sp)
   14750:	02912223          	sw	s1,36(sp)
   14754:	01412c23          	sw	s4,24(sp)
   14758:	01812423          	sw	s8,8(sp)
   1475c:	004bac03          	lw	s8,4(s7)
   14760:	000ba403          	lw	s0,0(s7)
   14764:	002c5a13          	srli	s4,s8,0x2
   14768:	060a0663          	beqz	s4,147d4 <__sprint_r.part.0+0xcc>
   1476c:	00000493          	li	s1,0
   14770:	00c0006f          	j	1477c <__sprint_r.part.0+0x74>
   14774:	00440413          	addi	s0,s0,4
   14778:	049a0c63          	beq	s4,s1,147d0 <__sprint_r.part.0+0xc8>
   1477c:	00042583          	lw	a1,0(s0)
   14780:	00090613          	mv	a2,s2
   14784:	00098513          	mv	a0,s3
   14788:	080020ef          	jal	16808 <_fputwc_r>
   1478c:	00148493          	addi	s1,s1,1
   14790:	ff5512e3          	bne	a0,s5,14774 <__sprint_r.part.0+0x6c>
   14794:	02812403          	lw	s0,40(sp)
   14798:	02412483          	lw	s1,36(sp)
   1479c:	02012903          	lw	s2,32(sp)
   147a0:	01c12983          	lw	s3,28(sp)
   147a4:	01812a03          	lw	s4,24(sp)
   147a8:	01412a83          	lw	s5,20(sp)
   147ac:	00c12b83          	lw	s7,12(sp)
   147b0:	00812c03          	lw	s8,8(sp)
   147b4:	fff00513          	li	a0,-1
   147b8:	02c12083          	lw	ra,44(sp)
   147bc:	000b2423          	sw	zero,8(s6)
   147c0:	000b2223          	sw	zero,4(s6)
   147c4:	01012b03          	lw	s6,16(sp)
   147c8:	03010113          	addi	sp,sp,48
   147cc:	00008067          	ret
   147d0:	008b2783          	lw	a5,8(s6)
   147d4:	ffcc7c13          	andi	s8,s8,-4
   147d8:	418787b3          	sub	a5,a5,s8
   147dc:	00fb2423          	sw	a5,8(s6)
   147e0:	008b8b93          	addi	s7,s7,8
   147e4:	f6079ce3          	bnez	a5,1475c <__sprint_r.part.0+0x54>
   147e8:	02812403          	lw	s0,40(sp)
   147ec:	02412483          	lw	s1,36(sp)
   147f0:	01812a03          	lw	s4,24(sp)
   147f4:	00812c03          	lw	s8,8(sp)
   147f8:	02012903          	lw	s2,32(sp)
   147fc:	01c12983          	lw	s3,28(sp)
   14800:	01412a83          	lw	s5,20(sp)
   14804:	00c12b83          	lw	s7,12(sp)
   14808:	00000513          	li	a0,0
   1480c:	fadff06f          	j	147b8 <__sprint_r.part.0+0xb0>
   14810:	0b5010ef          	jal	160c4 <__sfvwrite_r>
   14814:	02c12083          	lw	ra,44(sp)
   14818:	000b2423          	sw	zero,8(s6)
   1481c:	000b2223          	sw	zero,4(s6)
   14820:	01012b03          	lw	s6,16(sp)
   14824:	03010113          	addi	sp,sp,48
   14828:	00008067          	ret

0001482c <__sprint_r>:
   1482c:	00862703          	lw	a4,8(a2)
   14830:	00070463          	beqz	a4,14838 <__sprint_r+0xc>
   14834:	ed5ff06f          	j	14708 <__sprint_r.part.0>
   14838:	00062223          	sw	zero,4(a2)
   1483c:	00000513          	li	a0,0
   14840:	00008067          	ret

00014844 <_vfiprintf_r>:
   14844:	ed010113          	addi	sp,sp,-304
   14848:	11312e23          	sw	s3,284(sp)
   1484c:	11612823          	sw	s6,272(sp)
   14850:	11712623          	sw	s7,268(sp)
   14854:	11812423          	sw	s8,264(sp)
   14858:	12112623          	sw	ra,300(sp)
   1485c:	0fb12e23          	sw	s11,252(sp)
   14860:	00050c13          	mv	s8,a0
   14864:	00058b13          	mv	s6,a1
   14868:	00060993          	mv	s3,a2
   1486c:	00068b93          	mv	s7,a3
   14870:	00050863          	beqz	a0,14880 <_vfiprintf_r+0x3c>
   14874:	03452783          	lw	a5,52(a0)
   14878:	00079463          	bnez	a5,14880 <_vfiprintf_r+0x3c>
   1487c:	1e80106f          	j	15a64 <_vfiprintf_r+0x1220>
   14880:	00cb1783          	lh	a5,12(s6)
   14884:	01279713          	slli	a4,a5,0x12
   14888:	02074663          	bltz	a4,148b4 <_vfiprintf_r+0x70>
   1488c:	064b2703          	lw	a4,100(s6)
   14890:	00002637          	lui	a2,0x2
   14894:	ffffe6b7          	lui	a3,0xffffe
   14898:	00c7e7b3          	or	a5,a5,a2
   1489c:	fff68693          	addi	a3,a3,-1 # ffffdfff <__BSS_END__+0xfffdb2cf>
   148a0:	01079793          	slli	a5,a5,0x10
   148a4:	4107d793          	srai	a5,a5,0x10
   148a8:	00d77733          	and	a4,a4,a3
   148ac:	00fb1623          	sh	a5,12(s6)
   148b0:	06eb2223          	sw	a4,100(s6)
   148b4:	0087f713          	andi	a4,a5,8
   148b8:	12070e63          	beqz	a4,149f4 <_vfiprintf_r+0x1b0>
   148bc:	010b2703          	lw	a4,16(s6)
   148c0:	12070a63          	beqz	a4,149f4 <_vfiprintf_r+0x1b0>
   148c4:	01a7f793          	andi	a5,a5,26
   148c8:	00a00713          	li	a4,10
   148cc:	14e78663          	beq	a5,a4,14a18 <_vfiprintf_r+0x1d4>
   148d0:	11512a23          	sw	s5,276(sp)
   148d4:	00000d93          	li	s11,0
   148d8:	04c10a93          	addi	s5,sp,76
   148dc:	12812423          	sw	s0,296(sp)
   148e0:	11412c23          	sw	s4,280(sp)
   148e4:	11912223          	sw	s9,260(sp)
   148e8:	11a12023          	sw	s10,256(sp)
   148ec:	01812023          	sw	s8,0(sp)
   148f0:	12912223          	sw	s1,292(sp)
   148f4:	13212023          	sw	s2,288(sp)
   148f8:	05512023          	sw	s5,64(sp)
   148fc:	04012423          	sw	zero,72(sp)
   14900:	04012223          	sw	zero,68(sp)
   14904:	000a8413          	mv	s0,s5
   14908:	00012223          	sw	zero,4(sp)
   1490c:	00012a23          	sw	zero,20(sp)
   14910:	00012c23          	sw	zero,24(sp)
   14914:	00012e23          	sw	zero,28(sp)
   14918:	0000cd17          	auipc	s10,0xc
   1491c:	5f8d0d13          	addi	s10,s10,1528 # 20f10 <blanks.1+0x74>
   14920:	01000a13          	li	s4,16
   14924:	0000cc97          	auipc	s9,0xc
   14928:	758c8c93          	addi	s9,s9,1880 # 2107c <zeroes.0>
   1492c:	000d8c13          	mv	s8,s11
   14930:	000b8813          	mv	a6,s7
   14934:	0009c783          	lbu	a5,0(s3)
   14938:	32078863          	beqz	a5,14c68 <_vfiprintf_r+0x424>
   1493c:	00098493          	mv	s1,s3
   14940:	02500713          	li	a4,37
   14944:	3ce78863          	beq	a5,a4,14d14 <_vfiprintf_r+0x4d0>
   14948:	0014c783          	lbu	a5,1(s1)
   1494c:	00148493          	addi	s1,s1,1
   14950:	fe079ae3          	bnez	a5,14944 <_vfiprintf_r+0x100>
   14954:	41348933          	sub	s2,s1,s3
   14958:	31348863          	beq	s1,s3,14c68 <_vfiprintf_r+0x424>
   1495c:	04812703          	lw	a4,72(sp)
   14960:	04412783          	lw	a5,68(sp)
   14964:	01342023          	sw	s3,0(s0)
   14968:	00e90733          	add	a4,s2,a4
   1496c:	00178793          	addi	a5,a5,1
   14970:	01242223          	sw	s2,4(s0)
   14974:	04e12423          	sw	a4,72(sp)
   14978:	04f12223          	sw	a5,68(sp)
   1497c:	00700693          	li	a3,7
   14980:	00840413          	addi	s0,s0,8
   14984:	02f6d463          	bge	a3,a5,149ac <_vfiprintf_r+0x168>
   14988:	4e070ce3          	beqz	a4,15680 <_vfiprintf_r+0xe3c>
   1498c:	00012503          	lw	a0,0(sp)
   14990:	04010613          	addi	a2,sp,64
   14994:	000b0593          	mv	a1,s6
   14998:	01012423          	sw	a6,8(sp)
   1499c:	d6dff0ef          	jal	14708 <__sprint_r.part.0>
   149a0:	00812803          	lw	a6,8(sp)
   149a4:	0c051463          	bnez	a0,14a6c <_vfiprintf_r+0x228>
   149a8:	000a8413          	mv	s0,s5
   149ac:	0004c783          	lbu	a5,0(s1)
   149b0:	012c0c33          	add	s8,s8,s2
   149b4:	2a078a63          	beqz	a5,14c68 <_vfiprintf_r+0x424>
   149b8:	0014c703          	lbu	a4,1(s1)
   149bc:	00148993          	addi	s3,s1,1
   149c0:	02010da3          	sb	zero,59(sp)
   149c4:	fff00493          	li	s1,-1
   149c8:	00000b93          	li	s7,0
   149cc:	00000d93          	li	s11,0
   149d0:	05a00913          	li	s2,90
   149d4:	00198993          	addi	s3,s3,1
   149d8:	fe070793          	addi	a5,a4,-32
   149dc:	10f96c63          	bltu	s2,a5,14af4 <_vfiprintf_r+0x2b0>
   149e0:	00279793          	slli	a5,a5,0x2
   149e4:	01a787b3          	add	a5,a5,s10
   149e8:	0007a783          	lw	a5,0(a5)
   149ec:	01a787b3          	add	a5,a5,s10
   149f0:	00078067          	jr	a5
   149f4:	000b0593          	mv	a1,s6
   149f8:	000c0513          	mv	a0,s8
   149fc:	3d9010ef          	jal	165d4 <__swsetup_r>
   14a00:	00050463          	beqz	a0,14a08 <_vfiprintf_r+0x1c4>
   14a04:	16c0106f          	j	15b70 <_vfiprintf_r+0x132c>
   14a08:	00cb1783          	lh	a5,12(s6)
   14a0c:	00a00713          	li	a4,10
   14a10:	01a7f793          	andi	a5,a5,26
   14a14:	eae79ee3          	bne	a5,a4,148d0 <_vfiprintf_r+0x8c>
   14a18:	00eb1783          	lh	a5,14(s6)
   14a1c:	ea07cae3          	bltz	a5,148d0 <_vfiprintf_r+0x8c>
   14a20:	12c12083          	lw	ra,300(sp)
   14a24:	0fc12d83          	lw	s11,252(sp)
   14a28:	000b8693          	mv	a3,s7
   14a2c:	00098613          	mv	a2,s3
   14a30:	10c12b83          	lw	s7,268(sp)
   14a34:	11c12983          	lw	s3,284(sp)
   14a38:	000b0593          	mv	a1,s6
   14a3c:	000c0513          	mv	a0,s8
   14a40:	11012b03          	lw	s6,272(sp)
   14a44:	10812c03          	lw	s8,264(sp)
   14a48:	13010113          	addi	sp,sp,304
   14a4c:	16c0106f          	j	15bb8 <__sbprintf>
   14a50:	00012503          	lw	a0,0(sp)
   14a54:	04010613          	addi	a2,sp,64
   14a58:	000b0593          	mv	a1,s6
   14a5c:	01012423          	sw	a6,8(sp)
   14a60:	ca9ff0ef          	jal	14708 <__sprint_r.part.0>
   14a64:	00812803          	lw	a6,8(sp)
   14a68:	1e050863          	beqz	a0,14c58 <_vfiprintf_r+0x414>
   14a6c:	000c0d93          	mv	s11,s8
   14a70:	00cb5783          	lhu	a5,12(s6)
   14a74:	12812403          	lw	s0,296(sp)
   14a78:	12412483          	lw	s1,292(sp)
   14a7c:	0407f793          	andi	a5,a5,64
   14a80:	12012903          	lw	s2,288(sp)
   14a84:	11812a03          	lw	s4,280(sp)
   14a88:	11412a83          	lw	s5,276(sp)
   14a8c:	10412c83          	lw	s9,260(sp)
   14a90:	10012d03          	lw	s10,256(sp)
   14a94:	00078463          	beqz	a5,14a9c <_vfiprintf_r+0x258>
   14a98:	0d80106f          	j	15b70 <_vfiprintf_r+0x132c>
   14a9c:	12c12083          	lw	ra,300(sp)
   14aa0:	11c12983          	lw	s3,284(sp)
   14aa4:	11012b03          	lw	s6,272(sp)
   14aa8:	10c12b83          	lw	s7,268(sp)
   14aac:	10812c03          	lw	s8,264(sp)
   14ab0:	000d8513          	mv	a0,s11
   14ab4:	0fc12d83          	lw	s11,252(sp)
   14ab8:	13010113          	addi	sp,sp,304
   14abc:	00008067          	ret
   14ac0:	00000b93          	li	s7,0
   14ac4:	fd070793          	addi	a5,a4,-48
   14ac8:	00900613          	li	a2,9
   14acc:	0009c703          	lbu	a4,0(s3)
   14ad0:	002b9693          	slli	a3,s7,0x2
   14ad4:	01768bb3          	add	s7,a3,s7
   14ad8:	001b9b93          	slli	s7,s7,0x1
   14adc:	01778bb3          	add	s7,a5,s7
   14ae0:	fd070793          	addi	a5,a4,-48
   14ae4:	00198993          	addi	s3,s3,1
   14ae8:	fef672e3          	bgeu	a2,a5,14acc <_vfiprintf_r+0x288>
   14aec:	fe070793          	addi	a5,a4,-32
   14af0:	eef978e3          	bgeu	s2,a5,149e0 <_vfiprintf_r+0x19c>
   14af4:	16070a63          	beqz	a4,14c68 <_vfiprintf_r+0x424>
   14af8:	08e10623          	sb	a4,140(sp)
   14afc:	02010da3          	sb	zero,59(sp)
   14b00:	00100693          	li	a3,1
   14b04:	00100893          	li	a7,1
   14b08:	08c10913          	addi	s2,sp,140
   14b0c:	00000493          	li	s1,0
   14b10:	00000f13          	li	t5,0
   14b14:	04412603          	lw	a2,68(sp)
   14b18:	084dff93          	andi	t6,s11,132
   14b1c:	04812783          	lw	a5,72(sp)
   14b20:	00160593          	addi	a1,a2,1 # 2001 <exit-0xe0b3>
   14b24:	00058e13          	mv	t3,a1
   14b28:	000f9663          	bnez	t6,14b34 <_vfiprintf_r+0x2f0>
   14b2c:	40db8733          	sub	a4,s7,a3
   14b30:	1ee04ae3          	bgtz	a4,15524 <_vfiprintf_r+0xce0>
   14b34:	03b14703          	lbu	a4,59(sp)
   14b38:	02070a63          	beqz	a4,14b6c <_vfiprintf_r+0x328>
   14b3c:	03b10713          	addi	a4,sp,59
   14b40:	00178793          	addi	a5,a5,1
   14b44:	00e42023          	sw	a4,0(s0)
   14b48:	00100713          	li	a4,1
   14b4c:	00e42223          	sw	a4,4(s0)
   14b50:	04f12423          	sw	a5,72(sp)
   14b54:	05c12223          	sw	t3,68(sp)
   14b58:	00700713          	li	a4,7
   14b5c:	0fc748e3          	blt	a4,t3,1544c <_vfiprintf_r+0xc08>
   14b60:	000e0613          	mv	a2,t3
   14b64:	00840413          	addi	s0,s0,8
   14b68:	001e0e13          	addi	t3,t3,1
   14b6c:	060f0863          	beqz	t5,14bdc <_vfiprintf_r+0x398>
   14b70:	03c10713          	addi	a4,sp,60
   14b74:	00278793          	addi	a5,a5,2
   14b78:	00e42023          	sw	a4,0(s0)
   14b7c:	00200713          	li	a4,2
   14b80:	00e42223          	sw	a4,4(s0)
   14b84:	04f12423          	sw	a5,72(sp)
   14b88:	05c12223          	sw	t3,68(sp)
   14b8c:	00700713          	li	a4,7
   14b90:	13c756e3          	bge	a4,t3,154bc <_vfiprintf_r+0xc78>
   14b94:	34078ce3          	beqz	a5,156ec <_vfiprintf_r+0xea8>
   14b98:	00012503          	lw	a0,0(sp)
   14b9c:	04010613          	addi	a2,sp,64
   14ba0:	000b0593          	mv	a1,s6
   14ba4:	03012023          	sw	a6,32(sp)
   14ba8:	00d12823          	sw	a3,16(sp)
   14bac:	01112623          	sw	a7,12(sp)
   14bb0:	01f12423          	sw	t6,8(sp)
   14bb4:	b55ff0ef          	jal	14708 <__sprint_r.part.0>
   14bb8:	ea051ae3          	bnez	a0,14a6c <_vfiprintf_r+0x228>
   14bbc:	04412603          	lw	a2,68(sp)
   14bc0:	04812783          	lw	a5,72(sp)
   14bc4:	02012803          	lw	a6,32(sp)
   14bc8:	01012683          	lw	a3,16(sp)
   14bcc:	00c12883          	lw	a7,12(sp)
   14bd0:	00812f83          	lw	t6,8(sp)
   14bd4:	000a8413          	mv	s0,s5
   14bd8:	00160e13          	addi	t3,a2,1
   14bdc:	08000713          	li	a4,128
   14be0:	64ef8a63          	beq	t6,a4,15234 <_vfiprintf_r+0x9f0>
   14be4:	411484b3          	sub	s1,s1,a7
   14be8:	76904a63          	bgtz	s1,1535c <_vfiprintf_r+0xb18>
   14bec:	00f887b3          	add	a5,a7,a5
   14bf0:	01242023          	sw	s2,0(s0)
   14bf4:	01142223          	sw	a7,4(s0)
   14bf8:	04f12423          	sw	a5,72(sp)
   14bfc:	05c12223          	sw	t3,68(sp)
   14c00:	00700713          	li	a4,7
   14c04:	61c75c63          	bge	a4,t3,1521c <_vfiprintf_r+0x9d8>
   14c08:	10078e63          	beqz	a5,14d24 <_vfiprintf_r+0x4e0>
   14c0c:	00012503          	lw	a0,0(sp)
   14c10:	04010613          	addi	a2,sp,64
   14c14:	000b0593          	mv	a1,s6
   14c18:	01012623          	sw	a6,12(sp)
   14c1c:	00d12423          	sw	a3,8(sp)
   14c20:	ae9ff0ef          	jal	14708 <__sprint_r.part.0>
   14c24:	e40514e3          	bnez	a0,14a6c <_vfiprintf_r+0x228>
   14c28:	04812783          	lw	a5,72(sp)
   14c2c:	00c12803          	lw	a6,12(sp)
   14c30:	00812683          	lw	a3,8(sp)
   14c34:	000a8413          	mv	s0,s5
   14c38:	004dfd93          	andi	s11,s11,4
   14c3c:	000d8663          	beqz	s11,14c48 <_vfiprintf_r+0x404>
   14c40:	40db84b3          	sub	s1,s7,a3
   14c44:	0e904c63          	bgtz	s1,14d3c <_vfiprintf_r+0x4f8>
   14c48:	00dbd463          	bge	s7,a3,14c50 <_vfiprintf_r+0x40c>
   14c4c:	00068b93          	mv	s7,a3
   14c50:	017c0c33          	add	s8,s8,s7
   14c54:	de079ee3          	bnez	a5,14a50 <_vfiprintf_r+0x20c>
   14c58:	0009c783          	lbu	a5,0(s3)
   14c5c:	04012223          	sw	zero,68(sp)
   14c60:	000a8413          	mv	s0,s5
   14c64:	cc079ce3          	bnez	a5,1493c <_vfiprintf_r+0xf8>
   14c68:	04812783          	lw	a5,72(sp)
   14c6c:	000c0d93          	mv	s11,s8
   14c70:	00012c03          	lw	s8,0(sp)
   14c74:	de078ee3          	beqz	a5,14a70 <_vfiprintf_r+0x22c>
   14c78:	04010613          	addi	a2,sp,64
   14c7c:	000b0593          	mv	a1,s6
   14c80:	000c0513          	mv	a0,s8
   14c84:	a85ff0ef          	jal	14708 <__sprint_r.part.0>
   14c88:	de9ff06f          	j	14a70 <_vfiprintf_r+0x22c>
   14c8c:	00082b83          	lw	s7,0(a6)
   14c90:	00480813          	addi	a6,a6,4
   14c94:	2a0bc663          	bltz	s7,14f40 <_vfiprintf_r+0x6fc>
   14c98:	0009c703          	lbu	a4,0(s3)
   14c9c:	d39ff06f          	j	149d4 <_vfiprintf_r+0x190>
   14ca0:	0009c703          	lbu	a4,0(s3)
   14ca4:	020ded93          	ori	s11,s11,32
   14ca8:	d2dff06f          	j	149d4 <_vfiprintf_r+0x190>
   14cac:	010ded93          	ori	s11,s11,16
   14cb0:	020df793          	andi	a5,s11,32
   14cb4:	16078e63          	beqz	a5,14e30 <_vfiprintf_r+0x5ec>
   14cb8:	00780813          	addi	a6,a6,7
   14cbc:	ff887813          	andi	a6,a6,-8
   14cc0:	00482703          	lw	a4,4(a6)
   14cc4:	00082783          	lw	a5,0(a6)
   14cc8:	00880813          	addi	a6,a6,8
   14ccc:	00070893          	mv	a7,a4
   14cd0:	18074663          	bltz	a4,14e5c <_vfiprintf_r+0x618>
   14cd4:	1a04c463          	bltz	s1,14e7c <_vfiprintf_r+0x638>
   14cd8:	0117e733          	or	a4,a5,a7
   14cdc:	f7fdfd93          	andi	s11,s11,-129
   14ce0:	18071e63          	bnez	a4,14e7c <_vfiprintf_r+0x638>
   14ce4:	660492e3          	bnez	s1,15b48 <_vfiprintf_r+0x1304>
   14ce8:	03b14783          	lbu	a5,59(sp)
   14cec:	00000693          	li	a3,0
   14cf0:	00000893          	li	a7,0
   14cf4:	0f010913          	addi	s2,sp,240
   14cf8:	00078463          	beqz	a5,14d00 <_vfiprintf_r+0x4bc>
   14cfc:	00168693          	addi	a3,a3,1
   14d00:	002dff13          	andi	t5,s11,2
   14d04:	e00f08e3          	beqz	t5,14b14 <_vfiprintf_r+0x2d0>
   14d08:	00268693          	addi	a3,a3,2
   14d0c:	00200f13          	li	t5,2
   14d10:	e05ff06f          	j	14b14 <_vfiprintf_r+0x2d0>
   14d14:	41348933          	sub	s2,s1,s3
   14d18:	c53492e3          	bne	s1,s3,1495c <_vfiprintf_r+0x118>
   14d1c:	0004c783          	lbu	a5,0(s1)
   14d20:	c95ff06f          	j	149b4 <_vfiprintf_r+0x170>
   14d24:	04012223          	sw	zero,68(sp)
   14d28:	004dfd93          	andi	s11,s11,4
   14d2c:	100d80e3          	beqz	s11,1562c <_vfiprintf_r+0xde8>
   14d30:	40db84b3          	sub	s1,s7,a3
   14d34:	0e905ce3          	blez	s1,1562c <_vfiprintf_r+0xde8>
   14d38:	000a8413          	mv	s0,s5
   14d3c:	01000713          	li	a4,16
   14d40:	04412603          	lw	a2,68(sp)
   14d44:	60975ee3          	bge	a4,s1,15b60 <_vfiprintf_r+0x131c>
   14d48:	0000ce17          	auipc	t3,0xc
   14d4c:	344e0e13          	addi	t3,t3,836 # 2108c <blanks.1>
   14d50:	00d12423          	sw	a3,8(sp)
   14d54:	01000913          	li	s2,16
   14d58:	00040693          	mv	a3,s0
   14d5c:	00700d93          	li	s11,7
   14d60:	00048413          	mv	s0,s1
   14d64:	01012623          	sw	a6,12(sp)
   14d68:	000e0493          	mv	s1,t3
   14d6c:	0180006f          	j	14d84 <_vfiprintf_r+0x540>
   14d70:	00260593          	addi	a1,a2,2
   14d74:	00868693          	addi	a3,a3,8
   14d78:	00070613          	mv	a2,a4
   14d7c:	ff040413          	addi	s0,s0,-16
   14d80:	04895863          	bge	s2,s0,14dd0 <_vfiprintf_r+0x58c>
   14d84:	01078793          	addi	a5,a5,16
   14d88:	00160713          	addi	a4,a2,1
   14d8c:	0096a023          	sw	s1,0(a3)
   14d90:	0126a223          	sw	s2,4(a3)
   14d94:	04f12423          	sw	a5,72(sp)
   14d98:	04e12223          	sw	a4,68(sp)
   14d9c:	fceddae3          	bge	s11,a4,14d70 <_vfiprintf_r+0x52c>
   14da0:	48078263          	beqz	a5,15224 <_vfiprintf_r+0x9e0>
   14da4:	00012503          	lw	a0,0(sp)
   14da8:	04010613          	addi	a2,sp,64
   14dac:	000b0593          	mv	a1,s6
   14db0:	959ff0ef          	jal	14708 <__sprint_r.part.0>
   14db4:	ca051ce3          	bnez	a0,14a6c <_vfiprintf_r+0x228>
   14db8:	04412603          	lw	a2,68(sp)
   14dbc:	ff040413          	addi	s0,s0,-16
   14dc0:	04812783          	lw	a5,72(sp)
   14dc4:	000a8693          	mv	a3,s5
   14dc8:	00160593          	addi	a1,a2,1
   14dcc:	fa894ce3          	blt	s2,s0,14d84 <_vfiprintf_r+0x540>
   14dd0:	00048e13          	mv	t3,s1
   14dd4:	00c12803          	lw	a6,12(sp)
   14dd8:	00040493          	mv	s1,s0
   14ddc:	00068413          	mv	s0,a3
   14de0:	00812683          	lw	a3,8(sp)
   14de4:	009787b3          	add	a5,a5,s1
   14de8:	01c42023          	sw	t3,0(s0)
   14dec:	00942223          	sw	s1,4(s0)
   14df0:	04f12423          	sw	a5,72(sp)
   14df4:	04b12223          	sw	a1,68(sp)
   14df8:	00700713          	li	a4,7
   14dfc:	e4b756e3          	bge	a4,a1,14c48 <_vfiprintf_r+0x404>
   14e00:	020786e3          	beqz	a5,1562c <_vfiprintf_r+0xde8>
   14e04:	00012503          	lw	a0,0(sp)
   14e08:	04010613          	addi	a2,sp,64
   14e0c:	000b0593          	mv	a1,s6
   14e10:	01012623          	sw	a6,12(sp)
   14e14:	00d12423          	sw	a3,8(sp)
   14e18:	8f1ff0ef          	jal	14708 <__sprint_r.part.0>
   14e1c:	c40518e3          	bnez	a0,14a6c <_vfiprintf_r+0x228>
   14e20:	04812783          	lw	a5,72(sp)
   14e24:	00c12803          	lw	a6,12(sp)
   14e28:	00812683          	lw	a3,8(sp)
   14e2c:	e1dff06f          	j	14c48 <_vfiprintf_r+0x404>
   14e30:	010df713          	andi	a4,s11,16
   14e34:	00082783          	lw	a5,0(a6)
   14e38:	00480813          	addi	a6,a6,4
   14e3c:	0e071c63          	bnez	a4,14f34 <_vfiprintf_r+0x6f0>
   14e40:	040df713          	andi	a4,s11,64
   14e44:	0e070463          	beqz	a4,14f2c <_vfiprintf_r+0x6e8>
   14e48:	01079793          	slli	a5,a5,0x10
   14e4c:	4107d793          	srai	a5,a5,0x10
   14e50:	41f7d893          	srai	a7,a5,0x1f
   14e54:	00088713          	mv	a4,a7
   14e58:	e6075ee3          	bgez	a4,14cd4 <_vfiprintf_r+0x490>
   14e5c:	02d00693          	li	a3,45
   14e60:	00f03733          	snez	a4,a5
   14e64:	411008b3          	neg	a7,a7
   14e68:	02d10da3          	sb	a3,59(sp)
   14e6c:	40e888b3          	sub	a7,a7,a4
   14e70:	40f007b3          	neg	a5,a5
   14e74:	0004c463          	bltz	s1,14e7c <_vfiprintf_r+0x638>
   14e78:	f7fdfd93          	andi	s11,s11,-129
   14e7c:	0c089ae3          	bnez	a7,15750 <_vfiprintf_r+0xf0c>
   14e80:	00900713          	li	a4,9
   14e84:	0cf766e3          	bltu	a4,a5,15750 <_vfiprintf_r+0xf0c>
   14e88:	03078793          	addi	a5,a5,48
   14e8c:	0ff7f793          	zext.b	a5,a5
   14e90:	0ef107a3          	sb	a5,239(sp)
   14e94:	00048693          	mv	a3,s1
   14e98:	00904463          	bgtz	s1,14ea0 <_vfiprintf_r+0x65c>
   14e9c:	00100693          	li	a3,1
   14ea0:	00100893          	li	a7,1
   14ea4:	0ef10913          	addi	s2,sp,239
   14ea8:	03b14783          	lbu	a5,59(sp)
   14eac:	e40798e3          	bnez	a5,14cfc <_vfiprintf_r+0x4b8>
   14eb0:	e51ff06f          	j	14d00 <_vfiprintf_r+0x4bc>
   14eb4:	00082903          	lw	s2,0(a6)
   14eb8:	02010da3          	sb	zero,59(sp)
   14ebc:	00480813          	addi	a6,a6,4
   14ec0:	3c090ee3          	beqz	s2,15a9c <_vfiprintf_r+0x1258>
   14ec4:	01012423          	sw	a6,8(sp)
   14ec8:	2604cee3          	bltz	s1,15944 <_vfiprintf_r+0x1100>
   14ecc:	00048613          	mv	a2,s1
   14ed0:	00000593          	li	a1,0
   14ed4:	00090513          	mv	a0,s2
   14ed8:	1d9010ef          	jal	168b0 <memchr>
   14edc:	00812803          	lw	a6,8(sp)
   14ee0:	00048893          	mv	a7,s1
   14ee4:	00050463          	beqz	a0,14eec <_vfiprintf_r+0x6a8>
   14ee8:	412508b3          	sub	a7,a0,s2
   14eec:	03b14783          	lbu	a5,59(sp)
   14ef0:	fff8c693          	not	a3,a7
   14ef4:	41f6d693          	srai	a3,a3,0x1f
   14ef8:	00d8f6b3          	and	a3,a7,a3
   14efc:	00000493          	li	s1,0
   14f00:	00000f13          	li	t5,0
   14f04:	de079ce3          	bnez	a5,14cfc <_vfiprintf_r+0x4b8>
   14f08:	c0dff06f          	j	14b14 <_vfiprintf_r+0x2d0>
   14f0c:	00082783          	lw	a5,0(a6)
   14f10:	02010da3          	sb	zero,59(sp)
   14f14:	00480813          	addi	a6,a6,4
   14f18:	08f10623          	sb	a5,140(sp)
   14f1c:	00100693          	li	a3,1
   14f20:	00100893          	li	a7,1
   14f24:	08c10913          	addi	s2,sp,140
   14f28:	be5ff06f          	j	14b0c <_vfiprintf_r+0x2c8>
   14f2c:	200df713          	andi	a4,s11,512
   14f30:	3e0716e3          	bnez	a4,15b1c <_vfiprintf_r+0x12d8>
   14f34:	41f7d893          	srai	a7,a5,0x1f
   14f38:	00088713          	mv	a4,a7
   14f3c:	d95ff06f          	j	14cd0 <_vfiprintf_r+0x48c>
   14f40:	41700bb3          	neg	s7,s7
   14f44:	0009c703          	lbu	a4,0(s3)
   14f48:	004ded93          	ori	s11,s11,4
   14f4c:	a89ff06f          	j	149d4 <_vfiprintf_r+0x190>
   14f50:	02b00793          	li	a5,43
   14f54:	0009c703          	lbu	a4,0(s3)
   14f58:	02f10da3          	sb	a5,59(sp)
   14f5c:	a79ff06f          	j	149d4 <_vfiprintf_r+0x190>
   14f60:	0009c703          	lbu	a4,0(s3)
   14f64:	080ded93          	ori	s11,s11,128
   14f68:	a6dff06f          	j	149d4 <_vfiprintf_r+0x190>
   14f6c:	0009c703          	lbu	a4,0(s3)
   14f70:	02a00793          	li	a5,42
   14f74:	00198613          	addi	a2,s3,1
   14f78:	40f708e3          	beq	a4,a5,15b88 <_vfiprintf_r+0x1344>
   14f7c:	fd070793          	addi	a5,a4,-48
   14f80:	00900693          	li	a3,9
   14f84:	00000493          	li	s1,0
   14f88:	00900593          	li	a1,9
   14f8c:	02f6e463          	bltu	a3,a5,14fb4 <_vfiprintf_r+0x770>
   14f90:	00064703          	lbu	a4,0(a2)
   14f94:	00249693          	slli	a3,s1,0x2
   14f98:	009684b3          	add	s1,a3,s1
   14f9c:	00149493          	slli	s1,s1,0x1
   14fa0:	00f484b3          	add	s1,s1,a5
   14fa4:	fd070793          	addi	a5,a4,-48
   14fa8:	00160613          	addi	a2,a2,1
   14fac:	fef5f2e3          	bgeu	a1,a5,14f90 <_vfiprintf_r+0x74c>
   14fb0:	0e04c6e3          	bltz	s1,1589c <_vfiprintf_r+0x1058>
   14fb4:	00060993          	mv	s3,a2
   14fb8:	a21ff06f          	j	149d8 <_vfiprintf_r+0x194>
   14fbc:	00012503          	lw	a0,0(sp)
   14fc0:	01012423          	sw	a6,8(sp)
   14fc4:	365010ef          	jal	16b28 <_localeconv_r>
   14fc8:	00452783          	lw	a5,4(a0)
   14fcc:	00078513          	mv	a0,a5
   14fd0:	00f12e23          	sw	a5,28(sp)
   14fd4:	f89fb0ef          	jal	10f5c <strlen>
   14fd8:	00050793          	mv	a5,a0
   14fdc:	00012503          	lw	a0,0(sp)
   14fe0:	00f12c23          	sw	a5,24(sp)
   14fe4:	345010ef          	jal	16b28 <_localeconv_r>
   14fe8:	00852703          	lw	a4,8(a0)
   14fec:	01812783          	lw	a5,24(sp)
   14ff0:	00812803          	lw	a6,8(sp)
   14ff4:	00e12a23          	sw	a4,20(sp)
   14ff8:	ca0780e3          	beqz	a5,14c98 <_vfiprintf_r+0x454>
   14ffc:	01412783          	lw	a5,20(sp)
   15000:	0009c703          	lbu	a4,0(s3)
   15004:	9c0788e3          	beqz	a5,149d4 <_vfiprintf_r+0x190>
   15008:	0007c783          	lbu	a5,0(a5)
   1500c:	9c0784e3          	beqz	a5,149d4 <_vfiprintf_r+0x190>
   15010:	400ded93          	ori	s11,s11,1024
   15014:	9c1ff06f          	j	149d4 <_vfiprintf_r+0x190>
   15018:	0009c703          	lbu	a4,0(s3)
   1501c:	001ded93          	ori	s11,s11,1
   15020:	9b5ff06f          	j	149d4 <_vfiprintf_r+0x190>
   15024:	03b14783          	lbu	a5,59(sp)
   15028:	0009c703          	lbu	a4,0(s3)
   1502c:	9a0794e3          	bnez	a5,149d4 <_vfiprintf_r+0x190>
   15030:	02000793          	li	a5,32
   15034:	02f10da3          	sb	a5,59(sp)
   15038:	99dff06f          	j	149d4 <_vfiprintf_r+0x190>
   1503c:	010de713          	ori	a4,s11,16
   15040:	02077793          	andi	a5,a4,32
   15044:	66078863          	beqz	a5,156b4 <_vfiprintf_r+0xe70>
   15048:	00780813          	addi	a6,a6,7
   1504c:	ff887813          	andi	a6,a6,-8
   15050:	00082783          	lw	a5,0(a6)
   15054:	00482603          	lw	a2,4(a6)
   15058:	00880813          	addi	a6,a6,8
   1505c:	02010da3          	sb	zero,59(sp)
   15060:	bff77d93          	andi	s11,a4,-1025
   15064:	0c04c063          	bltz	s1,15124 <_vfiprintf_r+0x8e0>
   15068:	00c7e6b3          	or	a3,a5,a2
   1506c:	b7f77713          	andi	a4,a4,-1153
   15070:	1e0692e3          	bnez	a3,15a54 <_vfiprintf_r+0x1210>
   15074:	000d8693          	mv	a3,s11
   15078:	00000793          	li	a5,0
   1507c:	00070d93          	mv	s11,a4
   15080:	08049663          	bnez	s1,1510c <_vfiprintf_r+0x8c8>
   15084:	c60792e3          	bnez	a5,14ce8 <_vfiprintf_r+0x4a4>
   15088:	0016f893          	andi	a7,a3,1
   1508c:	7c088063          	beqz	a7,1584c <_vfiprintf_r+0x1008>
   15090:	03000793          	li	a5,48
   15094:	0ef107a3          	sb	a5,239(sp)
   15098:	00088693          	mv	a3,a7
   1509c:	0ef10913          	addi	s2,sp,239
   150a0:	e09ff06f          	j	14ea8 <_vfiprintf_r+0x664>
   150a4:	0009c703          	lbu	a4,0(s3)
   150a8:	06c00793          	li	a5,108
   150ac:	1cf708e3          	beq	a4,a5,15a7c <_vfiprintf_r+0x1238>
   150b0:	010ded93          	ori	s11,s11,16
   150b4:	921ff06f          	j	149d4 <_vfiprintf_r+0x190>
   150b8:	0009c703          	lbu	a4,0(s3)
   150bc:	06800793          	li	a5,104
   150c0:	1af706e3          	beq	a4,a5,15a6c <_vfiprintf_r+0x1228>
   150c4:	040ded93          	ori	s11,s11,64
   150c8:	90dff06f          	j	149d4 <_vfiprintf_r+0x190>
   150cc:	010de693          	ori	a3,s11,16
   150d0:	0206f793          	andi	a5,a3,32
   150d4:	5a078c63          	beqz	a5,1568c <_vfiprintf_r+0xe48>
   150d8:	00780813          	addi	a6,a6,7
   150dc:	ff887813          	andi	a6,a6,-8
   150e0:	00082783          	lw	a5,0(a6)
   150e4:	00482883          	lw	a7,4(a6)
   150e8:	00880813          	addi	a6,a6,8
   150ec:	02010da3          	sb	zero,59(sp)
   150f0:	00068d93          	mv	s11,a3
   150f4:	d804c4e3          	bltz	s1,14e7c <_vfiprintf_r+0x638>
   150f8:	0117e733          	or	a4,a5,a7
   150fc:	f7f6fd93          	andi	s11,a3,-129
   15100:	d6071ee3          	bnez	a4,14e7c <_vfiprintf_r+0x638>
   15104:	00100793          	li	a5,1
   15108:	f6048ee3          	beqz	s1,15084 <_vfiprintf_r+0x840>
   1510c:	00100713          	li	a4,1
   15110:	22e78ce3          	beq	a5,a4,15b48 <_vfiprintf_r+0x1304>
   15114:	00200713          	li	a4,2
   15118:	1ae78ae3          	beq	a5,a4,15acc <_vfiprintf_r+0x1288>
   1511c:	00000793          	li	a5,0
   15120:	00000613          	li	a2,0
   15124:	0f010913          	addi	s2,sp,240
   15128:	01d61693          	slli	a3,a2,0x1d
   1512c:	0077f713          	andi	a4,a5,7
   15130:	0037d793          	srli	a5,a5,0x3
   15134:	03070713          	addi	a4,a4,48
   15138:	00f6e7b3          	or	a5,a3,a5
   1513c:	00365613          	srli	a2,a2,0x3
   15140:	fee90fa3          	sb	a4,-1(s2)
   15144:	00c7e6b3          	or	a3,a5,a2
   15148:	00090593          	mv	a1,s2
   1514c:	fff90913          	addi	s2,s2,-1
   15150:	fc069ce3          	bnez	a3,15128 <_vfiprintf_r+0x8e4>
   15154:	001df793          	andi	a5,s11,1
   15158:	3a078a63          	beqz	a5,1550c <_vfiprintf_r+0xcc8>
   1515c:	03000793          	li	a5,48
   15160:	3af70663          	beq	a4,a5,1550c <_vfiprintf_r+0xcc8>
   15164:	ffe58593          	addi	a1,a1,-2
   15168:	fef90fa3          	sb	a5,-1(s2)
   1516c:	0f010793          	addi	a5,sp,240
   15170:	40b788b3          	sub	a7,a5,a1
   15174:	00048693          	mv	a3,s1
   15178:	7114cc63          	blt	s1,a7,15890 <_vfiprintf_r+0x104c>
   1517c:	00058913          	mv	s2,a1
   15180:	d29ff06f          	j	14ea8 <_vfiprintf_r+0x664>
   15184:	00008737          	lui	a4,0x8
   15188:	83070713          	addi	a4,a4,-2000 # 7830 <exit-0x8884>
   1518c:	02e11e23          	sh	a4,60(sp)
   15190:	0000c717          	auipc	a4,0xc
   15194:	9b870713          	addi	a4,a4,-1608 # 20b48 <_exit+0x198>
   15198:	00082783          	lw	a5,0(a6)
   1519c:	00000613          	li	a2,0
   151a0:	002ded93          	ori	s11,s11,2
   151a4:	00480813          	addi	a6,a6,4
   151a8:	00e12223          	sw	a4,4(sp)
   151ac:	02010da3          	sb	zero,59(sp)
   151b0:	3204c463          	bltz	s1,154d8 <_vfiprintf_r+0xc94>
   151b4:	00c7e733          	or	a4,a5,a2
   151b8:	f7fdf593          	andi	a1,s11,-129
   151bc:	30071863          	bnez	a4,154cc <_vfiprintf_r+0xc88>
   151c0:	000d8693          	mv	a3,s11
   151c4:	00200793          	li	a5,2
   151c8:	00058d93          	mv	s11,a1
   151cc:	eb5ff06f          	j	15080 <_vfiprintf_r+0x83c>
   151d0:	020df793          	andi	a5,s11,32
   151d4:	68079a63          	bnez	a5,15868 <_vfiprintf_r+0x1024>
   151d8:	010df793          	andi	a5,s11,16
   151dc:	0a0798e3          	bnez	a5,15a8c <_vfiprintf_r+0x1248>
   151e0:	040df793          	andi	a5,s11,64
   151e4:	120794e3          	bnez	a5,15b0c <_vfiprintf_r+0x12c8>
   151e8:	200dfd93          	andi	s11,s11,512
   151ec:	0a0d80e3          	beqz	s11,15a8c <_vfiprintf_r+0x1248>
   151f0:	00082783          	lw	a5,0(a6)
   151f4:	00480813          	addi	a6,a6,4
   151f8:	01878023          	sb	s8,0(a5)
   151fc:	f38ff06f          	j	14934 <_vfiprintf_r+0xf0>
   15200:	00100713          	li	a4,1
   15204:	00088793          	mv	a5,a7
   15208:	05212623          	sw	s2,76(sp)
   1520c:	05112823          	sw	a7,80(sp)
   15210:	05112423          	sw	a7,72(sp)
   15214:	04e12223          	sw	a4,68(sp)
   15218:	000a8413          	mv	s0,s5
   1521c:	00840413          	addi	s0,s0,8
   15220:	a19ff06f          	j	14c38 <_vfiprintf_r+0x3f4>
   15224:	00100593          	li	a1,1
   15228:	00000613          	li	a2,0
   1522c:	000a8693          	mv	a3,s5
   15230:	b4dff06f          	j	14d7c <_vfiprintf_r+0x538>
   15234:	40db8f33          	sub	t5,s7,a3
   15238:	9be056e3          	blez	t5,14be4 <_vfiprintf_r+0x3a0>
   1523c:	01000713          	li	a4,16
   15240:	13e75ce3          	bge	a4,t5,15b78 <_vfiprintf_r+0x1334>
   15244:	0000ce97          	auipc	t4,0xc
   15248:	e38e8e93          	addi	t4,t4,-456 # 2107c <zeroes.0>
   1524c:	00912623          	sw	s1,12(sp)
   15250:	00d12823          	sw	a3,16(sp)
   15254:	01000e13          	li	t3,16
   15258:	00040693          	mv	a3,s0
   1525c:	00700f93          	li	t6,7
   15260:	01112423          	sw	a7,8(sp)
   15264:	000f0413          	mv	s0,t5
   15268:	03012023          	sw	a6,32(sp)
   1526c:	000e8493          	mv	s1,t4
   15270:	0180006f          	j	15288 <_vfiprintf_r+0xa44>
   15274:	00260593          	addi	a1,a2,2
   15278:	00868693          	addi	a3,a3,8
   1527c:	00070613          	mv	a2,a4
   15280:	ff040413          	addi	s0,s0,-16
   15284:	048e5c63          	bge	t3,s0,152dc <_vfiprintf_r+0xa98>
   15288:	01078793          	addi	a5,a5,16
   1528c:	00160713          	addi	a4,a2,1
   15290:	0096a023          	sw	s1,0(a3)
   15294:	01c6a223          	sw	t3,4(a3)
   15298:	04f12423          	sw	a5,72(sp)
   1529c:	04e12223          	sw	a4,68(sp)
   152a0:	fcefdae3          	bge	t6,a4,15274 <_vfiprintf_r+0xa30>
   152a4:	18078c63          	beqz	a5,1543c <_vfiprintf_r+0xbf8>
   152a8:	00012503          	lw	a0,0(sp)
   152ac:	04010613          	addi	a2,sp,64
   152b0:	000b0593          	mv	a1,s6
   152b4:	c54ff0ef          	jal	14708 <__sprint_r.part.0>
   152b8:	fa051a63          	bnez	a0,14a6c <_vfiprintf_r+0x228>
   152bc:	04412603          	lw	a2,68(sp)
   152c0:	01000e13          	li	t3,16
   152c4:	ff040413          	addi	s0,s0,-16
   152c8:	04812783          	lw	a5,72(sp)
   152cc:	000a8693          	mv	a3,s5
   152d0:	00160593          	addi	a1,a2,1
   152d4:	00700f93          	li	t6,7
   152d8:	fa8e48e3          	blt	t3,s0,15288 <_vfiprintf_r+0xa44>
   152dc:	00040f13          	mv	t5,s0
   152e0:	00048e93          	mv	t4,s1
   152e4:	00068413          	mv	s0,a3
   152e8:	00812883          	lw	a7,8(sp)
   152ec:	01012683          	lw	a3,16(sp)
   152f0:	02012803          	lw	a6,32(sp)
   152f4:	00c12483          	lw	s1,12(sp)
   152f8:	01e787b3          	add	a5,a5,t5
   152fc:	01d42023          	sw	t4,0(s0)
   15300:	01e42223          	sw	t5,4(s0)
   15304:	04f12423          	sw	a5,72(sp)
   15308:	04b12223          	sw	a1,68(sp)
   1530c:	00700713          	li	a4,7
   15310:	54b75463          	bge	a4,a1,15858 <_vfiprintf_r+0x1014>
   15314:	7c078263          	beqz	a5,15ad8 <_vfiprintf_r+0x1294>
   15318:	00012503          	lw	a0,0(sp)
   1531c:	04010613          	addi	a2,sp,64
   15320:	000b0593          	mv	a1,s6
   15324:	01012823          	sw	a6,16(sp)
   15328:	00d12623          	sw	a3,12(sp)
   1532c:	01112423          	sw	a7,8(sp)
   15330:	bd8ff0ef          	jal	14708 <__sprint_r.part.0>
   15334:	f2051c63          	bnez	a0,14a6c <_vfiprintf_r+0x228>
   15338:	00812883          	lw	a7,8(sp)
   1533c:	04412603          	lw	a2,68(sp)
   15340:	04812783          	lw	a5,72(sp)
   15344:	411484b3          	sub	s1,s1,a7
   15348:	01012803          	lw	a6,16(sp)
   1534c:	00c12683          	lw	a3,12(sp)
   15350:	000a8413          	mv	s0,s5
   15354:	00160e13          	addi	t3,a2,1
   15358:	88905ae3          	blez	s1,14bec <_vfiprintf_r+0x3a8>
   1535c:	0000ce97          	auipc	t4,0xc
   15360:	d20e8e93          	addi	t4,t4,-736 # 2107c <zeroes.0>
   15364:	0a9a5063          	bge	s4,s1,15404 <_vfiprintf_r+0xbc0>
   15368:	00d12623          	sw	a3,12(sp)
   1536c:	00700f13          	li	t5,7
   15370:	00040693          	mv	a3,s0
   15374:	01112423          	sw	a7,8(sp)
   15378:	00048413          	mv	s0,s1
   1537c:	01012823          	sw	a6,16(sp)
   15380:	000c8493          	mv	s1,s9
   15384:	0180006f          	j	1539c <_vfiprintf_r+0xb58>
   15388:	00260e13          	addi	t3,a2,2
   1538c:	00868693          	addi	a3,a3,8
   15390:	00070613          	mv	a2,a4
   15394:	ff040413          	addi	s0,s0,-16
   15398:	048a5a63          	bge	s4,s0,153ec <_vfiprintf_r+0xba8>
   1539c:	01078793          	addi	a5,a5,16
   153a0:	00160713          	addi	a4,a2,1
   153a4:	0196a023          	sw	s9,0(a3)
   153a8:	0146a223          	sw	s4,4(a3)
   153ac:	04f12423          	sw	a5,72(sp)
   153b0:	04e12223          	sw	a4,68(sp)
   153b4:	fcef5ae3          	bge	t5,a4,15388 <_vfiprintf_r+0xb44>
   153b8:	06078a63          	beqz	a5,1542c <_vfiprintf_r+0xbe8>
   153bc:	00012503          	lw	a0,0(sp)
   153c0:	04010613          	addi	a2,sp,64
   153c4:	000b0593          	mv	a1,s6
   153c8:	b40ff0ef          	jal	14708 <__sprint_r.part.0>
   153cc:	ea051063          	bnez	a0,14a6c <_vfiprintf_r+0x228>
   153d0:	04412603          	lw	a2,68(sp)
   153d4:	ff040413          	addi	s0,s0,-16
   153d8:	04812783          	lw	a5,72(sp)
   153dc:	000a8693          	mv	a3,s5
   153e0:	00160e13          	addi	t3,a2,1
   153e4:	00700f13          	li	t5,7
   153e8:	fa8a4ae3          	blt	s4,s0,1539c <_vfiprintf_r+0xb58>
   153ec:	00048e93          	mv	t4,s1
   153f0:	00812883          	lw	a7,8(sp)
   153f4:	00040493          	mv	s1,s0
   153f8:	01012803          	lw	a6,16(sp)
   153fc:	00068413          	mv	s0,a3
   15400:	00c12683          	lw	a3,12(sp)
   15404:	009787b3          	add	a5,a5,s1
   15408:	01d42023          	sw	t4,0(s0)
   1540c:	00942223          	sw	s1,4(s0)
   15410:	04f12423          	sw	a5,72(sp)
   15414:	05c12223          	sw	t3,68(sp)
   15418:	00700713          	li	a4,7
   1541c:	23c74063          	blt	a4,t3,1563c <_vfiprintf_r+0xdf8>
   15420:	00840413          	addi	s0,s0,8
   15424:	001e0e13          	addi	t3,t3,1
   15428:	fc4ff06f          	j	14bec <_vfiprintf_r+0x3a8>
   1542c:	00100e13          	li	t3,1
   15430:	00000613          	li	a2,0
   15434:	000a8693          	mv	a3,s5
   15438:	f5dff06f          	j	15394 <_vfiprintf_r+0xb50>
   1543c:	00100593          	li	a1,1
   15440:	00000613          	li	a2,0
   15444:	000a8693          	mv	a3,s5
   15448:	e39ff06f          	j	15280 <_vfiprintf_r+0xa3c>
   1544c:	04078a63          	beqz	a5,154a0 <_vfiprintf_r+0xc5c>
   15450:	00012503          	lw	a0,0(sp)
   15454:	04010613          	addi	a2,sp,64
   15458:	000b0593          	mv	a1,s6
   1545c:	03012223          	sw	a6,36(sp)
   15460:	02d12023          	sw	a3,32(sp)
   15464:	01112823          	sw	a7,16(sp)
   15468:	01f12623          	sw	t6,12(sp)
   1546c:	01e12423          	sw	t5,8(sp)
   15470:	a98ff0ef          	jal	14708 <__sprint_r.part.0>
   15474:	de051c63          	bnez	a0,14a6c <_vfiprintf_r+0x228>
   15478:	04412603          	lw	a2,68(sp)
   1547c:	04812783          	lw	a5,72(sp)
   15480:	02412803          	lw	a6,36(sp)
   15484:	02012683          	lw	a3,32(sp)
   15488:	01012883          	lw	a7,16(sp)
   1548c:	00c12f83          	lw	t6,12(sp)
   15490:	00812f03          	lw	t5,8(sp)
   15494:	000a8413          	mv	s0,s5
   15498:	00160e13          	addi	t3,a2,1
   1549c:	ed0ff06f          	j	14b6c <_vfiprintf_r+0x328>
   154a0:	3e0f0063          	beqz	t5,15880 <_vfiprintf_r+0x103c>
   154a4:	03c10793          	addi	a5,sp,60
   154a8:	04f12623          	sw	a5,76(sp)
   154ac:	00200793          	li	a5,2
   154b0:	04f12823          	sw	a5,80(sp)
   154b4:	00100e13          	li	t3,1
   154b8:	000a8413          	mv	s0,s5
   154bc:	000e0613          	mv	a2,t3
   154c0:	00840413          	addi	s0,s0,8
   154c4:	001e0e13          	addi	t3,t3,1
   154c8:	f14ff06f          	j	14bdc <_vfiprintf_r+0x398>
   154cc:	00200713          	li	a4,2
   154d0:	00058d93          	mv	s11,a1
   154d4:	c40708e3          	beqz	a4,15124 <_vfiprintf_r+0x8e0>
   154d8:	00412583          	lw	a1,4(sp)
   154dc:	0f010913          	addi	s2,sp,240
   154e0:	00f7f713          	andi	a4,a5,15
   154e4:	00e58733          	add	a4,a1,a4
   154e8:	00074683          	lbu	a3,0(a4)
   154ec:	0047d793          	srli	a5,a5,0x4
   154f0:	01c61713          	slli	a4,a2,0x1c
   154f4:	00f767b3          	or	a5,a4,a5
   154f8:	00465613          	srli	a2,a2,0x4
   154fc:	fed90fa3          	sb	a3,-1(s2)
   15500:	00c7e733          	or	a4,a5,a2
   15504:	fff90913          	addi	s2,s2,-1
   15508:	fc071ce3          	bnez	a4,154e0 <_vfiprintf_r+0xc9c>
   1550c:	0f010793          	addi	a5,sp,240
   15510:	412788b3          	sub	a7,a5,s2
   15514:	00048693          	mv	a3,s1
   15518:	9914d8e3          	bge	s1,a7,14ea8 <_vfiprintf_r+0x664>
   1551c:	00088693          	mv	a3,a7
   15520:	989ff06f          	j	14ea8 <_vfiprintf_r+0x664>
   15524:	01000513          	li	a0,16
   15528:	62e55463          	bge	a0,a4,15b50 <_vfiprintf_r+0x130c>
   1552c:	0000ce17          	auipc	t3,0xc
   15530:	b60e0e13          	addi	t3,t3,-1184 # 2108c <blanks.1>
   15534:	02912023          	sw	s1,32(sp)
   15538:	02d12223          	sw	a3,36(sp)
   1553c:	01000e93          	li	t4,16
   15540:	00040693          	mv	a3,s0
   15544:	00700293          	li	t0,7
   15548:	01e12423          	sw	t5,8(sp)
   1554c:	01f12623          	sw	t6,12(sp)
   15550:	01112823          	sw	a7,16(sp)
   15554:	00070413          	mv	s0,a4
   15558:	03012423          	sw	a6,40(sp)
   1555c:	000e0493          	mv	s1,t3
   15560:	01c0006f          	j	1557c <_vfiprintf_r+0xd38>
   15564:	00260513          	addi	a0,a2,2
   15568:	00868693          	addi	a3,a3,8
   1556c:	00058613          	mv	a2,a1
   15570:	ff040413          	addi	s0,s0,-16
   15574:	048edc63          	bge	t4,s0,155cc <_vfiprintf_r+0xd88>
   15578:	00160593          	addi	a1,a2,1
   1557c:	01078793          	addi	a5,a5,16
   15580:	0096a023          	sw	s1,0(a3)
   15584:	01d6a223          	sw	t4,4(a3)
   15588:	04f12423          	sw	a5,72(sp)
   1558c:	04b12223          	sw	a1,68(sp)
   15590:	fcb2dae3          	bge	t0,a1,15564 <_vfiprintf_r+0xd20>
   15594:	08078463          	beqz	a5,1561c <_vfiprintf_r+0xdd8>
   15598:	00012503          	lw	a0,0(sp)
   1559c:	04010613          	addi	a2,sp,64
   155a0:	000b0593          	mv	a1,s6
   155a4:	964ff0ef          	jal	14708 <__sprint_r.part.0>
   155a8:	cc051263          	bnez	a0,14a6c <_vfiprintf_r+0x228>
   155ac:	04412603          	lw	a2,68(sp)
   155b0:	01000e93          	li	t4,16
   155b4:	ff040413          	addi	s0,s0,-16
   155b8:	04812783          	lw	a5,72(sp)
   155bc:	000a8693          	mv	a3,s5
   155c0:	00160513          	addi	a0,a2,1
   155c4:	00700293          	li	t0,7
   155c8:	fa8ec8e3          	blt	t4,s0,15578 <_vfiprintf_r+0xd34>
   155cc:	00040713          	mv	a4,s0
   155d0:	00048e13          	mv	t3,s1
   155d4:	00068413          	mv	s0,a3
   155d8:	00812f03          	lw	t5,8(sp)
   155dc:	00c12f83          	lw	t6,12(sp)
   155e0:	01012883          	lw	a7,16(sp)
   155e4:	02412683          	lw	a3,36(sp)
   155e8:	02812803          	lw	a6,40(sp)
   155ec:	02012483          	lw	s1,32(sp)
   155f0:	00e787b3          	add	a5,a5,a4
   155f4:	00e42223          	sw	a4,4(s0)
   155f8:	01c42023          	sw	t3,0(s0)
   155fc:	04f12423          	sw	a5,72(sp)
   15600:	04a12223          	sw	a0,68(sp)
   15604:	00700713          	li	a4,7
   15608:	0ea74a63          	blt	a4,a0,156fc <_vfiprintf_r+0xeb8>
   1560c:	00840413          	addi	s0,s0,8
   15610:	00150e13          	addi	t3,a0,1
   15614:	00050613          	mv	a2,a0
   15618:	d1cff06f          	j	14b34 <_vfiprintf_r+0x2f0>
   1561c:	00000613          	li	a2,0
   15620:	00100513          	li	a0,1
   15624:	000a8693          	mv	a3,s5
   15628:	f49ff06f          	j	15570 <_vfiprintf_r+0xd2c>
   1562c:	00dbd463          	bge	s7,a3,15634 <_vfiprintf_r+0xdf0>
   15630:	00068b93          	mv	s7,a3
   15634:	017c0c33          	add	s8,s8,s7
   15638:	e20ff06f          	j	14c58 <_vfiprintf_r+0x414>
   1563c:	bc0782e3          	beqz	a5,15200 <_vfiprintf_r+0x9bc>
   15640:	00012503          	lw	a0,0(sp)
   15644:	04010613          	addi	a2,sp,64
   15648:	000b0593          	mv	a1,s6
   1564c:	01012823          	sw	a6,16(sp)
   15650:	00d12623          	sw	a3,12(sp)
   15654:	01112423          	sw	a7,8(sp)
   15658:	8b0ff0ef          	jal	14708 <__sprint_r.part.0>
   1565c:	c0051863          	bnez	a0,14a6c <_vfiprintf_r+0x228>
   15660:	04412e03          	lw	t3,68(sp)
   15664:	04812783          	lw	a5,72(sp)
   15668:	01012803          	lw	a6,16(sp)
   1566c:	00c12683          	lw	a3,12(sp)
   15670:	00812883          	lw	a7,8(sp)
   15674:	000a8413          	mv	s0,s5
   15678:	001e0e13          	addi	t3,t3,1
   1567c:	d70ff06f          	j	14bec <_vfiprintf_r+0x3a8>
   15680:	04012223          	sw	zero,68(sp)
   15684:	000a8413          	mv	s0,s5
   15688:	b24ff06f          	j	149ac <_vfiprintf_r+0x168>
   1568c:	0106f713          	andi	a4,a3,16
   15690:	00082783          	lw	a5,0(a6)
   15694:	00480813          	addi	a6,a6,4
   15698:	00071a63          	bnez	a4,156ac <_vfiprintf_r+0xe68>
   1569c:	0406f713          	andi	a4,a3,64
   156a0:	40070c63          	beqz	a4,15ab8 <_vfiprintf_r+0x1274>
   156a4:	01079793          	slli	a5,a5,0x10
   156a8:	0107d793          	srli	a5,a5,0x10
   156ac:	00000893          	li	a7,0
   156b0:	a3dff06f          	j	150ec <_vfiprintf_r+0x8a8>
   156b4:	01077693          	andi	a3,a4,16
   156b8:	00082783          	lw	a5,0(a6)
   156bc:	00480813          	addi	a6,a6,4
   156c0:	02069263          	bnez	a3,156e4 <_vfiprintf_r+0xea0>
   156c4:	04077693          	andi	a3,a4,64
   156c8:	00068a63          	beqz	a3,156dc <_vfiprintf_r+0xe98>
   156cc:	01079793          	slli	a5,a5,0x10
   156d0:	0107d793          	srli	a5,a5,0x10
   156d4:	00000613          	li	a2,0
   156d8:	985ff06f          	j	1505c <_vfiprintf_r+0x818>
   156dc:	20077693          	andi	a3,a4,512
   156e0:	44069863          	bnez	a3,15b30 <_vfiprintf_r+0x12ec>
   156e4:	00000613          	li	a2,0
   156e8:	975ff06f          	j	1505c <_vfiprintf_r+0x818>
   156ec:	00100e13          	li	t3,1
   156f0:	00000613          	li	a2,0
   156f4:	000a8413          	mv	s0,s5
   156f8:	ce4ff06f          	j	14bdc <_vfiprintf_r+0x398>
   156fc:	24078e63          	beqz	a5,15958 <_vfiprintf_r+0x1114>
   15700:	00012503          	lw	a0,0(sp)
   15704:	04010613          	addi	a2,sp,64
   15708:	000b0593          	mv	a1,s6
   1570c:	03012223          	sw	a6,36(sp)
   15710:	02d12023          	sw	a3,32(sp)
   15714:	01112823          	sw	a7,16(sp)
   15718:	01f12623          	sw	t6,12(sp)
   1571c:	01e12423          	sw	t5,8(sp)
   15720:	fe9fe0ef          	jal	14708 <__sprint_r.part.0>
   15724:	b4051463          	bnez	a0,14a6c <_vfiprintf_r+0x228>
   15728:	04412603          	lw	a2,68(sp)
   1572c:	04812783          	lw	a5,72(sp)
   15730:	02412803          	lw	a6,36(sp)
   15734:	02012683          	lw	a3,32(sp)
   15738:	01012883          	lw	a7,16(sp)
   1573c:	00c12f83          	lw	t6,12(sp)
   15740:	00812f03          	lw	t5,8(sp)
   15744:	000a8413          	mv	s0,s5
   15748:	00160e13          	addi	t3,a2,1
   1574c:	be8ff06f          	j	14b34 <_vfiprintf_r+0x2f0>
   15750:	ccccde37          	lui	t3,0xccccd
   15754:	ccccd3b7          	lui	t2,0xccccd
   15758:	01412303          	lw	t1,20(sp)
   1575c:	400dff13          	andi	t5,s11,1024
   15760:	00000513          	li	a0,0
   15764:	0f010593          	addi	a1,sp,240
   15768:	00500e93          	li	t4,5
   1576c:	ccde0e13          	addi	t3,t3,-819 # cccccccd <__BSS_END__+0xccca9f9d>
   15770:	ccc38393          	addi	t2,t2,-820 # cccccccc <__BSS_END__+0xccca9f9c>
   15774:	0ff00f93          	li	t6,255
   15778:	0540006f          	j	157cc <_vfiprintf_r+0xf88>
   1577c:	00f93733          	sltu	a4,s2,a5
   15780:	00e90733          	add	a4,s2,a4
   15784:	03d77733          	remu	a4,a4,t4
   15788:	40e78733          	sub	a4,a5,a4
   1578c:	00e7b633          	sltu	a2,a5,a4
   15790:	40c88633          	sub	a2,a7,a2
   15794:	027702b3          	mul	t0,a4,t2
   15798:	03c60633          	mul	a2,a2,t3
   1579c:	03c735b3          	mulhu	a1,a4,t3
   157a0:	00560633          	add	a2,a2,t0
   157a4:	03c70733          	mul	a4,a4,t3
   157a8:	00b60633          	add	a2,a2,a1
   157ac:	01f61593          	slli	a1,a2,0x1f
   157b0:	00165613          	srli	a2,a2,0x1
   157b4:	00175713          	srli	a4,a4,0x1
   157b8:	00e5e733          	or	a4,a1,a4
   157bc:	32088663          	beqz	a7,15ae8 <_vfiprintf_r+0x12a4>
   157c0:	00070793          	mv	a5,a4
   157c4:	00060893          	mv	a7,a2
   157c8:	00068593          	mv	a1,a3
   157cc:	01178933          	add	s2,a5,a7
   157d0:	00f93733          	sltu	a4,s2,a5
   157d4:	00e90733          	add	a4,s2,a4
   157d8:	03d77733          	remu	a4,a4,t4
   157dc:	fff58693          	addi	a3,a1,-1
   157e0:	00150513          	addi	a0,a0,1
   157e4:	40e78733          	sub	a4,a5,a4
   157e8:	00e7b2b3          	sltu	t0,a5,a4
   157ec:	405882b3          	sub	t0,a7,t0
   157f0:	03c73633          	mulhu	a2,a4,t3
   157f4:	03c282b3          	mul	t0,t0,t3
   157f8:	03c70733          	mul	a4,a4,t3
   157fc:	00c282b3          	add	t0,t0,a2
   15800:	01f29293          	slli	t0,t0,0x1f
   15804:	00175613          	srli	a2,a4,0x1
   15808:	00c2e633          	or	a2,t0,a2
   1580c:	00261713          	slli	a4,a2,0x2
   15810:	00c70733          	add	a4,a4,a2
   15814:	00171713          	slli	a4,a4,0x1
   15818:	40e78733          	sub	a4,a5,a4
   1581c:	03070713          	addi	a4,a4,48
   15820:	fee58fa3          	sb	a4,-1(a1)
   15824:	f40f0ce3          	beqz	t5,1577c <_vfiprintf_r+0xf38>
   15828:	00034703          	lbu	a4,0(t1)
   1582c:	f4a718e3          	bne	a4,a0,1577c <_vfiprintf_r+0xf38>
   15830:	f5f506e3          	beq	a0,t6,1577c <_vfiprintf_r+0xf38>
   15834:	14089463          	bnez	a7,1597c <_vfiprintf_r+0x1138>
   15838:	00900713          	li	a4,9
   1583c:	14f76063          	bltu	a4,a5,1597c <_vfiprintf_r+0x1138>
   15840:	00612a23          	sw	t1,20(sp)
   15844:	00068913          	mv	s2,a3
   15848:	cc5ff06f          	j	1550c <_vfiprintf_r+0xcc8>
   1584c:	00000693          	li	a3,0
   15850:	0f010913          	addi	s2,sp,240
   15854:	e54ff06f          	j	14ea8 <_vfiprintf_r+0x664>
   15858:	00840413          	addi	s0,s0,8
   1585c:	00158e13          	addi	t3,a1,1
   15860:	00058613          	mv	a2,a1
   15864:	b80ff06f          	j	14be4 <_vfiprintf_r+0x3a0>
   15868:	00082783          	lw	a5,0(a6)
   1586c:	41fc5713          	srai	a4,s8,0x1f
   15870:	00480813          	addi	a6,a6,4
   15874:	0187a023          	sw	s8,0(a5)
   15878:	00e7a223          	sw	a4,4(a5)
   1587c:	8b8ff06f          	j	14934 <_vfiprintf_r+0xf0>
   15880:	00000613          	li	a2,0
   15884:	00100e13          	li	t3,1
   15888:	000a8413          	mv	s0,s5
   1588c:	b50ff06f          	j	14bdc <_vfiprintf_r+0x398>
   15890:	00088693          	mv	a3,a7
   15894:	00058913          	mv	s2,a1
   15898:	e10ff06f          	j	14ea8 <_vfiprintf_r+0x664>
   1589c:	fff00493          	li	s1,-1
   158a0:	00060993          	mv	s3,a2
   158a4:	934ff06f          	j	149d8 <_vfiprintf_r+0x194>
   158a8:	000d8693          	mv	a3,s11
   158ac:	825ff06f          	j	150d0 <_vfiprintf_r+0x88c>
   158b0:	000d8713          	mv	a4,s11
   158b4:	f8cff06f          	j	15040 <_vfiprintf_r+0x7fc>
   158b8:	0000b797          	auipc	a5,0xb
   158bc:	2a478793          	addi	a5,a5,676 # 20b5c <_exit+0x1ac>
   158c0:	00f12223          	sw	a5,4(sp)
   158c4:	020df793          	andi	a5,s11,32
   158c8:	04078a63          	beqz	a5,1591c <_vfiprintf_r+0x10d8>
   158cc:	00780813          	addi	a6,a6,7
   158d0:	ff887813          	andi	a6,a6,-8
   158d4:	00082783          	lw	a5,0(a6)
   158d8:	00482603          	lw	a2,4(a6)
   158dc:	00880813          	addi	a6,a6,8
   158e0:	001df693          	andi	a3,s11,1
   158e4:	00068e63          	beqz	a3,15900 <_vfiprintf_r+0x10bc>
   158e8:	00c7e6b3          	or	a3,a5,a2
   158ec:	00068a63          	beqz	a3,15900 <_vfiprintf_r+0x10bc>
   158f0:	03000693          	li	a3,48
   158f4:	02d10e23          	sb	a3,60(sp)
   158f8:	02e10ea3          	sb	a4,61(sp)
   158fc:	002ded93          	ori	s11,s11,2
   15900:	bffdfd93          	andi	s11,s11,-1025
   15904:	8a9ff06f          	j	151ac <_vfiprintf_r+0x968>
   15908:	0000b797          	auipc	a5,0xb
   1590c:	24078793          	addi	a5,a5,576 # 20b48 <_exit+0x198>
   15910:	00f12223          	sw	a5,4(sp)
   15914:	020df793          	andi	a5,s11,32
   15918:	fa079ae3          	bnez	a5,158cc <_vfiprintf_r+0x1088>
   1591c:	010df693          	andi	a3,s11,16
   15920:	00082783          	lw	a5,0(a6)
   15924:	00480813          	addi	a6,a6,4
   15928:	12069263          	bnez	a3,15a4c <_vfiprintf_r+0x1208>
   1592c:	040df693          	andi	a3,s11,64
   15930:	10068a63          	beqz	a3,15a44 <_vfiprintf_r+0x1200>
   15934:	01079793          	slli	a5,a5,0x10
   15938:	0107d793          	srli	a5,a5,0x10
   1593c:	00000613          	li	a2,0
   15940:	fa1ff06f          	j	158e0 <_vfiprintf_r+0x109c>
   15944:	00090513          	mv	a0,s2
   15948:	e14fb0ef          	jal	10f5c <strlen>
   1594c:	00812803          	lw	a6,8(sp)
   15950:	00050893          	mv	a7,a0
   15954:	d98ff06f          	j	14eec <_vfiprintf_r+0x6a8>
   15958:	03b14703          	lbu	a4,59(sp)
   1595c:	1a070063          	beqz	a4,15afc <_vfiprintf_r+0x12b8>
   15960:	03b10793          	addi	a5,sp,59
   15964:	04f12623          	sw	a5,76(sp)
   15968:	00100793          	li	a5,1
   1596c:	04f12823          	sw	a5,80(sp)
   15970:	00100e13          	li	t3,1
   15974:	000a8413          	mv	s0,s5
   15978:	9e8ff06f          	j	14b60 <_vfiprintf_r+0x31c>
   1597c:	02f12423          	sw	a5,40(sp)
   15980:	01812783          	lw	a5,24(sp)
   15984:	01c12583          	lw	a1,28(sp)
   15988:	03112623          	sw	a7,44(sp)
   1598c:	40f686b3          	sub	a3,a3,a5
   15990:	00078613          	mv	a2,a5
   15994:	00068513          	mv	a0,a3
   15998:	02712223          	sw	t2,36(sp)
   1599c:	03c12023          	sw	t3,32(sp)
   159a0:	01012a23          	sw	a6,20(sp)
   159a4:	01e12823          	sw	t5,16(sp)
   159a8:	00612623          	sw	t1,12(sp)
   159ac:	00d12423          	sw	a3,8(sp)
   159b0:	7c5000ef          	jal	16974 <strncpy>
   159b4:	02812783          	lw	a5,40(sp)
   159b8:	00500613          	li	a2,5
   159bc:	00c12303          	lw	t1,12(sp)
   159c0:	00f93733          	sltu	a4,s2,a5
   159c4:	00e90733          	add	a4,s2,a4
   159c8:	02c77733          	remu	a4,a4,a2
   159cc:	00134603          	lbu	a2,1(t1)
   159d0:	02c12883          	lw	a7,44(sp)
   159d4:	ccccd5b7          	lui	a1,0xccccd
   159d8:	00c03633          	snez	a2,a2
   159dc:	ccccd2b7          	lui	t0,0xccccd
   159e0:	00c30333          	add	t1,t1,a2
   159e4:	ccd58593          	addi	a1,a1,-819 # cccccccd <__BSS_END__+0xccca9f9d>
   159e8:	ccc28293          	addi	t0,t0,-820 # cccccccc <__BSS_END__+0xccca9f9c>
   159ec:	00812683          	lw	a3,8(sp)
   159f0:	01012f03          	lw	t5,16(sp)
   159f4:	01412803          	lw	a6,20(sp)
   159f8:	02012e03          	lw	t3,32(sp)
   159fc:	02412383          	lw	t2,36(sp)
   15a00:	00000513          	li	a0,0
   15a04:	00500e93          	li	t4,5
   15a08:	0ff00f93          	li	t6,255
   15a0c:	40e78733          	sub	a4,a5,a4
   15a10:	00e7b633          	sltu	a2,a5,a4
   15a14:	40c88633          	sub	a2,a7,a2
   15a18:	025702b3          	mul	t0,a4,t0
   15a1c:	02b60633          	mul	a2,a2,a1
   15a20:	02b738b3          	mulhu	a7,a4,a1
   15a24:	00560633          	add	a2,a2,t0
   15a28:	02b707b3          	mul	a5,a4,a1
   15a2c:	01160633          	add	a2,a2,a7
   15a30:	01f61713          	slli	a4,a2,0x1f
   15a34:	00165613          	srli	a2,a2,0x1
   15a38:	0017d793          	srli	a5,a5,0x1
   15a3c:	00f76733          	or	a4,a4,a5
   15a40:	d81ff06f          	j	157c0 <_vfiprintf_r+0xf7c>
   15a44:	200df693          	andi	a3,s11,512
   15a48:	0e069a63          	bnez	a3,15b3c <_vfiprintf_r+0x12f8>
   15a4c:	00000613          	li	a2,0
   15a50:	e91ff06f          	j	158e0 <_vfiprintf_r+0x109c>
   15a54:	00070d93          	mv	s11,a4
   15a58:	00000713          	li	a4,0
   15a5c:	ec070463          	beqz	a4,15124 <_vfiprintf_r+0x8e0>
   15a60:	a79ff06f          	j	154d8 <_vfiprintf_r+0xc94>
   15a64:	d85fa0ef          	jal	107e8 <__sinit>
   15a68:	e19fe06f          	j	14880 <_vfiprintf_r+0x3c>
   15a6c:	0019c703          	lbu	a4,1(s3)
   15a70:	200ded93          	ori	s11,s11,512
   15a74:	00198993          	addi	s3,s3,1
   15a78:	f5dfe06f          	j	149d4 <_vfiprintf_r+0x190>
   15a7c:	0019c703          	lbu	a4,1(s3)
   15a80:	020ded93          	ori	s11,s11,32
   15a84:	00198993          	addi	s3,s3,1
   15a88:	f4dfe06f          	j	149d4 <_vfiprintf_r+0x190>
   15a8c:	00082783          	lw	a5,0(a6)
   15a90:	00480813          	addi	a6,a6,4
   15a94:	0187a023          	sw	s8,0(a5)
   15a98:	e9dfe06f          	j	14934 <_vfiprintf_r+0xf0>
   15a9c:	00600793          	li	a5,6
   15aa0:	00048893          	mv	a7,s1
   15aa4:	0497e863          	bltu	a5,s1,15af4 <_vfiprintf_r+0x12b0>
   15aa8:	00088693          	mv	a3,a7
   15aac:	0000b917          	auipc	s2,0xb
   15ab0:	0c490913          	addi	s2,s2,196 # 20b70 <_exit+0x1c0>
   15ab4:	858ff06f          	j	14b0c <_vfiprintf_r+0x2c8>
   15ab8:	2006f713          	andi	a4,a3,512
   15abc:	be0708e3          	beqz	a4,156ac <_vfiprintf_r+0xe68>
   15ac0:	0ff7f793          	zext.b	a5,a5
   15ac4:	00000893          	li	a7,0
   15ac8:	e24ff06f          	j	150ec <_vfiprintf_r+0x8a8>
   15acc:	00000793          	li	a5,0
   15ad0:	00000613          	li	a2,0
   15ad4:	a05ff06f          	j	154d8 <_vfiprintf_r+0xc94>
   15ad8:	00100e13          	li	t3,1
   15adc:	00000613          	li	a2,0
   15ae0:	000a8413          	mv	s0,s5
   15ae4:	900ff06f          	j	14be4 <_vfiprintf_r+0x3a0>
   15ae8:	00900593          	li	a1,9
   15aec:	ccf5eae3          	bltu	a1,a5,157c0 <_vfiprintf_r+0xf7c>
   15af0:	d51ff06f          	j	15840 <_vfiprintf_r+0xffc>
   15af4:	00600893          	li	a7,6
   15af8:	fb1ff06f          	j	15aa8 <_vfiprintf_r+0x1264>
   15afc:	00000613          	li	a2,0
   15b00:	00100e13          	li	t3,1
   15b04:	000a8413          	mv	s0,s5
   15b08:	864ff06f          	j	14b6c <_vfiprintf_r+0x328>
   15b0c:	00082783          	lw	a5,0(a6)
   15b10:	00480813          	addi	a6,a6,4
   15b14:	01879023          	sh	s8,0(a5)
   15b18:	e1dfe06f          	j	14934 <_vfiprintf_r+0xf0>
   15b1c:	01879793          	slli	a5,a5,0x18
   15b20:	4187d793          	srai	a5,a5,0x18
   15b24:	41f7d893          	srai	a7,a5,0x1f
   15b28:	00088713          	mv	a4,a7
   15b2c:	9a4ff06f          	j	14cd0 <_vfiprintf_r+0x48c>
   15b30:	0ff7f793          	zext.b	a5,a5
   15b34:	00000613          	li	a2,0
   15b38:	d24ff06f          	j	1505c <_vfiprintf_r+0x818>
   15b3c:	0ff7f793          	zext.b	a5,a5
   15b40:	00000613          	li	a2,0
   15b44:	d9dff06f          	j	158e0 <_vfiprintf_r+0x109c>
   15b48:	03000793          	li	a5,48
   15b4c:	b44ff06f          	j	14e90 <_vfiprintf_r+0x64c>
   15b50:	00058513          	mv	a0,a1
   15b54:	0000be17          	auipc	t3,0xb
   15b58:	538e0e13          	addi	t3,t3,1336 # 2108c <blanks.1>
   15b5c:	a95ff06f          	j	155f0 <_vfiprintf_r+0xdac>
   15b60:	00160593          	addi	a1,a2,1
   15b64:	0000be17          	auipc	t3,0xb
   15b68:	528e0e13          	addi	t3,t3,1320 # 2108c <blanks.1>
   15b6c:	a78ff06f          	j	14de4 <_vfiprintf_r+0x5a0>
   15b70:	fff00d93          	li	s11,-1
   15b74:	f29fe06f          	j	14a9c <_vfiprintf_r+0x258>
   15b78:	000e0593          	mv	a1,t3
   15b7c:	0000be97          	auipc	t4,0xb
   15b80:	500e8e93          	addi	t4,t4,1280 # 2107c <zeroes.0>
   15b84:	f74ff06f          	j	152f8 <_vfiprintf_r+0xab4>
   15b88:	00082483          	lw	s1,0(a6)
   15b8c:	00480813          	addi	a6,a6,4
   15b90:	0004d463          	bgez	s1,15b98 <_vfiprintf_r+0x1354>
   15b94:	fff00493          	li	s1,-1
   15b98:	0019c703          	lbu	a4,1(s3)
   15b9c:	00060993          	mv	s3,a2
   15ba0:	e35fe06f          	j	149d4 <_vfiprintf_r+0x190>

00015ba4 <vfiprintf>:
   15ba4:	00060693          	mv	a3,a2
   15ba8:	00058613          	mv	a2,a1
   15bac:	00050593          	mv	a1,a0
   15bb0:	08c1a503          	lw	a0,140(gp) # 2290c <_impure_ptr>
   15bb4:	c91fe06f          	j	14844 <_vfiprintf_r>

00015bb8 <__sbprintf>:
   15bb8:	b8010113          	addi	sp,sp,-1152
   15bbc:	00c59783          	lh	a5,12(a1)
   15bc0:	00e5d703          	lhu	a4,14(a1)
   15bc4:	46812c23          	sw	s0,1144(sp)
   15bc8:	00058413          	mv	s0,a1
   15bcc:	000105b7          	lui	a1,0x10
   15bd0:	ffd58593          	addi	a1,a1,-3 # fffd <exit-0xb7>
   15bd4:	06442e03          	lw	t3,100(s0)
   15bd8:	01c42303          	lw	t1,28(s0)
   15bdc:	02442883          	lw	a7,36(s0)
   15be0:	01071713          	slli	a4,a4,0x10
   15be4:	00b7f7b3          	and	a5,a5,a1
   15be8:	00e7e7b3          	or	a5,a5,a4
   15bec:	40000813          	li	a6,1024
   15bf0:	00f12a23          	sw	a5,20(sp)
   15bf4:	00810593          	addi	a1,sp,8
   15bf8:	07010793          	addi	a5,sp,112
   15bfc:	46912a23          	sw	s1,1140(sp)
   15c00:	47212823          	sw	s2,1136(sp)
   15c04:	46112e23          	sw	ra,1148(sp)
   15c08:	00050913          	mv	s2,a0
   15c0c:	07c12623          	sw	t3,108(sp)
   15c10:	02612223          	sw	t1,36(sp)
   15c14:	03112623          	sw	a7,44(sp)
   15c18:	00f12423          	sw	a5,8(sp)
   15c1c:	00f12c23          	sw	a5,24(sp)
   15c20:	01012823          	sw	a6,16(sp)
   15c24:	01012e23          	sw	a6,28(sp)
   15c28:	02012023          	sw	zero,32(sp)
   15c2c:	c19fe0ef          	jal	14844 <_vfiprintf_r>
   15c30:	00050493          	mv	s1,a0
   15c34:	02055c63          	bgez	a0,15c6c <__sbprintf+0xb4>
   15c38:	01415783          	lhu	a5,20(sp)
   15c3c:	0407f793          	andi	a5,a5,64
   15c40:	00078863          	beqz	a5,15c50 <__sbprintf+0x98>
   15c44:	00c45783          	lhu	a5,12(s0)
   15c48:	0407e793          	ori	a5,a5,64
   15c4c:	00f41623          	sh	a5,12(s0)
   15c50:	47c12083          	lw	ra,1148(sp)
   15c54:	47812403          	lw	s0,1144(sp)
   15c58:	47012903          	lw	s2,1136(sp)
   15c5c:	00048513          	mv	a0,s1
   15c60:	47412483          	lw	s1,1140(sp)
   15c64:	48010113          	addi	sp,sp,1152
   15c68:	00008067          	ret
   15c6c:	00810593          	addi	a1,sp,8
   15c70:	00090513          	mv	a0,s2
   15c74:	370000ef          	jal	15fe4 <_fflush_r>
   15c78:	fc0500e3          	beqz	a0,15c38 <__sbprintf+0x80>
   15c7c:	fff00493          	li	s1,-1
   15c80:	fb9ff06f          	j	15c38 <__sbprintf+0x80>

00015c84 <_fclose_r>:
   15c84:	ff010113          	addi	sp,sp,-16
   15c88:	00112623          	sw	ra,12(sp)
   15c8c:	01212023          	sw	s2,0(sp)
   15c90:	02058863          	beqz	a1,15cc0 <_fclose_r+0x3c>
   15c94:	00812423          	sw	s0,8(sp)
   15c98:	00912223          	sw	s1,4(sp)
   15c9c:	00058413          	mv	s0,a1
   15ca0:	00050493          	mv	s1,a0
   15ca4:	00050663          	beqz	a0,15cb0 <_fclose_r+0x2c>
   15ca8:	03452783          	lw	a5,52(a0)
   15cac:	0c078c63          	beqz	a5,15d84 <_fclose_r+0x100>
   15cb0:	00c41783          	lh	a5,12(s0)
   15cb4:	02079263          	bnez	a5,15cd8 <_fclose_r+0x54>
   15cb8:	00812403          	lw	s0,8(sp)
   15cbc:	00412483          	lw	s1,4(sp)
   15cc0:	00c12083          	lw	ra,12(sp)
   15cc4:	00000913          	li	s2,0
   15cc8:	00090513          	mv	a0,s2
   15ccc:	00012903          	lw	s2,0(sp)
   15cd0:	01010113          	addi	sp,sp,16
   15cd4:	00008067          	ret
   15cd8:	00040593          	mv	a1,s0
   15cdc:	00048513          	mv	a0,s1
   15ce0:	0b8000ef          	jal	15d98 <__sflush_r>
   15ce4:	02c42783          	lw	a5,44(s0)
   15ce8:	00050913          	mv	s2,a0
   15cec:	00078a63          	beqz	a5,15d00 <_fclose_r+0x7c>
   15cf0:	01c42583          	lw	a1,28(s0)
   15cf4:	00048513          	mv	a0,s1
   15cf8:	000780e7          	jalr	a5
   15cfc:	06054463          	bltz	a0,15d64 <_fclose_r+0xe0>
   15d00:	00c45783          	lhu	a5,12(s0)
   15d04:	0807f793          	andi	a5,a5,128
   15d08:	06079663          	bnez	a5,15d74 <_fclose_r+0xf0>
   15d0c:	03042583          	lw	a1,48(s0)
   15d10:	00058c63          	beqz	a1,15d28 <_fclose_r+0xa4>
   15d14:	04040793          	addi	a5,s0,64
   15d18:	00f58663          	beq	a1,a5,15d24 <_fclose_r+0xa0>
   15d1c:	00048513          	mv	a0,s1
   15d20:	d4cfb0ef          	jal	1126c <_free_r>
   15d24:	02042823          	sw	zero,48(s0)
   15d28:	04442583          	lw	a1,68(s0)
   15d2c:	00058863          	beqz	a1,15d3c <_fclose_r+0xb8>
   15d30:	00048513          	mv	a0,s1
   15d34:	d38fb0ef          	jal	1126c <_free_r>
   15d38:	04042223          	sw	zero,68(s0)
   15d3c:	ad1fa0ef          	jal	1080c <__sfp_lock_acquire>
   15d40:	00041623          	sh	zero,12(s0)
   15d44:	acdfa0ef          	jal	10810 <__sfp_lock_release>
   15d48:	00c12083          	lw	ra,12(sp)
   15d4c:	00812403          	lw	s0,8(sp)
   15d50:	00412483          	lw	s1,4(sp)
   15d54:	00090513          	mv	a0,s2
   15d58:	00012903          	lw	s2,0(sp)
   15d5c:	01010113          	addi	sp,sp,16
   15d60:	00008067          	ret
   15d64:	00c45783          	lhu	a5,12(s0)
   15d68:	fff00913          	li	s2,-1
   15d6c:	0807f793          	andi	a5,a5,128
   15d70:	f8078ee3          	beqz	a5,15d0c <_fclose_r+0x88>
   15d74:	01042583          	lw	a1,16(s0)
   15d78:	00048513          	mv	a0,s1
   15d7c:	cf0fb0ef          	jal	1126c <_free_r>
   15d80:	f8dff06f          	j	15d0c <_fclose_r+0x88>
   15d84:	a65fa0ef          	jal	107e8 <__sinit>
   15d88:	f29ff06f          	j	15cb0 <_fclose_r+0x2c>

00015d8c <fclose>:
   15d8c:	00050593          	mv	a1,a0
   15d90:	08c1a503          	lw	a0,140(gp) # 2290c <_impure_ptr>
   15d94:	ef1ff06f          	j	15c84 <_fclose_r>

00015d98 <__sflush_r>:
   15d98:	00c59703          	lh	a4,12(a1)
   15d9c:	fe010113          	addi	sp,sp,-32
   15da0:	00812c23          	sw	s0,24(sp)
   15da4:	01312623          	sw	s3,12(sp)
   15da8:	00112e23          	sw	ra,28(sp)
   15dac:	00877793          	andi	a5,a4,8
   15db0:	00058413          	mv	s0,a1
   15db4:	00050993          	mv	s3,a0
   15db8:	12079063          	bnez	a5,15ed8 <__sflush_r+0x140>
   15dbc:	000017b7          	lui	a5,0x1
   15dc0:	80078793          	addi	a5,a5,-2048 # 800 <exit-0xf8b4>
   15dc4:	0045a683          	lw	a3,4(a1)
   15dc8:	00f767b3          	or	a5,a4,a5
   15dcc:	00f59623          	sh	a5,12(a1)
   15dd0:	18d05263          	blez	a3,15f54 <__sflush_r+0x1bc>
   15dd4:	02842803          	lw	a6,40(s0)
   15dd8:	0e080463          	beqz	a6,15ec0 <__sflush_r+0x128>
   15ddc:	00912a23          	sw	s1,20(sp)
   15de0:	01371693          	slli	a3,a4,0x13
   15de4:	0009a483          	lw	s1,0(s3)
   15de8:	0009a023          	sw	zero,0(s3)
   15dec:	01c42583          	lw	a1,28(s0)
   15df0:	1606ce63          	bltz	a3,15f6c <__sflush_r+0x1d4>
   15df4:	00000613          	li	a2,0
   15df8:	00100693          	li	a3,1
   15dfc:	00098513          	mv	a0,s3
   15e00:	000800e7          	jalr	a6
   15e04:	fff00793          	li	a5,-1
   15e08:	00050613          	mv	a2,a0
   15e0c:	1af50463          	beq	a0,a5,15fb4 <__sflush_r+0x21c>
   15e10:	00c41783          	lh	a5,12(s0)
   15e14:	02842803          	lw	a6,40(s0)
   15e18:	01c42583          	lw	a1,28(s0)
   15e1c:	0047f793          	andi	a5,a5,4
   15e20:	00078e63          	beqz	a5,15e3c <__sflush_r+0xa4>
   15e24:	00442703          	lw	a4,4(s0)
   15e28:	03042783          	lw	a5,48(s0)
   15e2c:	40e60633          	sub	a2,a2,a4
   15e30:	00078663          	beqz	a5,15e3c <__sflush_r+0xa4>
   15e34:	03c42783          	lw	a5,60(s0)
   15e38:	40f60633          	sub	a2,a2,a5
   15e3c:	00000693          	li	a3,0
   15e40:	00098513          	mv	a0,s3
   15e44:	000800e7          	jalr	a6
   15e48:	fff00793          	li	a5,-1
   15e4c:	12f51463          	bne	a0,a5,15f74 <__sflush_r+0x1dc>
   15e50:	0009a683          	lw	a3,0(s3)
   15e54:	01d00793          	li	a5,29
   15e58:	00c41703          	lh	a4,12(s0)
   15e5c:	16d7ea63          	bltu	a5,a3,15fd0 <__sflush_r+0x238>
   15e60:	204007b7          	lui	a5,0x20400
   15e64:	00178793          	addi	a5,a5,1 # 20400001 <__BSS_END__+0x203dd2d1>
   15e68:	00d7d7b3          	srl	a5,a5,a3
   15e6c:	0017f793          	andi	a5,a5,1
   15e70:	16078063          	beqz	a5,15fd0 <__sflush_r+0x238>
   15e74:	01042603          	lw	a2,16(s0)
   15e78:	fffff7b7          	lui	a5,0xfffff
   15e7c:	7ff78793          	addi	a5,a5,2047 # fffff7ff <__BSS_END__+0xfffdcacf>
   15e80:	00f777b3          	and	a5,a4,a5
   15e84:	00f41623          	sh	a5,12(s0)
   15e88:	00042223          	sw	zero,4(s0)
   15e8c:	00c42023          	sw	a2,0(s0)
   15e90:	01371793          	slli	a5,a4,0x13
   15e94:	0007d463          	bgez	a5,15e9c <__sflush_r+0x104>
   15e98:	10068263          	beqz	a3,15f9c <__sflush_r+0x204>
   15e9c:	03042583          	lw	a1,48(s0)
   15ea0:	0099a023          	sw	s1,0(s3)
   15ea4:	10058463          	beqz	a1,15fac <__sflush_r+0x214>
   15ea8:	04040793          	addi	a5,s0,64
   15eac:	00f58663          	beq	a1,a5,15eb8 <__sflush_r+0x120>
   15eb0:	00098513          	mv	a0,s3
   15eb4:	bb8fb0ef          	jal	1126c <_free_r>
   15eb8:	01412483          	lw	s1,20(sp)
   15ebc:	02042823          	sw	zero,48(s0)
   15ec0:	00000513          	li	a0,0
   15ec4:	01c12083          	lw	ra,28(sp)
   15ec8:	01812403          	lw	s0,24(sp)
   15ecc:	00c12983          	lw	s3,12(sp)
   15ed0:	02010113          	addi	sp,sp,32
   15ed4:	00008067          	ret
   15ed8:	01212823          	sw	s2,16(sp)
   15edc:	0105a903          	lw	s2,16(a1)
   15ee0:	08090263          	beqz	s2,15f64 <__sflush_r+0x1cc>
   15ee4:	00912a23          	sw	s1,20(sp)
   15ee8:	0005a483          	lw	s1,0(a1)
   15eec:	00377713          	andi	a4,a4,3
   15ef0:	0125a023          	sw	s2,0(a1)
   15ef4:	412484b3          	sub	s1,s1,s2
   15ef8:	00000793          	li	a5,0
   15efc:	00071463          	bnez	a4,15f04 <__sflush_r+0x16c>
   15f00:	0145a783          	lw	a5,20(a1)
   15f04:	00f42423          	sw	a5,8(s0)
   15f08:	00904863          	bgtz	s1,15f18 <__sflush_r+0x180>
   15f0c:	0540006f          	j	15f60 <__sflush_r+0x1c8>
   15f10:	00a90933          	add	s2,s2,a0
   15f14:	04905663          	blez	s1,15f60 <__sflush_r+0x1c8>
   15f18:	02442783          	lw	a5,36(s0)
   15f1c:	01c42583          	lw	a1,28(s0)
   15f20:	00048693          	mv	a3,s1
   15f24:	00090613          	mv	a2,s2
   15f28:	00098513          	mv	a0,s3
   15f2c:	000780e7          	jalr	a5
   15f30:	40a484b3          	sub	s1,s1,a0
   15f34:	fca04ee3          	bgtz	a0,15f10 <__sflush_r+0x178>
   15f38:	00c41703          	lh	a4,12(s0)
   15f3c:	01012903          	lw	s2,16(sp)
   15f40:	04076713          	ori	a4,a4,64
   15f44:	01412483          	lw	s1,20(sp)
   15f48:	00e41623          	sh	a4,12(s0)
   15f4c:	fff00513          	li	a0,-1
   15f50:	f75ff06f          	j	15ec4 <__sflush_r+0x12c>
   15f54:	03c5a683          	lw	a3,60(a1)
   15f58:	e6d04ee3          	bgtz	a3,15dd4 <__sflush_r+0x3c>
   15f5c:	f65ff06f          	j	15ec0 <__sflush_r+0x128>
   15f60:	01412483          	lw	s1,20(sp)
   15f64:	01012903          	lw	s2,16(sp)
   15f68:	f59ff06f          	j	15ec0 <__sflush_r+0x128>
   15f6c:	05042603          	lw	a2,80(s0)
   15f70:	eadff06f          	j	15e1c <__sflush_r+0x84>
   15f74:	00c41703          	lh	a4,12(s0)
   15f78:	01042683          	lw	a3,16(s0)
   15f7c:	fffff7b7          	lui	a5,0xfffff
   15f80:	7ff78793          	addi	a5,a5,2047 # fffff7ff <__BSS_END__+0xfffdcacf>
   15f84:	00f777b3          	and	a5,a4,a5
   15f88:	00f41623          	sh	a5,12(s0)
   15f8c:	00042223          	sw	zero,4(s0)
   15f90:	00d42023          	sw	a3,0(s0)
   15f94:	01371793          	slli	a5,a4,0x13
   15f98:	f007d2e3          	bgez	a5,15e9c <__sflush_r+0x104>
   15f9c:	03042583          	lw	a1,48(s0)
   15fa0:	04a42823          	sw	a0,80(s0)
   15fa4:	0099a023          	sw	s1,0(s3)
   15fa8:	f00590e3          	bnez	a1,15ea8 <__sflush_r+0x110>
   15fac:	01412483          	lw	s1,20(sp)
   15fb0:	f11ff06f          	j	15ec0 <__sflush_r+0x128>
   15fb4:	0009a783          	lw	a5,0(s3)
   15fb8:	e4078ce3          	beqz	a5,15e10 <__sflush_r+0x78>
   15fbc:	01d00713          	li	a4,29
   15fc0:	00e78c63          	beq	a5,a4,15fd8 <__sflush_r+0x240>
   15fc4:	01600713          	li	a4,22
   15fc8:	00e78863          	beq	a5,a4,15fd8 <__sflush_r+0x240>
   15fcc:	00c41703          	lh	a4,12(s0)
   15fd0:	04076713          	ori	a4,a4,64
   15fd4:	f71ff06f          	j	15f44 <__sflush_r+0x1ac>
   15fd8:	0099a023          	sw	s1,0(s3)
   15fdc:	01412483          	lw	s1,20(sp)
   15fe0:	ee1ff06f          	j	15ec0 <__sflush_r+0x128>

00015fe4 <_fflush_r>:
   15fe4:	fe010113          	addi	sp,sp,-32
   15fe8:	00812c23          	sw	s0,24(sp)
   15fec:	00112e23          	sw	ra,28(sp)
   15ff0:	00050413          	mv	s0,a0
   15ff4:	00050663          	beqz	a0,16000 <_fflush_r+0x1c>
   15ff8:	03452783          	lw	a5,52(a0)
   15ffc:	02078a63          	beqz	a5,16030 <_fflush_r+0x4c>
   16000:	00c59783          	lh	a5,12(a1)
   16004:	00079c63          	bnez	a5,1601c <_fflush_r+0x38>
   16008:	01c12083          	lw	ra,28(sp)
   1600c:	01812403          	lw	s0,24(sp)
   16010:	00000513          	li	a0,0
   16014:	02010113          	addi	sp,sp,32
   16018:	00008067          	ret
   1601c:	00040513          	mv	a0,s0
   16020:	01812403          	lw	s0,24(sp)
   16024:	01c12083          	lw	ra,28(sp)
   16028:	02010113          	addi	sp,sp,32
   1602c:	d6dff06f          	j	15d98 <__sflush_r>
   16030:	00b12623          	sw	a1,12(sp)
   16034:	fb4fa0ef          	jal	107e8 <__sinit>
   16038:	00c12583          	lw	a1,12(sp)
   1603c:	fc5ff06f          	j	16000 <_fflush_r+0x1c>

00016040 <fflush>:
   16040:	06050063          	beqz	a0,160a0 <fflush+0x60>
   16044:	00050593          	mv	a1,a0
   16048:	08c1a503          	lw	a0,140(gp) # 2290c <_impure_ptr>
   1604c:	00050663          	beqz	a0,16058 <fflush+0x18>
   16050:	03452783          	lw	a5,52(a0)
   16054:	00078c63          	beqz	a5,1606c <fflush+0x2c>
   16058:	00c59783          	lh	a5,12(a1)
   1605c:	00079663          	bnez	a5,16068 <fflush+0x28>
   16060:	00000513          	li	a0,0
   16064:	00008067          	ret
   16068:	d31ff06f          	j	15d98 <__sflush_r>
   1606c:	fe010113          	addi	sp,sp,-32
   16070:	00b12623          	sw	a1,12(sp)
   16074:	00a12423          	sw	a0,8(sp)
   16078:	00112e23          	sw	ra,28(sp)
   1607c:	f6cfa0ef          	jal	107e8 <__sinit>
   16080:	00c12583          	lw	a1,12(sp)
   16084:	00812503          	lw	a0,8(sp)
   16088:	00c59783          	lh	a5,12(a1)
   1608c:	02079663          	bnez	a5,160b8 <fflush+0x78>
   16090:	01c12083          	lw	ra,28(sp)
   16094:	00000513          	li	a0,0
   16098:	02010113          	addi	sp,sp,32
   1609c:	00008067          	ret
   160a0:	98018613          	addi	a2,gp,-1664 # 22200 <__sglue>
   160a4:	00000597          	auipc	a1,0x0
   160a8:	f4058593          	addi	a1,a1,-192 # 15fe4 <_fflush_r>
   160ac:	0000c517          	auipc	a0,0xc
   160b0:	16450513          	addi	a0,a0,356 # 22210 <_impure_data>
   160b4:	f88fa06f          	j	1083c <_fwalk_sglue>
   160b8:	01c12083          	lw	ra,28(sp)
   160bc:	02010113          	addi	sp,sp,32
   160c0:	cd9ff06f          	j	15d98 <__sflush_r>

000160c4 <__sfvwrite_r>:
   160c4:	00862783          	lw	a5,8(a2)
   160c8:	2c078463          	beqz	a5,16390 <__sfvwrite_r+0x2cc>
   160cc:	00c59683          	lh	a3,12(a1)
   160d0:	fd010113          	addi	sp,sp,-48
   160d4:	02812423          	sw	s0,40(sp)
   160d8:	01412c23          	sw	s4,24(sp)
   160dc:	01612823          	sw	s6,16(sp)
   160e0:	02112623          	sw	ra,44(sp)
   160e4:	0086f793          	andi	a5,a3,8
   160e8:	00060b13          	mv	s6,a2
   160ec:	00050a13          	mv	s4,a0
   160f0:	00058413          	mv	s0,a1
   160f4:	08078e63          	beqz	a5,16190 <__sfvwrite_r+0xcc>
   160f8:	0105a783          	lw	a5,16(a1)
   160fc:	08078a63          	beqz	a5,16190 <__sfvwrite_r+0xcc>
   16100:	02912223          	sw	s1,36(sp)
   16104:	03212023          	sw	s2,32(sp)
   16108:	01312e23          	sw	s3,28(sp)
   1610c:	01512a23          	sw	s5,20(sp)
   16110:	0026f793          	andi	a5,a3,2
   16114:	000b2483          	lw	s1,0(s6)
   16118:	0a078463          	beqz	a5,161c0 <__sfvwrite_r+0xfc>
   1611c:	02442783          	lw	a5,36(s0)
   16120:	01c42583          	lw	a1,28(s0)
   16124:	80000ab7          	lui	s5,0x80000
   16128:	00000993          	li	s3,0
   1612c:	00000913          	li	s2,0
   16130:	c00a8a93          	addi	s5,s5,-1024 # 7ffffc00 <__BSS_END__+0x7ffdced0>
   16134:	00098613          	mv	a2,s3
   16138:	000a0513          	mv	a0,s4
   1613c:	04090263          	beqz	s2,16180 <__sfvwrite_r+0xbc>
   16140:	00090693          	mv	a3,s2
   16144:	012af463          	bgeu	s5,s2,1614c <__sfvwrite_r+0x88>
   16148:	000a8693          	mv	a3,s5
   1614c:	000780e7          	jalr	a5
   16150:	46a05063          	blez	a0,165b0 <__sfvwrite_r+0x4ec>
   16154:	008b2783          	lw	a5,8(s6)
   16158:	00a989b3          	add	s3,s3,a0
   1615c:	40a90933          	sub	s2,s2,a0
   16160:	40a787b3          	sub	a5,a5,a0
   16164:	00fb2423          	sw	a5,8(s6)
   16168:	1a078663          	beqz	a5,16314 <__sfvwrite_r+0x250>
   1616c:	02442783          	lw	a5,36(s0)
   16170:	01c42583          	lw	a1,28(s0)
   16174:	00098613          	mv	a2,s3
   16178:	000a0513          	mv	a0,s4
   1617c:	fc0912e3          	bnez	s2,16140 <__sfvwrite_r+0x7c>
   16180:	0004a983          	lw	s3,0(s1)
   16184:	0044a903          	lw	s2,4(s1)
   16188:	00848493          	addi	s1,s1,8
   1618c:	fa9ff06f          	j	16134 <__sfvwrite_r+0x70>
   16190:	00040593          	mv	a1,s0
   16194:	000a0513          	mv	a0,s4
   16198:	43c000ef          	jal	165d4 <__swsetup_r>
   1619c:	1c051c63          	bnez	a0,16374 <__sfvwrite_r+0x2b0>
   161a0:	00c41683          	lh	a3,12(s0)
   161a4:	02912223          	sw	s1,36(sp)
   161a8:	03212023          	sw	s2,32(sp)
   161ac:	01312e23          	sw	s3,28(sp)
   161b0:	01512a23          	sw	s5,20(sp)
   161b4:	0026f793          	andi	a5,a3,2
   161b8:	000b2483          	lw	s1,0(s6)
   161bc:	f60790e3          	bnez	a5,1611c <__sfvwrite_r+0x58>
   161c0:	01712623          	sw	s7,12(sp)
   161c4:	01812423          	sw	s8,8(sp)
   161c8:	0016f793          	andi	a5,a3,1
   161cc:	1c079663          	bnez	a5,16398 <__sfvwrite_r+0x2d4>
   161d0:	00042783          	lw	a5,0(s0)
   161d4:	00842703          	lw	a4,8(s0)
   161d8:	80000ab7          	lui	s5,0x80000
   161dc:	01912223          	sw	s9,4(sp)
   161e0:	00000b93          	li	s7,0
   161e4:	00000993          	li	s3,0
   161e8:	fffa8a93          	addi	s5,s5,-1 # 7fffffff <__BSS_END__+0x7ffdd2cf>
   161ec:	00078513          	mv	a0,a5
   161f0:	00070c13          	mv	s8,a4
   161f4:	10098263          	beqz	s3,162f8 <__sfvwrite_r+0x234>
   161f8:	2006f613          	andi	a2,a3,512
   161fc:	28060863          	beqz	a2,1648c <__sfvwrite_r+0x3c8>
   16200:	00070c93          	mv	s9,a4
   16204:	32e9e663          	bltu	s3,a4,16530 <__sfvwrite_r+0x46c>
   16208:	4806f713          	andi	a4,a3,1152
   1620c:	08070a63          	beqz	a4,162a0 <__sfvwrite_r+0x1dc>
   16210:	01442603          	lw	a2,20(s0)
   16214:	01042583          	lw	a1,16(s0)
   16218:	00161713          	slli	a4,a2,0x1
   1621c:	00c70733          	add	a4,a4,a2
   16220:	40b78933          	sub	s2,a5,a1
   16224:	01f75c13          	srli	s8,a4,0x1f
   16228:	00ec0c33          	add	s8,s8,a4
   1622c:	00190793          	addi	a5,s2,1
   16230:	401c5c13          	srai	s8,s8,0x1
   16234:	013787b3          	add	a5,a5,s3
   16238:	000c0613          	mv	a2,s8
   1623c:	00fc7663          	bgeu	s8,a5,16248 <__sfvwrite_r+0x184>
   16240:	00078c13          	mv	s8,a5
   16244:	00078613          	mv	a2,a5
   16248:	4006f693          	andi	a3,a3,1024
   1624c:	30068e63          	beqz	a3,16568 <__sfvwrite_r+0x4a4>
   16250:	00060593          	mv	a1,a2
   16254:	000a0513          	mv	a0,s4
   16258:	b18fb0ef          	jal	11570 <_malloc_r>
   1625c:	00050c93          	mv	s9,a0
   16260:	34050c63          	beqz	a0,165b8 <__sfvwrite_r+0x4f4>
   16264:	01042583          	lw	a1,16(s0)
   16268:	00090613          	mv	a2,s2
   1626c:	285000ef          	jal	16cf0 <memcpy>
   16270:	00c45783          	lhu	a5,12(s0)
   16274:	b7f7f793          	andi	a5,a5,-1153
   16278:	0807e793          	ori	a5,a5,128
   1627c:	00f41623          	sh	a5,12(s0)
   16280:	012c8533          	add	a0,s9,s2
   16284:	412c07b3          	sub	a5,s8,s2
   16288:	01942823          	sw	s9,16(s0)
   1628c:	01842a23          	sw	s8,20(s0)
   16290:	00a42023          	sw	a0,0(s0)
   16294:	00098c13          	mv	s8,s3
   16298:	00f42423          	sw	a5,8(s0)
   1629c:	00098c93          	mv	s9,s3
   162a0:	000c8613          	mv	a2,s9
   162a4:	000b8593          	mv	a1,s7
   162a8:	13d000ef          	jal	16be4 <memmove>
   162ac:	00842703          	lw	a4,8(s0)
   162b0:	00042783          	lw	a5,0(s0)
   162b4:	00098913          	mv	s2,s3
   162b8:	41870733          	sub	a4,a4,s8
   162bc:	019787b3          	add	a5,a5,s9
   162c0:	00e42423          	sw	a4,8(s0)
   162c4:	00f42023          	sw	a5,0(s0)
   162c8:	00000993          	li	s3,0
   162cc:	008b2783          	lw	a5,8(s6)
   162d0:	012b8bb3          	add	s7,s7,s2
   162d4:	412787b3          	sub	a5,a5,s2
   162d8:	00fb2423          	sw	a5,8(s6)
   162dc:	02078663          	beqz	a5,16308 <__sfvwrite_r+0x244>
   162e0:	00042783          	lw	a5,0(s0)
   162e4:	00842703          	lw	a4,8(s0)
   162e8:	00c41683          	lh	a3,12(s0)
   162ec:	00078513          	mv	a0,a5
   162f0:	00070c13          	mv	s8,a4
   162f4:	f00992e3          	bnez	s3,161f8 <__sfvwrite_r+0x134>
   162f8:	0004ab83          	lw	s7,0(s1)
   162fc:	0044a983          	lw	s3,4(s1)
   16300:	00848493          	addi	s1,s1,8
   16304:	ee9ff06f          	j	161ec <__sfvwrite_r+0x128>
   16308:	00c12b83          	lw	s7,12(sp)
   1630c:	00812c03          	lw	s8,8(sp)
   16310:	00412c83          	lw	s9,4(sp)
   16314:	02c12083          	lw	ra,44(sp)
   16318:	02812403          	lw	s0,40(sp)
   1631c:	02412483          	lw	s1,36(sp)
   16320:	02012903          	lw	s2,32(sp)
   16324:	01c12983          	lw	s3,28(sp)
   16328:	01412a83          	lw	s5,20(sp)
   1632c:	01812a03          	lw	s4,24(sp)
   16330:	01012b03          	lw	s6,16(sp)
   16334:	00000513          	li	a0,0
   16338:	03010113          	addi	sp,sp,48
   1633c:	00008067          	ret
   16340:	00040593          	mv	a1,s0
   16344:	000a0513          	mv	a0,s4
   16348:	c9dff0ef          	jal	15fe4 <_fflush_r>
   1634c:	0a050e63          	beqz	a0,16408 <__sfvwrite_r+0x344>
   16350:	00c41783          	lh	a5,12(s0)
   16354:	00c12b83          	lw	s7,12(sp)
   16358:	00812c03          	lw	s8,8(sp)
   1635c:	02412483          	lw	s1,36(sp)
   16360:	02012903          	lw	s2,32(sp)
   16364:	01c12983          	lw	s3,28(sp)
   16368:	01412a83          	lw	s5,20(sp)
   1636c:	0407e793          	ori	a5,a5,64
   16370:	00f41623          	sh	a5,12(s0)
   16374:	02c12083          	lw	ra,44(sp)
   16378:	02812403          	lw	s0,40(sp)
   1637c:	01812a03          	lw	s4,24(sp)
   16380:	01012b03          	lw	s6,16(sp)
   16384:	fff00513          	li	a0,-1
   16388:	03010113          	addi	sp,sp,48
   1638c:	00008067          	ret
   16390:	00000513          	li	a0,0
   16394:	00008067          	ret
   16398:	00000a93          	li	s5,0
   1639c:	00000513          	li	a0,0
   163a0:	00000c13          	li	s8,0
   163a4:	00000993          	li	s3,0
   163a8:	08098263          	beqz	s3,1642c <__sfvwrite_r+0x368>
   163ac:	08050a63          	beqz	a0,16440 <__sfvwrite_r+0x37c>
   163b0:	000a8793          	mv	a5,s5
   163b4:	00098b93          	mv	s7,s3
   163b8:	0137f463          	bgeu	a5,s3,163c0 <__sfvwrite_r+0x2fc>
   163bc:	00078b93          	mv	s7,a5
   163c0:	00042503          	lw	a0,0(s0)
   163c4:	01042783          	lw	a5,16(s0)
   163c8:	00842903          	lw	s2,8(s0)
   163cc:	01442683          	lw	a3,20(s0)
   163d0:	00a7f663          	bgeu	a5,a0,163dc <__sfvwrite_r+0x318>
   163d4:	00d90933          	add	s2,s2,a3
   163d8:	09794463          	blt	s2,s7,16460 <__sfvwrite_r+0x39c>
   163dc:	16dbc063          	blt	s7,a3,1653c <__sfvwrite_r+0x478>
   163e0:	02442783          	lw	a5,36(s0)
   163e4:	01c42583          	lw	a1,28(s0)
   163e8:	000c0613          	mv	a2,s8
   163ec:	000a0513          	mv	a0,s4
   163f0:	000780e7          	jalr	a5
   163f4:	00050913          	mv	s2,a0
   163f8:	f4a05ce3          	blez	a0,16350 <__sfvwrite_r+0x28c>
   163fc:	412a8ab3          	sub	s5,s5,s2
   16400:	00100513          	li	a0,1
   16404:	f20a8ee3          	beqz	s5,16340 <__sfvwrite_r+0x27c>
   16408:	008b2783          	lw	a5,8(s6)
   1640c:	012c0c33          	add	s8,s8,s2
   16410:	412989b3          	sub	s3,s3,s2
   16414:	412787b3          	sub	a5,a5,s2
   16418:	00fb2423          	sw	a5,8(s6)
   1641c:	f80796e3          	bnez	a5,163a8 <__sfvwrite_r+0x2e4>
   16420:	00c12b83          	lw	s7,12(sp)
   16424:	00812c03          	lw	s8,8(sp)
   16428:	eedff06f          	j	16314 <__sfvwrite_r+0x250>
   1642c:	0044a983          	lw	s3,4(s1)
   16430:	00048793          	mv	a5,s1
   16434:	00848493          	addi	s1,s1,8
   16438:	fe098ae3          	beqz	s3,1642c <__sfvwrite_r+0x368>
   1643c:	0007ac03          	lw	s8,0(a5)
   16440:	00098613          	mv	a2,s3
   16444:	00a00593          	li	a1,10
   16448:	000c0513          	mv	a0,s8
   1644c:	464000ef          	jal	168b0 <memchr>
   16450:	14050a63          	beqz	a0,165a4 <__sfvwrite_r+0x4e0>
   16454:	00150513          	addi	a0,a0,1
   16458:	41850ab3          	sub	s5,a0,s8
   1645c:	f55ff06f          	j	163b0 <__sfvwrite_r+0x2ec>
   16460:	000c0593          	mv	a1,s8
   16464:	00090613          	mv	a2,s2
   16468:	77c000ef          	jal	16be4 <memmove>
   1646c:	00042783          	lw	a5,0(s0)
   16470:	00040593          	mv	a1,s0
   16474:	000a0513          	mv	a0,s4
   16478:	012787b3          	add	a5,a5,s2
   1647c:	00f42023          	sw	a5,0(s0)
   16480:	b65ff0ef          	jal	15fe4 <_fflush_r>
   16484:	f6050ce3          	beqz	a0,163fc <__sfvwrite_r+0x338>
   16488:	ec9ff06f          	j	16350 <__sfvwrite_r+0x28c>
   1648c:	01042683          	lw	a3,16(s0)
   16490:	04f6e263          	bltu	a3,a5,164d4 <__sfvwrite_r+0x410>
   16494:	01442603          	lw	a2,20(s0)
   16498:	02c9ee63          	bltu	s3,a2,164d4 <__sfvwrite_r+0x410>
   1649c:	00098793          	mv	a5,s3
   164a0:	013af463          	bgeu	s5,s3,164a8 <__sfvwrite_r+0x3e4>
   164a4:	000a8793          	mv	a5,s5
   164a8:	02c7e6b3          	rem	a3,a5,a2
   164ac:	02442703          	lw	a4,36(s0)
   164b0:	01c42583          	lw	a1,28(s0)
   164b4:	000b8613          	mv	a2,s7
   164b8:	000a0513          	mv	a0,s4
   164bc:	40d786b3          	sub	a3,a5,a3
   164c0:	000700e7          	jalr	a4
   164c4:	00050913          	mv	s2,a0
   164c8:	04a05a63          	blez	a0,1651c <__sfvwrite_r+0x458>
   164cc:	412989b3          	sub	s3,s3,s2
   164d0:	dfdff06f          	j	162cc <__sfvwrite_r+0x208>
   164d4:	00070913          	mv	s2,a4
   164d8:	00e9f463          	bgeu	s3,a4,164e0 <__sfvwrite_r+0x41c>
   164dc:	00098913          	mv	s2,s3
   164e0:	00078513          	mv	a0,a5
   164e4:	00090613          	mv	a2,s2
   164e8:	000b8593          	mv	a1,s7
   164ec:	6f8000ef          	jal	16be4 <memmove>
   164f0:	00842703          	lw	a4,8(s0)
   164f4:	00042783          	lw	a5,0(s0)
   164f8:	41270733          	sub	a4,a4,s2
   164fc:	012787b3          	add	a5,a5,s2
   16500:	00e42423          	sw	a4,8(s0)
   16504:	00f42023          	sw	a5,0(s0)
   16508:	fc0712e3          	bnez	a4,164cc <__sfvwrite_r+0x408>
   1650c:	00040593          	mv	a1,s0
   16510:	000a0513          	mv	a0,s4
   16514:	ad1ff0ef          	jal	15fe4 <_fflush_r>
   16518:	fa050ae3          	beqz	a0,164cc <__sfvwrite_r+0x408>
   1651c:	00c41783          	lh	a5,12(s0)
   16520:	00c12b83          	lw	s7,12(sp)
   16524:	00812c03          	lw	s8,8(sp)
   16528:	00412c83          	lw	s9,4(sp)
   1652c:	e31ff06f          	j	1635c <__sfvwrite_r+0x298>
   16530:	00098c13          	mv	s8,s3
   16534:	00098c93          	mv	s9,s3
   16538:	d69ff06f          	j	162a0 <__sfvwrite_r+0x1dc>
   1653c:	000b8613          	mv	a2,s7
   16540:	000c0593          	mv	a1,s8
   16544:	6a0000ef          	jal	16be4 <memmove>
   16548:	00842703          	lw	a4,8(s0)
   1654c:	00042783          	lw	a5,0(s0)
   16550:	000b8913          	mv	s2,s7
   16554:	41770733          	sub	a4,a4,s7
   16558:	017787b3          	add	a5,a5,s7
   1655c:	00e42423          	sw	a4,8(s0)
   16560:	00f42023          	sw	a5,0(s0)
   16564:	e99ff06f          	j	163fc <__sfvwrite_r+0x338>
   16568:	000a0513          	mv	a0,s4
   1656c:	440040ef          	jal	1a9ac <_realloc_r>
   16570:	00050c93          	mv	s9,a0
   16574:	d00516e3          	bnez	a0,16280 <__sfvwrite_r+0x1bc>
   16578:	01042583          	lw	a1,16(s0)
   1657c:	000a0513          	mv	a0,s4
   16580:	cedfa0ef          	jal	1126c <_free_r>
   16584:	00c41783          	lh	a5,12(s0)
   16588:	00c00713          	li	a4,12
   1658c:	00c12b83          	lw	s7,12(sp)
   16590:	00812c03          	lw	s8,8(sp)
   16594:	00412c83          	lw	s9,4(sp)
   16598:	00ea2023          	sw	a4,0(s4)
   1659c:	f7f7f793          	andi	a5,a5,-129
   165a0:	dbdff06f          	j	1635c <__sfvwrite_r+0x298>
   165a4:	00198793          	addi	a5,s3,1
   165a8:	00078a93          	mv	s5,a5
   165ac:	e09ff06f          	j	163b4 <__sfvwrite_r+0x2f0>
   165b0:	00c41783          	lh	a5,12(s0)
   165b4:	da9ff06f          	j	1635c <__sfvwrite_r+0x298>
   165b8:	00c00713          	li	a4,12
   165bc:	00c41783          	lh	a5,12(s0)
   165c0:	00c12b83          	lw	s7,12(sp)
   165c4:	00812c03          	lw	s8,8(sp)
   165c8:	00412c83          	lw	s9,4(sp)
   165cc:	00ea2023          	sw	a4,0(s4)
   165d0:	d8dff06f          	j	1635c <__sfvwrite_r+0x298>

000165d4 <__swsetup_r>:
   165d4:	ff010113          	addi	sp,sp,-16
   165d8:	00812423          	sw	s0,8(sp)
   165dc:	00912223          	sw	s1,4(sp)
   165e0:	00112623          	sw	ra,12(sp)
   165e4:	08c1a783          	lw	a5,140(gp) # 2290c <_impure_ptr>
   165e8:	00050493          	mv	s1,a0
   165ec:	00058413          	mv	s0,a1
   165f0:	00078663          	beqz	a5,165fc <__swsetup_r+0x28>
   165f4:	0347a703          	lw	a4,52(a5)
   165f8:	0e070c63          	beqz	a4,166f0 <__swsetup_r+0x11c>
   165fc:	00c41783          	lh	a5,12(s0)
   16600:	0087f713          	andi	a4,a5,8
   16604:	06070a63          	beqz	a4,16678 <__swsetup_r+0xa4>
   16608:	01042703          	lw	a4,16(s0)
   1660c:	08070663          	beqz	a4,16698 <__swsetup_r+0xc4>
   16610:	0017f693          	andi	a3,a5,1
   16614:	02068863          	beqz	a3,16644 <__swsetup_r+0x70>
   16618:	01442683          	lw	a3,20(s0)
   1661c:	00042423          	sw	zero,8(s0)
   16620:	00000513          	li	a0,0
   16624:	40d006b3          	neg	a3,a3
   16628:	00d42c23          	sw	a3,24(s0)
   1662c:	02070a63          	beqz	a4,16660 <__swsetup_r+0x8c>
   16630:	00c12083          	lw	ra,12(sp)
   16634:	00812403          	lw	s0,8(sp)
   16638:	00412483          	lw	s1,4(sp)
   1663c:	01010113          	addi	sp,sp,16
   16640:	00008067          	ret
   16644:	0027f693          	andi	a3,a5,2
   16648:	00000613          	li	a2,0
   1664c:	00069463          	bnez	a3,16654 <__swsetup_r+0x80>
   16650:	01442603          	lw	a2,20(s0)
   16654:	00c42423          	sw	a2,8(s0)
   16658:	00000513          	li	a0,0
   1665c:	fc071ae3          	bnez	a4,16630 <__swsetup_r+0x5c>
   16660:	0807f713          	andi	a4,a5,128
   16664:	fc0706e3          	beqz	a4,16630 <__swsetup_r+0x5c>
   16668:	0407e793          	ori	a5,a5,64
   1666c:	00f41623          	sh	a5,12(s0)
   16670:	fff00513          	li	a0,-1
   16674:	fbdff06f          	j	16630 <__swsetup_r+0x5c>
   16678:	0107f713          	andi	a4,a5,16
   1667c:	08070063          	beqz	a4,166fc <__swsetup_r+0x128>
   16680:	0047f713          	andi	a4,a5,4
   16684:	02071c63          	bnez	a4,166bc <__swsetup_r+0xe8>
   16688:	01042703          	lw	a4,16(s0)
   1668c:	0087e793          	ori	a5,a5,8
   16690:	00f41623          	sh	a5,12(s0)
   16694:	f6071ee3          	bnez	a4,16610 <__swsetup_r+0x3c>
   16698:	2807f693          	andi	a3,a5,640
   1669c:	20000613          	li	a2,512
   166a0:	f6c688e3          	beq	a3,a2,16610 <__swsetup_r+0x3c>
   166a4:	00040593          	mv	a1,s0
   166a8:	00048513          	mv	a0,s1
   166ac:	1a9040ef          	jal	1b054 <__smakebuf_r>
   166b0:	00c41783          	lh	a5,12(s0)
   166b4:	01042703          	lw	a4,16(s0)
   166b8:	f59ff06f          	j	16610 <__swsetup_r+0x3c>
   166bc:	03042583          	lw	a1,48(s0)
   166c0:	00058e63          	beqz	a1,166dc <__swsetup_r+0x108>
   166c4:	04040713          	addi	a4,s0,64
   166c8:	00e58863          	beq	a1,a4,166d8 <__swsetup_r+0x104>
   166cc:	00048513          	mv	a0,s1
   166d0:	b9dfa0ef          	jal	1126c <_free_r>
   166d4:	00c41783          	lh	a5,12(s0)
   166d8:	02042823          	sw	zero,48(s0)
   166dc:	01042703          	lw	a4,16(s0)
   166e0:	fdb7f793          	andi	a5,a5,-37
   166e4:	00042223          	sw	zero,4(s0)
   166e8:	00e42023          	sw	a4,0(s0)
   166ec:	fa1ff06f          	j	1668c <__swsetup_r+0xb8>
   166f0:	00078513          	mv	a0,a5
   166f4:	8f4fa0ef          	jal	107e8 <__sinit>
   166f8:	f05ff06f          	j	165fc <__swsetup_r+0x28>
   166fc:	00900713          	li	a4,9
   16700:	00e4a023          	sw	a4,0(s1)
   16704:	0407e793          	ori	a5,a5,64
   16708:	00f41623          	sh	a5,12(s0)
   1670c:	fff00513          	li	a0,-1
   16710:	f21ff06f          	j	16630 <__swsetup_r+0x5c>

00016714 <__fputwc>:
   16714:	fe010113          	addi	sp,sp,-32
   16718:	00812c23          	sw	s0,24(sp)
   1671c:	00912a23          	sw	s1,20(sp)
   16720:	01212823          	sw	s2,16(sp)
   16724:	00112e23          	sw	ra,28(sp)
   16728:	00050913          	mv	s2,a0
   1672c:	00058493          	mv	s1,a1
   16730:	00060413          	mv	s0,a2
   16734:	368000ef          	jal	16a9c <__locale_mb_cur_max>
   16738:	00100793          	li	a5,1
   1673c:	00f51c63          	bne	a0,a5,16754 <__fputwc+0x40>
   16740:	fff48793          	addi	a5,s1,-1
   16744:	0fe00713          	li	a4,254
   16748:	00f76663          	bltu	a4,a5,16754 <__fputwc+0x40>
   1674c:	00910623          	sb	s1,12(sp)
   16750:	0240006f          	j	16774 <__fputwc+0x60>
   16754:	05c40693          	addi	a3,s0,92
   16758:	00048613          	mv	a2,s1
   1675c:	00c10593          	addi	a1,sp,12
   16760:	00090513          	mv	a0,s2
   16764:	7f0040ef          	jal	1af54 <_wcrtomb_r>
   16768:	fff00793          	li	a5,-1
   1676c:	08f50463          	beq	a0,a5,167f4 <__fputwc+0xe0>
   16770:	02050c63          	beqz	a0,167a8 <__fputwc+0x94>
   16774:	00842783          	lw	a5,8(s0)
   16778:	00c14583          	lbu	a1,12(sp)
   1677c:	fff78793          	addi	a5,a5,-1
   16780:	00f42423          	sw	a5,8(s0)
   16784:	0007da63          	bgez	a5,16798 <__fputwc+0x84>
   16788:	01842703          	lw	a4,24(s0)
   1678c:	02e7cc63          	blt	a5,a4,167c4 <__fputwc+0xb0>
   16790:	00a00793          	li	a5,10
   16794:	02f58863          	beq	a1,a5,167c4 <__fputwc+0xb0>
   16798:	00042783          	lw	a5,0(s0)
   1679c:	00178713          	addi	a4,a5,1
   167a0:	00e42023          	sw	a4,0(s0)
   167a4:	00b78023          	sb	a1,0(a5)
   167a8:	01c12083          	lw	ra,28(sp)
   167ac:	01812403          	lw	s0,24(sp)
   167b0:	01012903          	lw	s2,16(sp)
   167b4:	00048513          	mv	a0,s1
   167b8:	01412483          	lw	s1,20(sp)
   167bc:	02010113          	addi	sp,sp,32
   167c0:	00008067          	ret
   167c4:	00040613          	mv	a2,s0
   167c8:	00090513          	mv	a0,s2
   167cc:	2f9040ef          	jal	1b2c4 <__swbuf_r>
   167d0:	fff00793          	li	a5,-1
   167d4:	fcf51ae3          	bne	a0,a5,167a8 <__fputwc+0x94>
   167d8:	fff00513          	li	a0,-1
   167dc:	01c12083          	lw	ra,28(sp)
   167e0:	01812403          	lw	s0,24(sp)
   167e4:	01412483          	lw	s1,20(sp)
   167e8:	01012903          	lw	s2,16(sp)
   167ec:	02010113          	addi	sp,sp,32
   167f0:	00008067          	ret
   167f4:	00c45783          	lhu	a5,12(s0)
   167f8:	fff00513          	li	a0,-1
   167fc:	0407e793          	ori	a5,a5,64
   16800:	00f41623          	sh	a5,12(s0)
   16804:	fd9ff06f          	j	167dc <__fputwc+0xc8>

00016808 <_fputwc_r>:
   16808:	00c61783          	lh	a5,12(a2)
   1680c:	01279713          	slli	a4,a5,0x12
   16810:	02074063          	bltz	a4,16830 <_fputwc_r+0x28>
   16814:	06462703          	lw	a4,100(a2)
   16818:	000026b7          	lui	a3,0x2
   1681c:	00d7e7b3          	or	a5,a5,a3
   16820:	000026b7          	lui	a3,0x2
   16824:	00d76733          	or	a4,a4,a3
   16828:	00f61623          	sh	a5,12(a2)
   1682c:	06e62223          	sw	a4,100(a2)
   16830:	ee5ff06f          	j	16714 <__fputwc>

00016834 <fputwc>:
   16834:	fe010113          	addi	sp,sp,-32
   16838:	00812c23          	sw	s0,24(sp)
   1683c:	00112e23          	sw	ra,28(sp)
   16840:	08c1a403          	lw	s0,140(gp) # 2290c <_impure_ptr>
   16844:	00058613          	mv	a2,a1
   16848:	00050593          	mv	a1,a0
   1684c:	00040663          	beqz	s0,16858 <fputwc+0x24>
   16850:	03442783          	lw	a5,52(s0)
   16854:	04078063          	beqz	a5,16894 <fputwc+0x60>
   16858:	00c61783          	lh	a5,12(a2)
   1685c:	01279713          	slli	a4,a5,0x12
   16860:	02074063          	bltz	a4,16880 <fputwc+0x4c>
   16864:	06462703          	lw	a4,100(a2)
   16868:	000026b7          	lui	a3,0x2
   1686c:	00d7e7b3          	or	a5,a5,a3
   16870:	000026b7          	lui	a3,0x2
   16874:	00d76733          	or	a4,a4,a3
   16878:	00f61623          	sh	a5,12(a2)
   1687c:	06e62223          	sw	a4,100(a2)
   16880:	00040513          	mv	a0,s0
   16884:	01812403          	lw	s0,24(sp)
   16888:	01c12083          	lw	ra,28(sp)
   1688c:	02010113          	addi	sp,sp,32
   16890:	e85ff06f          	j	16714 <__fputwc>
   16894:	00a12423          	sw	a0,8(sp)
   16898:	00040513          	mv	a0,s0
   1689c:	00c12623          	sw	a2,12(sp)
   168a0:	f49f90ef          	jal	107e8 <__sinit>
   168a4:	00c12603          	lw	a2,12(sp)
   168a8:	00812583          	lw	a1,8(sp)
   168ac:	fadff06f          	j	16858 <fputwc+0x24>

000168b0 <memchr>:
   168b0:	00357793          	andi	a5,a0,3
   168b4:	0ff5f693          	zext.b	a3,a1
   168b8:	02078a63          	beqz	a5,168ec <memchr+0x3c>
   168bc:	fff60793          	addi	a5,a2,-1
   168c0:	02060e63          	beqz	a2,168fc <memchr+0x4c>
   168c4:	fff00613          	li	a2,-1
   168c8:	0180006f          	j	168e0 <memchr+0x30>
   168cc:	00150513          	addi	a0,a0,1
   168d0:	00357713          	andi	a4,a0,3
   168d4:	00070e63          	beqz	a4,168f0 <memchr+0x40>
   168d8:	fff78793          	addi	a5,a5,-1
   168dc:	02c78063          	beq	a5,a2,168fc <memchr+0x4c>
   168e0:	00054703          	lbu	a4,0(a0)
   168e4:	fed714e3          	bne	a4,a3,168cc <memchr+0x1c>
   168e8:	00008067          	ret
   168ec:	00060793          	mv	a5,a2
   168f0:	00300713          	li	a4,3
   168f4:	00f76863          	bltu	a4,a5,16904 <memchr+0x54>
   168f8:	06079063          	bnez	a5,16958 <memchr+0xa8>
   168fc:	00000513          	li	a0,0
   16900:	00008067          	ret
   16904:	0ff5f593          	zext.b	a1,a1
   16908:	00859713          	slli	a4,a1,0x8
   1690c:	00b705b3          	add	a1,a4,a1
   16910:	01059713          	slli	a4,a1,0x10
   16914:	feff08b7          	lui	a7,0xfeff0
   16918:	80808837          	lui	a6,0x80808
   1691c:	00e585b3          	add	a1,a1,a4
   16920:	eff88893          	addi	a7,a7,-257 # fefefeff <__BSS_END__+0xfefcd1cf>
   16924:	08080813          	addi	a6,a6,128 # 80808080 <__BSS_END__+0x807e5350>
   16928:	00300313          	li	t1,3
   1692c:	0100006f          	j	1693c <memchr+0x8c>
   16930:	ffc78793          	addi	a5,a5,-4
   16934:	00450513          	addi	a0,a0,4
   16938:	fcf370e3          	bgeu	t1,a5,168f8 <memchr+0x48>
   1693c:	00052703          	lw	a4,0(a0)
   16940:	00e5c733          	xor	a4,a1,a4
   16944:	01170633          	add	a2,a4,a7
   16948:	fff74713          	not	a4,a4
   1694c:	00e67733          	and	a4,a2,a4
   16950:	01077733          	and	a4,a4,a6
   16954:	fc070ee3          	beqz	a4,16930 <memchr+0x80>
   16958:	00f507b3          	add	a5,a0,a5
   1695c:	00c0006f          	j	16968 <memchr+0xb8>
   16960:	00150513          	addi	a0,a0,1
   16964:	f8a78ce3          	beq	a5,a0,168fc <memchr+0x4c>
   16968:	00054703          	lbu	a4,0(a0)
   1696c:	fed71ae3          	bne	a4,a3,16960 <memchr+0xb0>
   16970:	00008067          	ret

00016974 <strncpy>:
   16974:	00a5e7b3          	or	a5,a1,a0
   16978:	0037f793          	andi	a5,a5,3
   1697c:	00079663          	bnez	a5,16988 <strncpy+0x14>
   16980:	00300793          	li	a5,3
   16984:	04c7e663          	bltu	a5,a2,169d0 <strncpy+0x5c>
   16988:	00050713          	mv	a4,a0
   1698c:	01c0006f          	j	169a8 <strncpy+0x34>
   16990:	fff5c683          	lbu	a3,-1(a1)
   16994:	fff60813          	addi	a6,a2,-1
   16998:	fed78fa3          	sb	a3,-1(a5)
   1699c:	00068e63          	beqz	a3,169b8 <strncpy+0x44>
   169a0:	00078713          	mv	a4,a5
   169a4:	00080613          	mv	a2,a6
   169a8:	00158593          	addi	a1,a1,1
   169ac:	00170793          	addi	a5,a4,1
   169b0:	fe0610e3          	bnez	a2,16990 <strncpy+0x1c>
   169b4:	00008067          	ret
   169b8:	00c70733          	add	a4,a4,a2
   169bc:	06080063          	beqz	a6,16a1c <strncpy+0xa8>
   169c0:	00178793          	addi	a5,a5,1
   169c4:	fe078fa3          	sb	zero,-1(a5)
   169c8:	fee79ce3          	bne	a5,a4,169c0 <strncpy+0x4c>
   169cc:	00008067          	ret
   169d0:	feff0337          	lui	t1,0xfeff0
   169d4:	808088b7          	lui	a7,0x80808
   169d8:	00050713          	mv	a4,a0
   169dc:	eff30313          	addi	t1,t1,-257 # fefefeff <__BSS_END__+0xfefcd1cf>
   169e0:	08088893          	addi	a7,a7,128 # 80808080 <__BSS_END__+0x807e5350>
   169e4:	00300e13          	li	t3,3
   169e8:	0180006f          	j	16a00 <strncpy+0x8c>
   169ec:	00d72023          	sw	a3,0(a4)
   169f0:	ffc60613          	addi	a2,a2,-4
   169f4:	00470713          	addi	a4,a4,4
   169f8:	00458593          	addi	a1,a1,4
   169fc:	face76e3          	bgeu	t3,a2,169a8 <strncpy+0x34>
   16a00:	0005a683          	lw	a3,0(a1)
   16a04:	006687b3          	add	a5,a3,t1
   16a08:	fff6c813          	not	a6,a3
   16a0c:	0107f7b3          	and	a5,a5,a6
   16a10:	0117f7b3          	and	a5,a5,a7
   16a14:	fc078ce3          	beqz	a5,169ec <strncpy+0x78>
   16a18:	f91ff06f          	j	169a8 <strncpy+0x34>
   16a1c:	00008067          	ret

00016a20 <_setlocale_r>:
   16a20:	04060063          	beqz	a2,16a60 <_setlocale_r+0x40>
   16a24:	ff010113          	addi	sp,sp,-16
   16a28:	0000a597          	auipc	a1,0xa
   16a2c:	15858593          	addi	a1,a1,344 # 20b80 <_exit+0x1d0>
   16a30:	00060513          	mv	a0,a2
   16a34:	00812423          	sw	s0,8(sp)
   16a38:	00112623          	sw	ra,12(sp)
   16a3c:	00060413          	mv	s0,a2
   16a40:	454000ef          	jal	16e94 <strcmp>
   16a44:	02051463          	bnez	a0,16a6c <_setlocale_r+0x4c>
   16a48:	0000a517          	auipc	a0,0xa
   16a4c:	13450513          	addi	a0,a0,308 # 20b7c <_exit+0x1cc>
   16a50:	00c12083          	lw	ra,12(sp)
   16a54:	00812403          	lw	s0,8(sp)
   16a58:	01010113          	addi	sp,sp,16
   16a5c:	00008067          	ret
   16a60:	0000a517          	auipc	a0,0xa
   16a64:	11c50513          	addi	a0,a0,284 # 20b7c <_exit+0x1cc>
   16a68:	00008067          	ret
   16a6c:	0000a597          	auipc	a1,0xa
   16a70:	11058593          	addi	a1,a1,272 # 20b7c <_exit+0x1cc>
   16a74:	00040513          	mv	a0,s0
   16a78:	41c000ef          	jal	16e94 <strcmp>
   16a7c:	fc0506e3          	beqz	a0,16a48 <_setlocale_r+0x28>
   16a80:	0000a597          	auipc	a1,0xa
   16a84:	1d458593          	addi	a1,a1,468 # 20c54 <_exit+0x2a4>
   16a88:	00040513          	mv	a0,s0
   16a8c:	408000ef          	jal	16e94 <strcmp>
   16a90:	fa050ce3          	beqz	a0,16a48 <_setlocale_r+0x28>
   16a94:	00000513          	li	a0,0
   16a98:	fb9ff06f          	j	16a50 <_setlocale_r+0x30>

00016a9c <__locale_mb_cur_max>:
   16a9c:	fe01c503          	lbu	a0,-32(gp) # 22860 <__global_locale+0x128>
   16aa0:	00008067          	ret

00016aa4 <setlocale>:
   16aa4:	04058063          	beqz	a1,16ae4 <setlocale+0x40>
   16aa8:	ff010113          	addi	sp,sp,-16
   16aac:	00812423          	sw	s0,8(sp)
   16ab0:	00058413          	mv	s0,a1
   16ab4:	00040513          	mv	a0,s0
   16ab8:	0000a597          	auipc	a1,0xa
   16abc:	0c858593          	addi	a1,a1,200 # 20b80 <_exit+0x1d0>
   16ac0:	00112623          	sw	ra,12(sp)
   16ac4:	3d0000ef          	jal	16e94 <strcmp>
   16ac8:	02051463          	bnez	a0,16af0 <setlocale+0x4c>
   16acc:	0000a517          	auipc	a0,0xa
   16ad0:	0b050513          	addi	a0,a0,176 # 20b7c <_exit+0x1cc>
   16ad4:	00c12083          	lw	ra,12(sp)
   16ad8:	00812403          	lw	s0,8(sp)
   16adc:	01010113          	addi	sp,sp,16
   16ae0:	00008067          	ret
   16ae4:	0000a517          	auipc	a0,0xa
   16ae8:	09850513          	addi	a0,a0,152 # 20b7c <_exit+0x1cc>
   16aec:	00008067          	ret
   16af0:	0000a597          	auipc	a1,0xa
   16af4:	08c58593          	addi	a1,a1,140 # 20b7c <_exit+0x1cc>
   16af8:	00040513          	mv	a0,s0
   16afc:	398000ef          	jal	16e94 <strcmp>
   16b00:	fc0506e3          	beqz	a0,16acc <setlocale+0x28>
   16b04:	0000a597          	auipc	a1,0xa
   16b08:	15058593          	addi	a1,a1,336 # 20c54 <_exit+0x2a4>
   16b0c:	00040513          	mv	a0,s0
   16b10:	384000ef          	jal	16e94 <strcmp>
   16b14:	fa050ce3          	beqz	a0,16acc <setlocale+0x28>
   16b18:	00000513          	li	a0,0
   16b1c:	fb9ff06f          	j	16ad4 <setlocale+0x30>

00016b20 <__localeconv_l>:
   16b20:	0f050513          	addi	a0,a0,240
   16b24:	00008067          	ret

00016b28 <_localeconv_r>:
   16b28:	fa818513          	addi	a0,gp,-88 # 22828 <__global_locale+0xf0>
   16b2c:	00008067          	ret

00016b30 <localeconv>:
   16b30:	fa818513          	addi	a0,gp,-88 # 22828 <__global_locale+0xf0>
   16b34:	00008067          	ret

00016b38 <_sbrk_r>:
   16b38:	ff010113          	addi	sp,sp,-16
   16b3c:	00812423          	sw	s0,8(sp)
   16b40:	00050413          	mv	s0,a0
   16b44:	00058513          	mv	a0,a1
   16b48:	0801ae23          	sw	zero,156(gp) # 2291c <errno>
   16b4c:	00112623          	sw	ra,12(sp)
   16b50:	621090ef          	jal	20970 <_sbrk>
   16b54:	fff00793          	li	a5,-1
   16b58:	00f50a63          	beq	a0,a5,16b6c <_sbrk_r+0x34>
   16b5c:	00c12083          	lw	ra,12(sp)
   16b60:	00812403          	lw	s0,8(sp)
   16b64:	01010113          	addi	sp,sp,16
   16b68:	00008067          	ret
   16b6c:	09c1a783          	lw	a5,156(gp) # 2291c <errno>
   16b70:	fe0786e3          	beqz	a5,16b5c <_sbrk_r+0x24>
   16b74:	00c12083          	lw	ra,12(sp)
   16b78:	00f42023          	sw	a5,0(s0)
   16b7c:	00812403          	lw	s0,8(sp)
   16b80:	01010113          	addi	sp,sp,16
   16b84:	00008067          	ret

00016b88 <__libc_fini_array>:
   16b88:	ff010113          	addi	sp,sp,-16
   16b8c:	00812423          	sw	s0,8(sp)
   16b90:	0000b797          	auipc	a5,0xb
   16b94:	4e878793          	addi	a5,a5,1256 # 22078 <__do_global_dtors_aux_fini_array_entry>
   16b98:	0000b417          	auipc	s0,0xb
   16b9c:	4e440413          	addi	s0,s0,1252 # 2207c <__fini_array_end>
   16ba0:	40f40433          	sub	s0,s0,a5
   16ba4:	00912223          	sw	s1,4(sp)
   16ba8:	00112623          	sw	ra,12(sp)
   16bac:	40245493          	srai	s1,s0,0x2
   16bb0:	02048063          	beqz	s1,16bd0 <__libc_fini_array+0x48>
   16bb4:	ffc40413          	addi	s0,s0,-4
   16bb8:	00f40433          	add	s0,s0,a5
   16bbc:	00042783          	lw	a5,0(s0)
   16bc0:	fff48493          	addi	s1,s1,-1
   16bc4:	ffc40413          	addi	s0,s0,-4
   16bc8:	000780e7          	jalr	a5
   16bcc:	fe0498e3          	bnez	s1,16bbc <__libc_fini_array+0x34>
   16bd0:	00c12083          	lw	ra,12(sp)
   16bd4:	00812403          	lw	s0,8(sp)
   16bd8:	00412483          	lw	s1,4(sp)
   16bdc:	01010113          	addi	sp,sp,16
   16be0:	00008067          	ret

00016be4 <memmove>:
   16be4:	02a5f663          	bgeu	a1,a0,16c10 <memmove+0x2c>
   16be8:	00c58733          	add	a4,a1,a2
   16bec:	02e57263          	bgeu	a0,a4,16c10 <memmove+0x2c>
   16bf0:	00c507b3          	add	a5,a0,a2
   16bf4:	04060663          	beqz	a2,16c40 <memmove+0x5c>
   16bf8:	fff74683          	lbu	a3,-1(a4)
   16bfc:	fff78793          	addi	a5,a5,-1
   16c00:	fff70713          	addi	a4,a4,-1
   16c04:	00d78023          	sb	a3,0(a5)
   16c08:	fef518e3          	bne	a0,a5,16bf8 <memmove+0x14>
   16c0c:	00008067          	ret
   16c10:	00f00793          	li	a5,15
   16c14:	02c7e863          	bltu	a5,a2,16c44 <memmove+0x60>
   16c18:	00050793          	mv	a5,a0
   16c1c:	fff60693          	addi	a3,a2,-1
   16c20:	0c060263          	beqz	a2,16ce4 <memmove+0x100>
   16c24:	00168693          	addi	a3,a3,1 # 2001 <exit-0xe0b3>
   16c28:	00d786b3          	add	a3,a5,a3
   16c2c:	0005c703          	lbu	a4,0(a1)
   16c30:	00178793          	addi	a5,a5,1
   16c34:	00158593          	addi	a1,a1,1
   16c38:	fee78fa3          	sb	a4,-1(a5)
   16c3c:	fed798e3          	bne	a5,a3,16c2c <memmove+0x48>
   16c40:	00008067          	ret
   16c44:	00b567b3          	or	a5,a0,a1
   16c48:	0037f793          	andi	a5,a5,3
   16c4c:	08079663          	bnez	a5,16cd8 <memmove+0xf4>
   16c50:	ff060893          	addi	a7,a2,-16
   16c54:	ff08f893          	andi	a7,a7,-16
   16c58:	01088893          	addi	a7,a7,16
   16c5c:	011506b3          	add	a3,a0,a7
   16c60:	00058713          	mv	a4,a1
   16c64:	00050793          	mv	a5,a0
   16c68:	00072803          	lw	a6,0(a4)
   16c6c:	01070713          	addi	a4,a4,16
   16c70:	01078793          	addi	a5,a5,16
   16c74:	ff07a823          	sw	a6,-16(a5)
   16c78:	ff472803          	lw	a6,-12(a4)
   16c7c:	ff07aa23          	sw	a6,-12(a5)
   16c80:	ff872803          	lw	a6,-8(a4)
   16c84:	ff07ac23          	sw	a6,-8(a5)
   16c88:	ffc72803          	lw	a6,-4(a4)
   16c8c:	ff07ae23          	sw	a6,-4(a5)
   16c90:	fcd79ce3          	bne	a5,a3,16c68 <memmove+0x84>
   16c94:	00c67813          	andi	a6,a2,12
   16c98:	011585b3          	add	a1,a1,a7
   16c9c:	00f67713          	andi	a4,a2,15
   16ca0:	04080463          	beqz	a6,16ce8 <memmove+0x104>
   16ca4:	ffc70813          	addi	a6,a4,-4
   16ca8:	ffc87813          	andi	a6,a6,-4
   16cac:	00480813          	addi	a6,a6,4
   16cb0:	010687b3          	add	a5,a3,a6
   16cb4:	00058713          	mv	a4,a1
   16cb8:	00072883          	lw	a7,0(a4)
   16cbc:	00468693          	addi	a3,a3,4
   16cc0:	00470713          	addi	a4,a4,4
   16cc4:	ff16ae23          	sw	a7,-4(a3)
   16cc8:	fef698e3          	bne	a3,a5,16cb8 <memmove+0xd4>
   16ccc:	00367613          	andi	a2,a2,3
   16cd0:	010585b3          	add	a1,a1,a6
   16cd4:	f49ff06f          	j	16c1c <memmove+0x38>
   16cd8:	fff60693          	addi	a3,a2,-1
   16cdc:	00050793          	mv	a5,a0
   16ce0:	f45ff06f          	j	16c24 <memmove+0x40>
   16ce4:	00008067          	ret
   16ce8:	00070613          	mv	a2,a4
   16cec:	f31ff06f          	j	16c1c <memmove+0x38>

00016cf0 <memcpy>:
   16cf0:	00a5c7b3          	xor	a5,a1,a0
   16cf4:	0037f793          	andi	a5,a5,3
   16cf8:	00c508b3          	add	a7,a0,a2
   16cfc:	06079463          	bnez	a5,16d64 <memcpy+0x74>
   16d00:	00300793          	li	a5,3
   16d04:	06c7f063          	bgeu	a5,a2,16d64 <memcpy+0x74>
   16d08:	00357793          	andi	a5,a0,3
   16d0c:	00050713          	mv	a4,a0
   16d10:	06079a63          	bnez	a5,16d84 <memcpy+0x94>
   16d14:	ffc8f613          	andi	a2,a7,-4
   16d18:	40e606b3          	sub	a3,a2,a4
   16d1c:	02000793          	li	a5,32
   16d20:	08d7ce63          	blt	a5,a3,16dbc <memcpy+0xcc>
   16d24:	00058693          	mv	a3,a1
   16d28:	00070793          	mv	a5,a4
   16d2c:	02c77863          	bgeu	a4,a2,16d5c <memcpy+0x6c>
   16d30:	0006a803          	lw	a6,0(a3)
   16d34:	00478793          	addi	a5,a5,4
   16d38:	00468693          	addi	a3,a3,4
   16d3c:	ff07ae23          	sw	a6,-4(a5)
   16d40:	fec7e8e3          	bltu	a5,a2,16d30 <memcpy+0x40>
   16d44:	fff60793          	addi	a5,a2,-1
   16d48:	40e787b3          	sub	a5,a5,a4
   16d4c:	ffc7f793          	andi	a5,a5,-4
   16d50:	00478793          	addi	a5,a5,4
   16d54:	00f70733          	add	a4,a4,a5
   16d58:	00f585b3          	add	a1,a1,a5
   16d5c:	01176863          	bltu	a4,a7,16d6c <memcpy+0x7c>
   16d60:	00008067          	ret
   16d64:	00050713          	mv	a4,a0
   16d68:	05157863          	bgeu	a0,a7,16db8 <memcpy+0xc8>
   16d6c:	0005c783          	lbu	a5,0(a1)
   16d70:	00170713          	addi	a4,a4,1
   16d74:	00158593          	addi	a1,a1,1
   16d78:	fef70fa3          	sb	a5,-1(a4)
   16d7c:	fee898e3          	bne	a7,a4,16d6c <memcpy+0x7c>
   16d80:	00008067          	ret
   16d84:	0005c683          	lbu	a3,0(a1)
   16d88:	00170713          	addi	a4,a4,1
   16d8c:	00377793          	andi	a5,a4,3
   16d90:	fed70fa3          	sb	a3,-1(a4)
   16d94:	00158593          	addi	a1,a1,1
   16d98:	f6078ee3          	beqz	a5,16d14 <memcpy+0x24>
   16d9c:	0005c683          	lbu	a3,0(a1)
   16da0:	00170713          	addi	a4,a4,1
   16da4:	00377793          	andi	a5,a4,3
   16da8:	fed70fa3          	sb	a3,-1(a4)
   16dac:	00158593          	addi	a1,a1,1
   16db0:	fc079ae3          	bnez	a5,16d84 <memcpy+0x94>
   16db4:	f61ff06f          	j	16d14 <memcpy+0x24>
   16db8:	00008067          	ret
   16dbc:	ff010113          	addi	sp,sp,-16
   16dc0:	00812623          	sw	s0,12(sp)
   16dc4:	02000413          	li	s0,32
   16dc8:	0005a383          	lw	t2,0(a1)
   16dcc:	0045a283          	lw	t0,4(a1)
   16dd0:	0085af83          	lw	t6,8(a1)
   16dd4:	00c5af03          	lw	t5,12(a1)
   16dd8:	0105ae83          	lw	t4,16(a1)
   16ddc:	0145ae03          	lw	t3,20(a1)
   16de0:	0185a303          	lw	t1,24(a1)
   16de4:	01c5a803          	lw	a6,28(a1)
   16de8:	0205a683          	lw	a3,32(a1)
   16dec:	02470713          	addi	a4,a4,36
   16df0:	40e607b3          	sub	a5,a2,a4
   16df4:	fc772e23          	sw	t2,-36(a4)
   16df8:	fe572023          	sw	t0,-32(a4)
   16dfc:	fff72223          	sw	t6,-28(a4)
   16e00:	ffe72423          	sw	t5,-24(a4)
   16e04:	ffd72623          	sw	t4,-20(a4)
   16e08:	ffc72823          	sw	t3,-16(a4)
   16e0c:	fe672a23          	sw	t1,-12(a4)
   16e10:	ff072c23          	sw	a6,-8(a4)
   16e14:	fed72e23          	sw	a3,-4(a4)
   16e18:	02458593          	addi	a1,a1,36
   16e1c:	faf446e3          	blt	s0,a5,16dc8 <memcpy+0xd8>
   16e20:	00058693          	mv	a3,a1
   16e24:	00070793          	mv	a5,a4
   16e28:	02c77863          	bgeu	a4,a2,16e58 <memcpy+0x168>
   16e2c:	0006a803          	lw	a6,0(a3)
   16e30:	00478793          	addi	a5,a5,4
   16e34:	00468693          	addi	a3,a3,4
   16e38:	ff07ae23          	sw	a6,-4(a5)
   16e3c:	fec7e8e3          	bltu	a5,a2,16e2c <memcpy+0x13c>
   16e40:	fff60793          	addi	a5,a2,-1
   16e44:	40e787b3          	sub	a5,a5,a4
   16e48:	ffc7f793          	andi	a5,a5,-4
   16e4c:	00478793          	addi	a5,a5,4
   16e50:	00f70733          	add	a4,a4,a5
   16e54:	00f585b3          	add	a1,a1,a5
   16e58:	01176863          	bltu	a4,a7,16e68 <memcpy+0x178>
   16e5c:	00c12403          	lw	s0,12(sp)
   16e60:	01010113          	addi	sp,sp,16
   16e64:	00008067          	ret
   16e68:	0005c783          	lbu	a5,0(a1)
   16e6c:	00170713          	addi	a4,a4,1
   16e70:	00158593          	addi	a1,a1,1
   16e74:	fef70fa3          	sb	a5,-1(a4)
   16e78:	fee882e3          	beq	a7,a4,16e5c <memcpy+0x16c>
   16e7c:	0005c783          	lbu	a5,0(a1)
   16e80:	00170713          	addi	a4,a4,1
   16e84:	00158593          	addi	a1,a1,1
   16e88:	fef70fa3          	sb	a5,-1(a4)
   16e8c:	fce89ee3          	bne	a7,a4,16e68 <memcpy+0x178>
   16e90:	fcdff06f          	j	16e5c <memcpy+0x16c>

00016e94 <strcmp>:
   16e94:	00b56733          	or	a4,a0,a1
   16e98:	fff00393          	li	t2,-1
   16e9c:	00377713          	andi	a4,a4,3
   16ea0:	10071063          	bnez	a4,16fa0 <strcmp+0x10c>
   16ea4:	7f7f87b7          	lui	a5,0x7f7f8
   16ea8:	f7f78793          	addi	a5,a5,-129 # 7f7f7f7f <__BSS_END__+0x7f7d524f>
   16eac:	00052603          	lw	a2,0(a0)
   16eb0:	0005a683          	lw	a3,0(a1)
   16eb4:	00f672b3          	and	t0,a2,a5
   16eb8:	00f66333          	or	t1,a2,a5
   16ebc:	00f282b3          	add	t0,t0,a5
   16ec0:	0062e2b3          	or	t0,t0,t1
   16ec4:	10729263          	bne	t0,t2,16fc8 <strcmp+0x134>
   16ec8:	08d61663          	bne	a2,a3,16f54 <strcmp+0xc0>
   16ecc:	00452603          	lw	a2,4(a0)
   16ed0:	0045a683          	lw	a3,4(a1)
   16ed4:	00f672b3          	and	t0,a2,a5
   16ed8:	00f66333          	or	t1,a2,a5
   16edc:	00f282b3          	add	t0,t0,a5
   16ee0:	0062e2b3          	or	t0,t0,t1
   16ee4:	0c729e63          	bne	t0,t2,16fc0 <strcmp+0x12c>
   16ee8:	06d61663          	bne	a2,a3,16f54 <strcmp+0xc0>
   16eec:	00852603          	lw	a2,8(a0)
   16ef0:	0085a683          	lw	a3,8(a1)
   16ef4:	00f672b3          	and	t0,a2,a5
   16ef8:	00f66333          	or	t1,a2,a5
   16efc:	00f282b3          	add	t0,t0,a5
   16f00:	0062e2b3          	or	t0,t0,t1
   16f04:	0c729863          	bne	t0,t2,16fd4 <strcmp+0x140>
   16f08:	04d61663          	bne	a2,a3,16f54 <strcmp+0xc0>
   16f0c:	00c52603          	lw	a2,12(a0)
   16f10:	00c5a683          	lw	a3,12(a1)
   16f14:	00f672b3          	and	t0,a2,a5
   16f18:	00f66333          	or	t1,a2,a5
   16f1c:	00f282b3          	add	t0,t0,a5
   16f20:	0062e2b3          	or	t0,t0,t1
   16f24:	0c729263          	bne	t0,t2,16fe8 <strcmp+0x154>
   16f28:	02d61663          	bne	a2,a3,16f54 <strcmp+0xc0>
   16f2c:	01052603          	lw	a2,16(a0)
   16f30:	0105a683          	lw	a3,16(a1)
   16f34:	00f672b3          	and	t0,a2,a5
   16f38:	00f66333          	or	t1,a2,a5
   16f3c:	00f282b3          	add	t0,t0,a5
   16f40:	0062e2b3          	or	t0,t0,t1
   16f44:	0a729c63          	bne	t0,t2,16ffc <strcmp+0x168>
   16f48:	01450513          	addi	a0,a0,20
   16f4c:	01458593          	addi	a1,a1,20
   16f50:	f4d60ee3          	beq	a2,a3,16eac <strcmp+0x18>
   16f54:	01061713          	slli	a4,a2,0x10
   16f58:	01069793          	slli	a5,a3,0x10
   16f5c:	00f71e63          	bne	a4,a5,16f78 <strcmp+0xe4>
   16f60:	01065713          	srli	a4,a2,0x10
   16f64:	0106d793          	srli	a5,a3,0x10
   16f68:	40f70533          	sub	a0,a4,a5
   16f6c:	0ff57593          	zext.b	a1,a0
   16f70:	02059063          	bnez	a1,16f90 <strcmp+0xfc>
   16f74:	00008067          	ret
   16f78:	01075713          	srli	a4,a4,0x10
   16f7c:	0107d793          	srli	a5,a5,0x10
   16f80:	40f70533          	sub	a0,a4,a5
   16f84:	0ff57593          	zext.b	a1,a0
   16f88:	00059463          	bnez	a1,16f90 <strcmp+0xfc>
   16f8c:	00008067          	ret
   16f90:	0ff77713          	zext.b	a4,a4
   16f94:	0ff7f793          	zext.b	a5,a5
   16f98:	40f70533          	sub	a0,a4,a5
   16f9c:	00008067          	ret
   16fa0:	00054603          	lbu	a2,0(a0)
   16fa4:	0005c683          	lbu	a3,0(a1)
   16fa8:	00150513          	addi	a0,a0,1
   16fac:	00158593          	addi	a1,a1,1
   16fb0:	00d61463          	bne	a2,a3,16fb8 <strcmp+0x124>
   16fb4:	fe0616e3          	bnez	a2,16fa0 <strcmp+0x10c>
   16fb8:	40d60533          	sub	a0,a2,a3
   16fbc:	00008067          	ret
   16fc0:	00450513          	addi	a0,a0,4
   16fc4:	00458593          	addi	a1,a1,4
   16fc8:	fcd61ce3          	bne	a2,a3,16fa0 <strcmp+0x10c>
   16fcc:	00000513          	li	a0,0
   16fd0:	00008067          	ret
   16fd4:	00850513          	addi	a0,a0,8
   16fd8:	00858593          	addi	a1,a1,8
   16fdc:	fcd612e3          	bne	a2,a3,16fa0 <strcmp+0x10c>
   16fe0:	00000513          	li	a0,0
   16fe4:	00008067          	ret
   16fe8:	00c50513          	addi	a0,a0,12
   16fec:	00c58593          	addi	a1,a1,12
   16ff0:	fad618e3          	bne	a2,a3,16fa0 <strcmp+0x10c>
   16ff4:	00000513          	li	a0,0
   16ff8:	00008067          	ret
   16ffc:	01050513          	addi	a0,a0,16
   17000:	01058593          	addi	a1,a1,16
   17004:	f8d61ee3          	bne	a2,a3,16fa0 <strcmp+0x10c>
   17008:	00000513          	li	a0,0
   1700c:	00008067          	ret

00017010 <frexpl>:
   17010:	f9010113          	addi	sp,sp,-112
   17014:	07212023          	sw	s2,96(sp)
   17018:	00c5a903          	lw	s2,12(a1)
   1701c:	05412c23          	sw	s4,88(sp)
   17020:	05512a23          	sw	s5,84(sp)
   17024:	05612823          	sw	s6,80(sp)
   17028:	0045aa83          	lw	s5,4(a1)
   1702c:	0005ab03          	lw	s6,0(a1)
   17030:	0085aa03          	lw	s4,8(a1)
   17034:	05312e23          	sw	s3,92(sp)
   17038:	000089b7          	lui	s3,0x8
   1703c:	06812423          	sw	s0,104(sp)
   17040:	06912223          	sw	s1,100(sp)
   17044:	06112623          	sw	ra,108(sp)
   17048:	01095493          	srli	s1,s2,0x10
   1704c:	fff98993          	addi	s3,s3,-1 # 7fff <exit-0x80b5>
   17050:	03612823          	sw	s6,48(sp)
   17054:	03512a23          	sw	s5,52(sp)
   17058:	03412c23          	sw	s4,56(sp)
   1705c:	03212e23          	sw	s2,60(sp)
   17060:	0134f4b3          	and	s1,s1,s3
   17064:	00062023          	sw	zero,0(a2)
   17068:	00050413          	mv	s0,a0
   1706c:	09348063          	beq	s1,s3,170ec <frexpl+0xdc>
   17070:	01010593          	addi	a1,sp,16
   17074:	02010513          	addi	a0,sp,32
   17078:	05712623          	sw	s7,76(sp)
   1707c:	03612023          	sw	s6,32(sp)
   17080:	00060b93          	mv	s7,a2
   17084:	03512223          	sw	s5,36(sp)
   17088:	03412423          	sw	s4,40(sp)
   1708c:	03212623          	sw	s2,44(sp)
   17090:	00012823          	sw	zero,16(sp)
   17094:	00012a23          	sw	zero,20(sp)
   17098:	00012c23          	sw	zero,24(sp)
   1709c:	00012e23          	sw	zero,28(sp)
   170a0:	371060ef          	jal	1dc10 <__eqtf2>
   170a4:	0e050e63          	beqz	a0,171a0 <frexpl+0x190>
   170a8:	00000693          	li	a3,0
   170ac:	06048e63          	beqz	s1,17128 <frexpl+0x118>
   170b0:	ffffc737          	lui	a4,0xffffc
   170b4:	00270713          	addi	a4,a4,2 # ffffc002 <__BSS_END__+0xfffd92d2>
   170b8:	03c12903          	lw	s2,60(sp)
   170bc:	00e484b3          	add	s1,s1,a4
   170c0:	800107b7          	lui	a5,0x80010
   170c4:	00d484b3          	add	s1,s1,a3
   170c8:	fff78793          	addi	a5,a5,-1 # 8000ffff <__BSS_END__+0x7ffed2cf>
   170cc:	009ba023          	sw	s1,0(s7)
   170d0:	03012b03          	lw	s6,48(sp)
   170d4:	03412a83          	lw	s5,52(sp)
   170d8:	03812a03          	lw	s4,56(sp)
   170dc:	04c12b83          	lw	s7,76(sp)
   170e0:	00f97933          	and	s2,s2,a5
   170e4:	3ffe07b7          	lui	a5,0x3ffe0
   170e8:	00f96933          	or	s2,s2,a5
   170ec:	01642023          	sw	s6,0(s0)
   170f0:	01542223          	sw	s5,4(s0)
   170f4:	01442423          	sw	s4,8(s0)
   170f8:	01242623          	sw	s2,12(s0)
   170fc:	06c12083          	lw	ra,108(sp)
   17100:	00040513          	mv	a0,s0
   17104:	06812403          	lw	s0,104(sp)
   17108:	06412483          	lw	s1,100(sp)
   1710c:	06012903          	lw	s2,96(sp)
   17110:	05c12983          	lw	s3,92(sp)
   17114:	05812a03          	lw	s4,88(sp)
   17118:	05412a83          	lw	s5,84(sp)
   1711c:	05012b03          	lw	s6,80(sp)
   17120:	07010113          	addi	sp,sp,112
   17124:	00008067          	ret
   17128:	0000a797          	auipc	a5,0xa
   1712c:	db878793          	addi	a5,a5,-584 # 20ee0 <blanks.1+0x44>
   17130:	0007a603          	lw	a2,0(a5)
   17134:	0047a683          	lw	a3,4(a5)
   17138:	0087a703          	lw	a4,8(a5)
   1713c:	00c7a783          	lw	a5,12(a5)
   17140:	00c12023          	sw	a2,0(sp)
   17144:	01010593          	addi	a1,sp,16
   17148:	00010613          	mv	a2,sp
   1714c:	02010513          	addi	a0,sp,32
   17150:	00d12223          	sw	a3,4(sp)
   17154:	00e12423          	sw	a4,8(sp)
   17158:	00f12623          	sw	a5,12(sp)
   1715c:	01612823          	sw	s6,16(sp)
   17160:	01512a23          	sw	s5,20(sp)
   17164:	01412c23          	sw	s4,24(sp)
   17168:	01212e23          	sw	s2,28(sp)
   1716c:	5d1060ef          	jal	1df3c <__multf3>
   17170:	02012703          	lw	a4,32(sp)
   17174:	02c12783          	lw	a5,44(sp)
   17178:	f8e00693          	li	a3,-114
   1717c:	02e12823          	sw	a4,48(sp)
   17180:	02412703          	lw	a4,36(sp)
   17184:	0107d493          	srli	s1,a5,0x10
   17188:	02f12e23          	sw	a5,60(sp)
   1718c:	02e12a23          	sw	a4,52(sp)
   17190:	02812703          	lw	a4,40(sp)
   17194:	0134f4b3          	and	s1,s1,s3
   17198:	02e12c23          	sw	a4,56(sp)
   1719c:	f15ff06f          	j	170b0 <frexpl+0xa0>
   171a0:	04c12b83          	lw	s7,76(sp)
   171a4:	f49ff06f          	j	170ec <frexpl+0xdc>

000171a8 <__register_exitproc>:
   171a8:	0a018713          	addi	a4,gp,160 # 22920 <__atexit>
   171ac:	00072783          	lw	a5,0(a4)
   171b0:	04078c63          	beqz	a5,17208 <__register_exitproc+0x60>
   171b4:	0047a703          	lw	a4,4(a5)
   171b8:	01f00813          	li	a6,31
   171bc:	06e84e63          	blt	a6,a4,17238 <__register_exitproc+0x90>
   171c0:	00271813          	slli	a6,a4,0x2
   171c4:	02050663          	beqz	a0,171f0 <__register_exitproc+0x48>
   171c8:	01078333          	add	t1,a5,a6
   171cc:	08c32423          	sw	a2,136(t1)
   171d0:	1887a883          	lw	a7,392(a5)
   171d4:	00100613          	li	a2,1
   171d8:	00e61633          	sll	a2,a2,a4
   171dc:	00c8e8b3          	or	a7,a7,a2
   171e0:	1917a423          	sw	a7,392(a5)
   171e4:	10d32423          	sw	a3,264(t1)
   171e8:	00200693          	li	a3,2
   171ec:	02d50463          	beq	a0,a3,17214 <__register_exitproc+0x6c>
   171f0:	00170713          	addi	a4,a4,1
   171f4:	00e7a223          	sw	a4,4(a5)
   171f8:	010787b3          	add	a5,a5,a6
   171fc:	00b7a423          	sw	a1,8(a5)
   17200:	00000513          	li	a0,0
   17204:	00008067          	ret
   17208:	32018793          	addi	a5,gp,800 # 22ba0 <__atexit0>
   1720c:	00f72023          	sw	a5,0(a4)
   17210:	fa5ff06f          	j	171b4 <__register_exitproc+0xc>
   17214:	18c7a683          	lw	a3,396(a5)
   17218:	00170713          	addi	a4,a4,1
   1721c:	00e7a223          	sw	a4,4(a5)
   17220:	00c6e6b3          	or	a3,a3,a2
   17224:	18d7a623          	sw	a3,396(a5)
   17228:	010787b3          	add	a5,a5,a6
   1722c:	00b7a423          	sw	a1,8(a5)
   17230:	00000513          	li	a0,0
   17234:	00008067          	ret
   17238:	fff00513          	li	a0,-1
   1723c:	00008067          	ret

00017240 <_ldtoa_r>:
   17240:	0000a897          	auipc	a7,0xa
   17244:	e5c88893          	addi	a7,a7,-420 # 2109c <blanks.1+0x10>
   17248:	0008af83          	lw	t6,0(a7)
   1724c:	0048af03          	lw	t5,4(a7)
   17250:	0088ae83          	lw	t4,8(a7)
   17254:	00c8ae03          	lw	t3,12(a7)
   17258:	0108a303          	lw	t1,16(a7)
   1725c:	03852883          	lw	a7,56(a0)
   17260:	f4010113          	addi	sp,sp,-192
   17264:	0b212823          	sw	s2,176(sp)
   17268:	0b312623          	sw	s3,172(sp)
   1726c:	0b412423          	sw	s4,168(sp)
   17270:	0b612023          	sw	s6,160(sp)
   17274:	09712e23          	sw	s7,156(sp)
   17278:	09812c23          	sw	s8,152(sp)
   1727c:	09912a23          	sw	s9,148(sp)
   17280:	09a12823          	sw	s10,144(sp)
   17284:	0a112e23          	sw	ra,188(sp)
   17288:	0a812c23          	sw	s0,184(sp)
   1728c:	0a912a23          	sw	s1,180(sp)
   17290:	0b512223          	sw	s5,164(sp)
   17294:	09b12623          	sw	s11,140(sp)
   17298:	07f12623          	sw	t6,108(sp)
   1729c:	07e12823          	sw	t5,112(sp)
   172a0:	07d12a23          	sw	t4,116(sp)
   172a4:	07c12c23          	sw	t3,120(sp)
   172a8:	06612e23          	sw	t1,124(sp)
   172ac:	02c12023          	sw	a2,32(sp)
   172b0:	02d12223          	sw	a3,36(sp)
   172b4:	0005aa03          	lw	s4,0(a1)
   172b8:	0045a983          	lw	s3,4(a1)
   172bc:	0085a903          	lw	s2,8(a1)
   172c0:	00c5ac03          	lw	s8,12(a1)
   172c4:	00050b13          	mv	s6,a0
   172c8:	00070b93          	mv	s7,a4
   172cc:	00078c93          	mv	s9,a5
   172d0:	00080d13          	mv	s10,a6
   172d4:	02088263          	beqz	a7,172f8 <_ldtoa_r+0xb8>
   172d8:	03c52683          	lw	a3,60(a0)
   172dc:	00100713          	li	a4,1
   172e0:	00088593          	mv	a1,a7
   172e4:	00d71733          	sll	a4,a4,a3
   172e8:	00d8a223          	sw	a3,4(a7)
   172ec:	00e8a423          	sw	a4,8(a7)
   172f0:	54c020ef          	jal	1983c <_Bfree>
   172f4:	020b2c23          	sw	zero,56(s6)
   172f8:	07812603          	lw	a2,120(sp)
   172fc:	01fc5693          	srli	a3,s8,0x1f
   17300:	001c1a93          	slli	s5,s8,0x1
   17304:	40165713          	srai	a4,a2,0x1
   17308:	ffffc4b7          	lui	s1,0xffffc
   1730c:	001c1413          	slli	s0,s8,0x1
   17310:	00d77733          	and	a4,a4,a3
   17314:	011ada93          	srli	s5,s5,0x11
   17318:	f9148493          	addi	s1,s1,-111 # ffffbf91 <__BSS_END__+0xfffd9261>
   1731c:	010c1d93          	slli	s11,s8,0x10
   17320:	00dca023          	sw	a3,0(s9)
   17324:	00145413          	srli	s0,s0,0x1
   17328:	00c74733          	xor	a4,a4,a2
   1732c:	010ddd93          	srli	s11,s11,0x10
   17330:	009a87b3          	add	a5,s5,s1
   17334:	03010593          	addi	a1,sp,48
   17338:	04010513          	addi	a0,sp,64
   1733c:	05412023          	sw	s4,64(sp)
   17340:	05312223          	sw	s3,68(sp)
   17344:	05212423          	sw	s2,72(sp)
   17348:	04812623          	sw	s0,76(sp)
   1734c:	03412823          	sw	s4,48(sp)
   17350:	03312a23          	sw	s3,52(sp)
   17354:	03212c23          	sw	s2,56(sp)
   17358:	02812e23          	sw	s0,60(sp)
   1735c:	06e12c23          	sw	a4,120(sp)
   17360:	00f12e23          	sw	a5,28(sp)
   17364:	05412e23          	sw	s4,92(sp)
   17368:	07312023          	sw	s3,96(sp)
   1736c:	07212223          	sw	s2,100(sp)
   17370:	07b12423          	sw	s11,104(sp)
   17374:	05c090ef          	jal	203d0 <__unordtf2>
   17378:	18051e63          	bnez	a0,17514 <_ldtoa_r+0x2d4>
   1737c:	0000a797          	auipc	a5,0xa
   17380:	b7478793          	addi	a5,a5,-1164 # 20ef0 <blanks.1+0x54>
   17384:	0007a603          	lw	a2,0(a5)
   17388:	0047a683          	lw	a3,4(a5)
   1738c:	0087a483          	lw	s1,8(a5)
   17390:	00c7ac83          	lw	s9,12(a5)
   17394:	03010593          	addi	a1,sp,48
   17398:	04010513          	addi	a0,sp,64
   1739c:	05412023          	sw	s4,64(sp)
   173a0:	05312223          	sw	s3,68(sp)
   173a4:	05212423          	sw	s2,72(sp)
   173a8:	04812623          	sw	s0,76(sp)
   173ac:	02c12823          	sw	a2,48(sp)
   173b0:	02c12623          	sw	a2,44(sp)
   173b4:	02d12a23          	sw	a3,52(sp)
   173b8:	02d12423          	sw	a3,40(sp)
   173bc:	02912c23          	sw	s1,56(sp)
   173c0:	03912e23          	sw	s9,60(sp)
   173c4:	00c090ef          	jal	203d0 <__unordtf2>
   173c8:	08051c63          	bnez	a0,17460 <_ldtoa_r+0x220>
   173cc:	03010593          	addi	a1,sp,48
   173d0:	04010513          	addi	a0,sp,64
   173d4:	239060ef          	jal	1de0c <__letf2>
   173d8:	08a05463          	blez	a0,17460 <_ldtoa_r+0x220>
   173dc:	00300793          	li	a5,3
   173e0:	04f12c23          	sw	a5,88(sp)
   173e4:	02012783          	lw	a5,32(sp)
   173e8:	02412803          	lw	a6,36(sp)
   173ec:	01c12603          	lw	a2,28(sp)
   173f0:	05810713          	addi	a4,sp,88
   173f4:	01a12023          	sw	s10,0(sp)
   173f8:	000b8893          	mv	a7,s7
   173fc:	05c10693          	addi	a3,sp,92
   17400:	06c10593          	addi	a1,sp,108
   17404:	000b0513          	mv	a0,s6
   17408:	238000ef          	jal	17640 <__gdtoa>
   1740c:	000ba703          	lw	a4,0(s7)
   17410:	ffff87b7          	lui	a5,0xffff8
   17414:	00f71863          	bne	a4,a5,17424 <_ldtoa_r+0x1e4>
   17418:	800007b7          	lui	a5,0x80000
   1741c:	fff78793          	addi	a5,a5,-1 # 7fffffff <__BSS_END__+0x7ffdd2cf>
   17420:	00fba023          	sw	a5,0(s7)
   17424:	0bc12083          	lw	ra,188(sp)
   17428:	0b812403          	lw	s0,184(sp)
   1742c:	0b412483          	lw	s1,180(sp)
   17430:	0b012903          	lw	s2,176(sp)
   17434:	0ac12983          	lw	s3,172(sp)
   17438:	0a812a03          	lw	s4,168(sp)
   1743c:	0a412a83          	lw	s5,164(sp)
   17440:	0a012b03          	lw	s6,160(sp)
   17444:	09c12b83          	lw	s7,156(sp)
   17448:	09812c03          	lw	s8,152(sp)
   1744c:	09412c83          	lw	s9,148(sp)
   17450:	09012d03          	lw	s10,144(sp)
   17454:	08c12d83          	lw	s11,140(sp)
   17458:	0c010113          	addi	sp,sp,192
   1745c:	00008067          	ret
   17460:	0000a797          	auipc	a5,0xa
   17464:	aa078793          	addi	a5,a5,-1376 # 20f00 <blanks.1+0x64>
   17468:	0007a603          	lw	a2,0(a5)
   1746c:	0047a683          	lw	a3,4(a5)
   17470:	0087a703          	lw	a4,8(a5)
   17474:	00c7a783          	lw	a5,12(a5)
   17478:	03010593          	addi	a1,sp,48
   1747c:	04010513          	addi	a0,sp,64
   17480:	05412023          	sw	s4,64(sp)
   17484:	05312223          	sw	s3,68(sp)
   17488:	05212423          	sw	s2,72(sp)
   1748c:	04812623          	sw	s0,76(sp)
   17490:	02c12823          	sw	a2,48(sp)
   17494:	02d12a23          	sw	a3,52(sp)
   17498:	02e12c23          	sw	a4,56(sp)
   1749c:	02f12e23          	sw	a5,60(sp)
   174a0:	03d060ef          	jal	1dcdc <__getf2>
   174a4:	00054e63          	bltz	a0,174c0 <_ldtoa_r+0x280>
   174a8:	000107b7          	lui	a5,0x10
   174ac:	00fdedb3          	or	s11,s11,a5
   174b0:	00100793          	li	a5,1
   174b4:	04f12c23          	sw	a5,88(sp)
   174b8:	07b12423          	sw	s11,104(sp)
   174bc:	f29ff06f          	j	173e4 <_ldtoa_r+0x1a4>
   174c0:	03010593          	addi	a1,sp,48
   174c4:	04010513          	addi	a0,sp,64
   174c8:	05412023          	sw	s4,64(sp)
   174cc:	05312223          	sw	s3,68(sp)
   174d0:	05212423          	sw	s2,72(sp)
   174d4:	05812623          	sw	s8,76(sp)
   174d8:	02012823          	sw	zero,48(sp)
   174dc:	02012a23          	sw	zero,52(sp)
   174e0:	02012c23          	sw	zero,56(sp)
   174e4:	02012e23          	sw	zero,60(sp)
   174e8:	728060ef          	jal	1dc10 <__eqtf2>
   174ec:	00051663          	bnez	a0,174f8 <_ldtoa_r+0x2b8>
   174f0:	04012c23          	sw	zero,88(sp)
   174f4:	ef1ff06f          	j	173e4 <_ldtoa_r+0x1a4>
   174f8:	ffffc4b7          	lui	s1,0xffffc
   174fc:	00200793          	li	a5,2
   17500:	f9248493          	addi	s1,s1,-110 # ffffbf92 <__BSS_END__+0xfffd9262>
   17504:	04f12c23          	sw	a5,88(sp)
   17508:	009a87b3          	add	a5,s5,s1
   1750c:	00f12e23          	sw	a5,28(sp)
   17510:	ed5ff06f          	j	173e4 <_ldtoa_r+0x1a4>
   17514:	00400793          	li	a5,4
   17518:	04f12c23          	sw	a5,88(sp)
   1751c:	ec9ff06f          	j	173e4 <_ldtoa_r+0x1a4>

00017520 <_ldcheck>:
   17520:	fb010113          	addi	sp,sp,-80
   17524:	04912223          	sw	s1,68(sp)
   17528:	00c52483          	lw	s1,12(a0)
   1752c:	05212023          	sw	s2,64(sp)
   17530:	03312e23          	sw	s3,60(sp)
   17534:	00852903          	lw	s2,8(a0)
   17538:	00452983          	lw	s3,4(a0)
   1753c:	03412c23          	sw	s4,56(sp)
   17540:	00052a03          	lw	s4,0(a0)
   17544:	00149493          	slli	s1,s1,0x1
   17548:	0014d493          	srli	s1,s1,0x1
   1754c:	00010593          	mv	a1,sp
   17550:	01010513          	addi	a0,sp,16
   17554:	04112623          	sw	ra,76(sp)
   17558:	01412823          	sw	s4,16(sp)
   1755c:	01312a23          	sw	s3,20(sp)
   17560:	01212c23          	sw	s2,24(sp)
   17564:	00912e23          	sw	s1,28(sp)
   17568:	01412023          	sw	s4,0(sp)
   1756c:	01312223          	sw	s3,4(sp)
   17570:	01212423          	sw	s2,8(sp)
   17574:	00912623          	sw	s1,12(sp)
   17578:	659080ef          	jal	203d0 <__unordtf2>
   1757c:	0a051263          	bnez	a0,17620 <_ldcheck+0x100>
   17580:	0000a797          	auipc	a5,0xa
   17584:	97078793          	addi	a5,a5,-1680 # 20ef0 <blanks.1+0x54>
   17588:	03512a23          	sw	s5,52(sp)
   1758c:	03612823          	sw	s6,48(sp)
   17590:	03712623          	sw	s7,44(sp)
   17594:	03812423          	sw	s8,40(sp)
   17598:	0047ab83          	lw	s7,4(a5)
   1759c:	0007ac03          	lw	s8,0(a5)
   175a0:	0087ab03          	lw	s6,8(a5)
   175a4:	00c7aa83          	lw	s5,12(a5)
   175a8:	00010593          	mv	a1,sp
   175ac:	01010513          	addi	a0,sp,16
   175b0:	04812423          	sw	s0,72(sp)
   175b4:	01812023          	sw	s8,0(sp)
   175b8:	01712223          	sw	s7,4(sp)
   175bc:	01612423          	sw	s6,8(sp)
   175c0:	01512623          	sw	s5,12(sp)
   175c4:	00100413          	li	s0,1
   175c8:	609080ef          	jal	203d0 <__unordtf2>
   175cc:	04050063          	beqz	a0,1760c <_ldcheck+0xec>
   175d0:	04c12083          	lw	ra,76(sp)
   175d4:	00144513          	xori	a0,s0,1
   175d8:	04812403          	lw	s0,72(sp)
   175dc:	0ff57513          	zext.b	a0,a0
   175e0:	03412a83          	lw	s5,52(sp)
   175e4:	03012b03          	lw	s6,48(sp)
   175e8:	02c12b83          	lw	s7,44(sp)
   175ec:	02812c03          	lw	s8,40(sp)
   175f0:	04412483          	lw	s1,68(sp)
   175f4:	04012903          	lw	s2,64(sp)
   175f8:	03c12983          	lw	s3,60(sp)
   175fc:	03812a03          	lw	s4,56(sp)
   17600:	00151513          	slli	a0,a0,0x1
   17604:	05010113          	addi	sp,sp,80
   17608:	00008067          	ret
   1760c:	00010593          	mv	a1,sp
   17610:	01010513          	addi	a0,sp,16
   17614:	7f8060ef          	jal	1de0c <__letf2>
   17618:	00152413          	slti	s0,a0,1
   1761c:	fb5ff06f          	j	175d0 <_ldcheck+0xb0>
   17620:	04c12083          	lw	ra,76(sp)
   17624:	04412483          	lw	s1,68(sp)
   17628:	04012903          	lw	s2,64(sp)
   1762c:	03c12983          	lw	s3,60(sp)
   17630:	03812a03          	lw	s4,56(sp)
   17634:	00100513          	li	a0,1
   17638:	05010113          	addi	sp,sp,80
   1763c:	00008067          	ret

00017640 <__gdtoa>:
   17640:	f4010113          	addi	sp,sp,-192
   17644:	0b612023          	sw	s6,160(sp)
   17648:	00072b03          	lw	s6,0(a4)
   1764c:	0b212823          	sw	s2,176(sp)
   17650:	0a112e23          	sw	ra,188(sp)
   17654:	fcfb7313          	andi	t1,s6,-49
   17658:	00672023          	sw	t1,0(a4)
   1765c:	00070913          	mv	s2,a4
   17660:	0c012703          	lw	a4,192(sp)
   17664:	00fb7313          	andi	t1,s6,15
   17668:	00300e13          	li	t3,3
   1766c:	00e12423          	sw	a4,8(sp)
   17670:	00a12023          	sw	a0,0(sp)
   17674:	00c12823          	sw	a2,16(sp)
   17678:	01012a23          	sw	a6,20(sp)
   1767c:	01112223          	sw	a7,4(sp)
   17680:	7fc30663          	beq	t1,t3,17e6c <__gdtoa+0x82c>
   17684:	09812c23          	sw	s8,152(sp)
   17688:	00cb7c13          	andi	s8,s6,12
   1768c:	720c1e63          	bnez	s8,17dc8 <__gdtoa+0x788>
   17690:	78030c63          	beqz	t1,17e28 <__gdtoa+0x7e8>
   17694:	0b512223          	sw	s5,164(sp)
   17698:	0005aa83          	lw	s5,0(a1)
   1769c:	0b312623          	sw	s3,172(sp)
   176a0:	09a12823          	sw	s10,144(sp)
   176a4:	09b12623          	sw	s11,140(sp)
   176a8:	00068993          	mv	s3,a3
   176ac:	02000693          	li	a3,32
   176b0:	00058d93          	mv	s11,a1
   176b4:	00078d13          	mv	s10,a5
   176b8:	00000593          	li	a1,0
   176bc:	02000793          	li	a5,32
   176c0:	0156d863          	bge	a3,s5,176d0 <__gdtoa+0x90>
   176c4:	00179793          	slli	a5,a5,0x1
   176c8:	00158593          	addi	a1,a1,1
   176cc:	ff57cce3          	blt	a5,s5,176c4 <__gdtoa+0x84>
   176d0:	00012503          	lw	a0,0(sp)
   176d4:	0b4020ef          	jal	19788 <_Balloc>
   176d8:	00a12623          	sw	a0,12(sp)
   176dc:	3c0508e3          	beqz	a0,182ac <__gdtoa+0xc6c>
   176e0:	00c12783          	lw	a5,12(sp)
   176e4:	fffa8513          	addi	a0,s5,-1
   176e8:	40555513          	srai	a0,a0,0x5
   176ec:	00251593          	slli	a1,a0,0x2
   176f0:	01478693          	addi	a3,a5,20
   176f4:	0b412423          	sw	s4,168(sp)
   176f8:	09912a23          	sw	s9,148(sp)
   176fc:	00b985b3          	add	a1,s3,a1
   17700:	00098793          	mv	a5,s3
   17704:	0007a703          	lw	a4,0(a5)
   17708:	00478793          	addi	a5,a5,4
   1770c:	00468693          	addi	a3,a3,4
   17710:	fee6ae23          	sw	a4,-4(a3)
   17714:	fef5f8e3          	bgeu	a1,a5,17704 <__gdtoa+0xc4>
   17718:	00158593          	addi	a1,a1,1
   1771c:	00198793          	addi	a5,s3,1
   17720:	00400693          	li	a3,4
   17724:	00f5e663          	bltu	a1,a5,17730 <__gdtoa+0xf0>
   17728:	00251513          	slli	a0,a0,0x2
   1772c:	00450693          	addi	a3,a0,4
   17730:	00c12783          	lw	a5,12(sp)
   17734:	4026dc93          	srai	s9,a3,0x2
   17738:	00d786b3          	add	a3,a5,a3
   1773c:	00c0006f          	j	17748 <__gdtoa+0x108>
   17740:	ffc68693          	addi	a3,a3,-4
   17744:	7a0c8063          	beqz	s9,17ee4 <__gdtoa+0x8a4>
   17748:	0106a783          	lw	a5,16(a3)
   1774c:	000c8a13          	mv	s4,s9
   17750:	fffc8c93          	addi	s9,s9,-1
   17754:	fe0786e3          	beqz	a5,17740 <__gdtoa+0x100>
   17758:	00c12703          	lw	a4,12(sp)
   1775c:	004c8793          	addi	a5,s9,4
   17760:	00279793          	slli	a5,a5,0x2
   17764:	00f707b3          	add	a5,a4,a5
   17768:	0047a503          	lw	a0,4(a5)
   1776c:	01472823          	sw	s4,16(a4)
   17770:	005a1a13          	slli	s4,s4,0x5
   17774:	370020ef          	jal	19ae4 <__hi0bits>
   17778:	40aa0cb3          	sub	s9,s4,a0
   1777c:	00c12503          	lw	a0,12(sp)
   17780:	735010ef          	jal	196b4 <__trailz_D2A>
   17784:	01012783          	lw	a5,16(sp)
   17788:	06a12e23          	sw	a0,124(sp)
   1778c:	02f12023          	sw	a5,32(sp)
   17790:	78051e63          	bnez	a0,17f2c <__gdtoa+0x8ec>
   17794:	00c12783          	lw	a5,12(sp)
   17798:	0107a683          	lw	a3,16(a5)
   1779c:	66068463          	beqz	a3,17e04 <__gdtoa+0x7c4>
   177a0:	00c12503          	lw	a0,12(sp)
   177a4:	07c10593          	addi	a1,sp,124
   177a8:	0a812c23          	sw	s0,184(sp)
   177ac:	0a912a23          	sw	s1,180(sp)
   177b0:	09712e23          	sw	s7,156(sp)
   177b4:	56d020ef          	jal	1a520 <__b2d>
   177b8:	00c59a13          	slli	s4,a1,0xc
   177bc:	00ca5a13          	srli	s4,s4,0xc
   177c0:	3ff006b7          	lui	a3,0x3ff00
   177c4:	00da64b3          	or	s1,s4,a3
   177c8:	02012703          	lw	a4,32(sp)
   177cc:	0000b697          	auipc	a3,0xb
   177d0:	0dc68693          	addi	a3,a3,220 # 228a8 <__SDATA_BEGIN__>
   177d4:	0006a603          	lw	a2,0(a3)
   177d8:	0046a683          	lw	a3,4(a3)
   177dc:	01970733          	add	a4,a4,s9
   177e0:	00050793          	mv	a5,a0
   177e4:	00048593          	mv	a1,s1
   177e8:	fff70a13          	addi	s4,a4,-1
   177ec:	04f12423          	sw	a5,72(sp)
   177f0:	00e12e23          	sw	a4,28(sp)
   177f4:	02a12c23          	sw	a0,56(sp)
   177f8:	349050ef          	jal	1d340 <__subdf3>
   177fc:	0000b697          	auipc	a3,0xb
   17800:	0b468693          	addi	a3,a3,180 # 228b0 <__SDATA_BEGIN__+0x8>
   17804:	0006a603          	lw	a2,0(a3)
   17808:	0046a683          	lw	a3,4(a3)
   1780c:	544050ef          	jal	1cd50 <__muldf3>
   17810:	0000b697          	auipc	a3,0xb
   17814:	0a868693          	addi	a3,a3,168 # 228b8 <__SDATA_BEGIN__+0x10>
   17818:	0006a603          	lw	a2,0(a3)
   1781c:	0046a683          	lw	a3,4(a3)
   17820:	478040ef          	jal	1bc98 <__adddf3>
   17824:	00050b93          	mv	s7,a0
   17828:	000a0513          	mv	a0,s4
   1782c:	00058413          	mv	s0,a1
   17830:	338060ef          	jal	1db68 <__floatsidf>
   17834:	0000b697          	auipc	a3,0xb
   17838:	08c68693          	addi	a3,a3,140 # 228c0 <__SDATA_BEGIN__+0x18>
   1783c:	0006a603          	lw	a2,0(a3)
   17840:	0046a683          	lw	a3,4(a3)
   17844:	50c050ef          	jal	1cd50 <__muldf3>
   17848:	00050613          	mv	a2,a0
   1784c:	00058693          	mv	a3,a1
   17850:	000b8513          	mv	a0,s7
   17854:	00040593          	mv	a1,s0
   17858:	440040ef          	jal	1bc98 <__adddf3>
   1785c:	00050b93          	mv	s7,a0
   17860:	00058413          	mv	s0,a1
   17864:	000a0513          	mv	a0,s4
   17868:	000a5863          	bgez	s4,17878 <__gdtoa+0x238>
   1786c:	01c12703          	lw	a4,28(sp)
   17870:	00100513          	li	a0,1
   17874:	40e50533          	sub	a0,a0,a4
   17878:	bcb50513          	addi	a0,a0,-1077
   1787c:	02a05c63          	blez	a0,178b4 <__gdtoa+0x274>
   17880:	2e8060ef          	jal	1db68 <__floatsidf>
   17884:	0000b697          	auipc	a3,0xb
   17888:	04468693          	addi	a3,a3,68 # 228c8 <__SDATA_BEGIN__+0x20>
   1788c:	0006a603          	lw	a2,0(a3)
   17890:	0046a683          	lw	a3,4(a3)
   17894:	4bc050ef          	jal	1cd50 <__muldf3>
   17898:	00050613          	mv	a2,a0
   1789c:	00058693          	mv	a3,a1
   178a0:	000b8513          	mv	a0,s7
   178a4:	00040593          	mv	a1,s0
   178a8:	3f0040ef          	jal	1bc98 <__adddf3>
   178ac:	00050b93          	mv	s7,a0
   178b0:	00058413          	mv	s0,a1
   178b4:	00040593          	mv	a1,s0
   178b8:	000b8513          	mv	a0,s7
   178bc:	22c060ef          	jal	1dae8 <__fixdfsi>
   178c0:	00050813          	mv	a6,a0
   178c4:	00040593          	mv	a1,s0
   178c8:	000b8513          	mv	a0,s7
   178cc:	00000613          	li	a2,0
   178d0:	00000693          	li	a3,0
   178d4:	01012c23          	sw	a6,24(sp)
   178d8:	39c050ef          	jal	1cc74 <__ledf2>
   178dc:	02055463          	bgez	a0,17904 <__gdtoa+0x2c4>
   178e0:	01812503          	lw	a0,24(sp)
   178e4:	284060ef          	jal	1db68 <__floatsidf>
   178e8:	000b8613          	mv	a2,s7
   178ec:	00040693          	mv	a3,s0
   178f0:	21c050ef          	jal	1cb0c <__eqdf2>
   178f4:	01812783          	lw	a5,24(sp)
   178f8:	00a03533          	snez	a0,a0
   178fc:	40a787b3          	sub	a5,a5,a0
   17900:	00f12c23          	sw	a5,24(sp)
   17904:	014a1613          	slli	a2,s4,0x14
   17908:	01812783          	lw	a5,24(sp)
   1790c:	00960833          	add	a6,a2,s1
   17910:	414c8433          	sub	s0,s9,s4
   17914:	01600693          	li	a3,22
   17918:	05012023          	sw	a6,64(sp)
   1791c:	fff40b93          	addi	s7,s0,-1
   17920:	5cf6e863          	bltu	a3,a5,17ef0 <__gdtoa+0x8b0>
   17924:	00009317          	auipc	t1,0x9
   17928:	7ec30313          	addi	t1,t1,2028 # 21110 <__mprec_tens>
   1792c:	00379693          	slli	a3,a5,0x3
   17930:	00d306b3          	add	a3,t1,a3
   17934:	03812883          	lw	a7,56(sp)
   17938:	0006a503          	lw	a0,0(a3)
   1793c:	0046a583          	lw	a1,4(a3)
   17940:	00088613          	mv	a2,a7
   17944:	00080693          	mv	a3,a6
   17948:	250050ef          	jal	1cb98 <__gedf2>
   1794c:	1aa044e3          	bgtz	a0,182f4 <__gdtoa+0xcb4>
   17950:	02012e23          	sw	zero,60(sp)
   17954:	02012223          	sw	zero,36(sp)
   17958:	00804a63          	bgtz	s0,1796c <__gdtoa+0x32c>
   1795c:	00100693          	li	a3,1
   17960:	408687b3          	sub	a5,a3,s0
   17964:	02f12223          	sw	a5,36(sp)
   17968:	00000b93          	li	s7,0
   1796c:	01812783          	lw	a5,24(sp)
   17970:	02012623          	sw	zero,44(sp)
   17974:	00fb8bb3          	add	s7,s7,a5
   17978:	02f12823          	sw	a5,48(sp)
   1797c:	00900693          	li	a3,9
   17980:	5da6e863          	bltu	a3,s10,17f50 <__gdtoa+0x910>
   17984:	00500693          	li	a3,5
   17988:	19a6c0e3          	blt	a3,s10,18308 <__gdtoa+0xcc8>
   1798c:	01c12783          	lw	a5,28(sp)
   17990:	00400613          	li	a2,4
   17994:	3fd78a13          	addi	s4,a5,1021
   17998:	7f8a3a13          	sltiu	s4,s4,2040
   1799c:	00cd1463          	bne	s10,a2,179a4 <__gdtoa+0x364>
   179a0:	0a00106f          	j	18a40 <__gdtoa+0x1400>
   179a4:	00dd1463          	bne	s10,a3,179ac <__gdtoa+0x36c>
   179a8:	08c0106f          	j	18a34 <__gdtoa+0x13f4>
   179ac:	00200693          	li	a3,2
   179b0:	00dd1463          	bne	s10,a3,179b8 <__gdtoa+0x378>
   179b4:	0980106f          	j	18a4c <__gdtoa+0x140c>
   179b8:	00300693          	li	a3,3
   179bc:	02012a23          	sw	zero,52(sp)
   179c0:	5add1063          	bne	s10,a3,17f60 <__gdtoa+0x920>
   179c4:	03012783          	lw	a5,48(sp)
   179c8:	01412703          	lw	a4,20(sp)
   179cc:	00e787b3          	add	a5,a5,a4
   179d0:	04f12623          	sw	a5,76(sp)
   179d4:	00178793          	addi	a5,a5,1
   179d8:	00f12e23          	sw	a5,28(sp)
   179dc:	62f058e3          	blez	a5,1880c <__gdtoa+0x11cc>
   179e0:	00078693          	mv	a3,a5
   179e4:	00078593          	mv	a1,a5
   179e8:	00012503          	lw	a0,0(sp)
   179ec:	06d12e23          	sw	a3,124(sp)
   179f0:	0cd010ef          	jal	192bc <__rv_alloc_D2A>
   179f4:	00050f13          	mv	t5,a0
   179f8:	4c050063          	beqz	a0,17eb8 <__gdtoa+0x878>
   179fc:	00cda783          	lw	a5,12(s11)
   17a00:	fff78793          	addi	a5,a5,-1
   17a04:	02f12423          	sw	a5,40(sp)
   17a08:	5a078663          	beqz	a5,17fb4 <__gdtoa+0x974>
   17a0c:	0c07cee3          	bltz	a5,182e8 <__gdtoa+0xca8>
   17a10:	100b7e93          	andi	t4,s6,256
   17a14:	580e9863          	bnez	t4,17fa4 <__gdtoa+0x964>
   17a18:	02012783          	lw	a5,32(sp)
   17a1c:	0007c863          	bltz	a5,17a2c <__gdtoa+0x3ec>
   17a20:	03012783          	lw	a5,48(sp)
   17a24:	00e00693          	li	a3,14
   17a28:	10f6d4e3          	bge	a3,a5,18330 <__gdtoa+0xcf0>
   17a2c:	03412783          	lw	a5,52(sp)
   17a30:	0a0784e3          	beqz	a5,182d8 <__gdtoa+0xc98>
   17a34:	02012783          	lw	a5,32(sp)
   17a38:	419a8e33          	sub	t3,s5,s9
   17a3c:	004da683          	lw	a3,4(s11)
   17a40:	001e0613          	addi	a2,t3,1
   17a44:	06c12e23          	sw	a2,124(sp)
   17a48:	41c78e33          	sub	t3,a5,t3
   17a4c:	62de52e3          	bge	t3,a3,18870 <__gdtoa+0x1230>
   17a50:	ffdd0613          	addi	a2,s10,-3
   17a54:	ffd67613          	andi	a2,a2,-3
   17a58:	500604e3          	beqz	a2,18760 <__gdtoa+0x1120>
   17a5c:	40d786b3          	sub	a3,a5,a3
   17a60:	00168693          	addi	a3,a3,1
   17a64:	06d12e23          	sw	a3,124(sp)
   17a68:	00100613          	li	a2,1
   17a6c:	01a65c63          	bge	a2,s10,17a84 <__gdtoa+0x444>
   17a70:	01c12783          	lw	a5,28(sp)
   17a74:	00f05863          	blez	a5,17a84 <__gdtoa+0x444>
   17a78:	01c12783          	lw	a5,28(sp)
   17a7c:	00d7d463          	bge	a5,a3,17a84 <__gdtoa+0x444>
   17a80:	6400106f          	j	190c0 <__gdtoa+0x1a80>
   17a84:	02412783          	lw	a5,36(sp)
   17a88:	02c12483          	lw	s1,44(sp)
   17a8c:	00db8bb3          	add	s7,s7,a3
   17a90:	00078a93          	mv	s5,a5
   17a94:	00f687b3          	add	a5,a3,a5
   17a98:	02f12223          	sw	a5,36(sp)
   17a9c:	00012503          	lw	a0,0(sp)
   17aa0:	00100593          	li	a1,1
   17aa4:	03e12023          	sw	t5,32(sp)
   17aa8:	194020ef          	jal	19c3c <__i2b>
   17aac:	02012f03          	lw	t5,32(sp)
   17ab0:	00050a13          	mv	s4,a0
   17ab4:	40050263          	beqz	a0,17eb8 <__gdtoa+0x878>
   17ab8:	020a8663          	beqz	s5,17ae4 <__gdtoa+0x4a4>
   17abc:	03705463          	blez	s7,17ae4 <__gdtoa+0x4a4>
   17ac0:	000a8693          	mv	a3,s5
   17ac4:	015bd463          	bge	s7,s5,17acc <__gdtoa+0x48c>
   17ac8:	000b8693          	mv	a3,s7
   17acc:	02412783          	lw	a5,36(sp)
   17ad0:	06d12e23          	sw	a3,124(sp)
   17ad4:	40da8ab3          	sub	s5,s5,a3
   17ad8:	40d787b3          	sub	a5,a5,a3
   17adc:	02f12223          	sw	a5,36(sp)
   17ae0:	40db8bb3          	sub	s7,s7,a3
   17ae4:	02c12783          	lw	a5,44(sp)
   17ae8:	02078863          	beqz	a5,17b18 <__gdtoa+0x4d8>
   17aec:	03412783          	lw	a5,52(sp)
   17af0:	00078463          	beqz	a5,17af8 <__gdtoa+0x4b8>
   17af4:	660490e3          	bnez	s1,18954 <__gdtoa+0x1314>
   17af8:	02c12603          	lw	a2,44(sp)
   17afc:	00c12583          	lw	a1,12(sp)
   17b00:	00012503          	lw	a0,0(sp)
   17b04:	03e12023          	sw	t5,32(sp)
   17b08:	418020ef          	jal	19f20 <__pow5mult>
   17b0c:	00a12623          	sw	a0,12(sp)
   17b10:	02012f03          	lw	t5,32(sp)
   17b14:	3a050263          	beqz	a0,17eb8 <__gdtoa+0x878>
   17b18:	00012503          	lw	a0,0(sp)
   17b1c:	00100593          	li	a1,1
   17b20:	03e12023          	sw	t5,32(sp)
   17b24:	118020ef          	jal	19c3c <__i2b>
   17b28:	00050313          	mv	t1,a0
   17b2c:	38050663          	beqz	a0,17eb8 <__gdtoa+0x878>
   17b30:	01812783          	lw	a5,24(sp)
   17b34:	02012f03          	lw	t5,32(sp)
   17b38:	4e0790e3          	bnez	a5,18818 <__gdtoa+0x11d8>
   17b3c:	00100693          	li	a3,1
   17b40:	1da6d0e3          	bge	a3,s10,18500 <__gdtoa+0xec0>
   17b44:	01f00b13          	li	s6,31
   17b48:	02412783          	lw	a5,36(sp)
   17b4c:	417b0b33          	sub	s6,s6,s7
   17b50:	ffcb0b13          	addi	s6,s6,-4
   17b54:	01fb7b13          	andi	s6,s6,31
   17b58:	00fb0633          	add	a2,s6,a5
   17b5c:	07612e23          	sw	s6,124(sp)
   17b60:	000b0793          	mv	a5,s6
   17b64:	2ac044e3          	bgtz	a2,1860c <__gdtoa+0xfcc>
   17b68:	00fb8633          	add	a2,s7,a5
   17b6c:	32c04863          	bgtz	a2,17e9c <__gdtoa+0x85c>
   17b70:	03c12783          	lw	a5,60(sp)
   17b74:	2c0792e3          	bnez	a5,18638 <__gdtoa+0xff8>
   17b78:	01c12783          	lw	a5,28(sp)
   17b7c:	40f058e3          	blez	a5,1878c <__gdtoa+0x114c>
   17b80:	03412783          	lw	a5,52(sp)
   17b84:	30078ee3          	beqz	a5,186a0 <__gdtoa+0x1060>
   17b88:	015b0633          	add	a2,s6,s5
   17b8c:	66c048e3          	bgtz	a2,189fc <__gdtoa+0x13bc>
   17b90:	01812783          	lw	a5,24(sp)
   17b94:	000a0a93          	mv	s5,s4
   17b98:	00078463          	beqz	a5,17ba0 <__gdtoa+0x560>
   17b9c:	6e50006f          	j	18a80 <__gdtoa+0x1440>
   17ba0:	01212a23          	sw	s2,20(sp)
   17ba4:	00c12483          	lw	s1,12(sp)
   17ba8:	00012903          	lw	s2,0(sp)
   17bac:	000f0d93          	mv	s11,t5
   17bb0:	00100793          	li	a5,1
   17bb4:	00200413          	li	s0,2
   17bb8:	00030c13          	mv	s8,t1
   17bbc:	01e12823          	sw	t5,16(sp)
   17bc0:	0b00006f          	j	17c70 <__gdtoa+0x630>
   17bc4:	00090513          	mv	a0,s2
   17bc8:	475010ef          	jal	1983c <_Bfree>
   17bcc:	000b5463          	bgez	s6,17bd4 <__gdtoa+0x594>
   17bd0:	06c0106f          	j	18c3c <__gdtoa+0x15fc>
   17bd4:	016d6b33          	or	s6,s10,s6
   17bd8:	000b1a63          	bnez	s6,17bec <__gdtoa+0x5ac>
   17bdc:	0009a783          	lw	a5,0(s3)
   17be0:	0017f793          	andi	a5,a5,1
   17be4:	00079463          	bnez	a5,17bec <__gdtoa+0x5ac>
   17be8:	0540106f          	j	18c3c <__gdtoa+0x15fc>
   17bec:	02812783          	lw	a5,40(sp)
   17bf0:	00878463          	beq	a5,s0,17bf8 <__gdtoa+0x5b8>
   17bf4:	3c80106f          	j	18fbc <__gdtoa+0x197c>
   17bf8:	019d8023          	sb	s9,0(s11)
   17bfc:	07c12783          	lw	a5,124(sp)
   17c00:	01c12703          	lw	a4,28(sp)
   17c04:	001d8d93          	addi	s11,s11,1
   17c08:	00f71463          	bne	a4,a5,17c10 <__gdtoa+0x5d0>
   17c0c:	2d40106f          	j	18ee0 <__gdtoa+0x18a0>
   17c10:	00048593          	mv	a1,s1
   17c14:	00000693          	li	a3,0
   17c18:	00a00613          	li	a2,10
   17c1c:	00090513          	mv	a0,s2
   17c20:	441010ef          	jal	19860 <__multadd>
   17c24:	00050493          	mv	s1,a0
   17c28:	28050863          	beqz	a0,17eb8 <__gdtoa+0x878>
   17c2c:	00000693          	li	a3,0
   17c30:	00a00613          	li	a2,10
   17c34:	000a0593          	mv	a1,s4
   17c38:	00090513          	mv	a0,s2
   17c3c:	135a00e3          	beq	s4,s5,1855c <__gdtoa+0xf1c>
   17c40:	421010ef          	jal	19860 <__multadd>
   17c44:	00050a13          	mv	s4,a0
   17c48:	26050863          	beqz	a0,17eb8 <__gdtoa+0x878>
   17c4c:	000a8593          	mv	a1,s5
   17c50:	00000693          	li	a3,0
   17c54:	00a00613          	li	a2,10
   17c58:	00090513          	mv	a0,s2
   17c5c:	405010ef          	jal	19860 <__multadd>
   17c60:	00050a93          	mv	s5,a0
   17c64:	24050a63          	beqz	a0,17eb8 <__gdtoa+0x878>
   17c68:	07c12783          	lw	a5,124(sp)
   17c6c:	00178793          	addi	a5,a5,1
   17c70:	000c0593          	mv	a1,s8
   17c74:	00048513          	mv	a0,s1
   17c78:	06f12e23          	sw	a5,124(sp)
   17c7c:	75c010ef          	jal	193d8 <__quorem_D2A>
   17c80:	00050b93          	mv	s7,a0
   17c84:	000a0593          	mv	a1,s4
   17c88:	00048513          	mv	a0,s1
   17c8c:	574020ef          	jal	1a200 <__mcmp>
   17c90:	000c0593          	mv	a1,s8
   17c94:	00050b13          	mv	s6,a0
   17c98:	000a8613          	mv	a2,s5
   17c9c:	00090513          	mv	a0,s2
   17ca0:	5b8020ef          	jal	1a258 <__mdiff>
   17ca4:	030b8c93          	addi	s9,s7,48
   17ca8:	00050593          	mv	a1,a0
   17cac:	20050663          	beqz	a0,17eb8 <__gdtoa+0x878>
   17cb0:	00c52783          	lw	a5,12(a0)
   17cb4:	f00798e3          	bnez	a5,17bc4 <__gdtoa+0x584>
   17cb8:	00a12623          	sw	a0,12(sp)
   17cbc:	00048513          	mv	a0,s1
   17cc0:	540020ef          	jal	1a200 <__mcmp>
   17cc4:	00c12583          	lw	a1,12(sp)
   17cc8:	00050693          	mv	a3,a0
   17ccc:	00090513          	mv	a0,s2
   17cd0:	00d12623          	sw	a3,12(sp)
   17cd4:	369010ef          	jal	1983c <_Bfree>
   17cd8:	00c12683          	lw	a3,12(sp)
   17cdc:	00dd6733          	or	a4,s10,a3
   17ce0:	00070463          	beqz	a4,17ce8 <__gdtoa+0x6a8>
   17ce4:	5680106f          	j	1924c <__gdtoa+0x1c0c>
   17ce8:	0009a783          	lw	a5,0(s3)
   17cec:	0017f793          	andi	a5,a5,1
   17cf0:	080790e3          	bnez	a5,18570 <__gdtoa+0xf30>
   17cf4:	02812783          	lw	a5,40(sp)
   17cf8:	00079463          	bnez	a5,17d00 <__gdtoa+0x6c0>
   17cfc:	4280106f          	j	19124 <__gdtoa+0x1ae4>
   17d00:	ef604ce3          	bgtz	s6,17bf8 <__gdtoa+0x5b8>
   17d04:	0104a603          	lw	a2,16(s1)
   17d08:	00912623          	sw	s1,12(sp)
   17d0c:	00100693          	li	a3,1
   17d10:	000c0313          	mv	t1,s8
   17d14:	01012f03          	lw	t5,16(sp)
   17d18:	01412903          	lw	s2,20(sp)
   17d1c:	00070c13          	mv	s8,a4
   17d20:	00048793          	mv	a5,s1
   17d24:	00c6c463          	blt	a3,a2,17d2c <__gdtoa+0x6ec>
   17d28:	4ec0106f          	j	19214 <__gdtoa+0x1bd4>
   17d2c:	02812783          	lw	a5,40(sp)
   17d30:	00200693          	li	a3,2
   17d34:	00d79463          	bne	a5,a3,17d3c <__gdtoa+0x6fc>
   17d38:	4ac0106f          	j	191e4 <__gdtoa+0x1ba4>
   17d3c:	00c12483          	lw	s1,12(sp)
   17d40:	00012b83          	lw	s7,0(sp)
   17d44:	00030b13          	mv	s6,t1
   17d48:	000f0c13          	mv	s8,t5
   17d4c:	0240006f          	j	17d70 <__gdtoa+0x730>
   17d50:	311010ef          	jal	19860 <__multadd>
   17d54:	000b0593          	mv	a1,s6
   17d58:	00050493          	mv	s1,a0
   17d5c:	14050e63          	beqz	a0,17eb8 <__gdtoa+0x878>
   17d60:	678010ef          	jal	193d8 <__quorem_D2A>
   17d64:	03050c93          	addi	s9,a0,48
   17d68:	00098d93          	mv	s11,s3
   17d6c:	00040a93          	mv	s5,s0
   17d70:	000a8593          	mv	a1,s5
   17d74:	000b0513          	mv	a0,s6
   17d78:	488020ef          	jal	1a200 <__mcmp>
   17d7c:	00050793          	mv	a5,a0
   17d80:	00000693          	li	a3,0
   17d84:	00a00613          	li	a2,10
   17d88:	000a8593          	mv	a1,s5
   17d8c:	000b8513          	mv	a0,s7
   17d90:	001d8993          	addi	s3,s11,1
   17d94:	00f04463          	bgtz	a5,17d9c <__gdtoa+0x75c>
   17d98:	42c0106f          	j	191c4 <__gdtoa+0x1b84>
   17d9c:	ff998fa3          	sb	s9,-1(s3)
   17da0:	2c1010ef          	jal	19860 <__multadd>
   17da4:	00050413          	mv	s0,a0
   17da8:	00000693          	li	a3,0
   17dac:	00a00613          	li	a2,10
   17db0:	00048593          	mv	a1,s1
   17db4:	000b8513          	mv	a0,s7
   17db8:	10040063          	beqz	s0,17eb8 <__gdtoa+0x878>
   17dbc:	f95a1ae3          	bne	s4,s5,17d50 <__gdtoa+0x710>
   17dc0:	00040a13          	mv	s4,s0
   17dc4:	f8dff06f          	j	17d50 <__gdtoa+0x710>
   17dc8:	00400793          	li	a5,4
   17dcc:	10f31863          	bne	t1,a5,17edc <__gdtoa+0x89c>
   17dd0:	00412703          	lw	a4,4(sp)
   17dd4:	00812603          	lw	a2,8(sp)
   17dd8:	09812c03          	lw	s8,152(sp)
   17ddc:	0bc12083          	lw	ra,188(sp)
   17de0:	0b012903          	lw	s2,176(sp)
   17de4:	0a012b03          	lw	s6,160(sp)
   17de8:	ffff87b7          	lui	a5,0xffff8
   17dec:	00f72023          	sw	a5,0(a4)
   17df0:	00300693          	li	a3,3
   17df4:	00009597          	auipc	a1,0x9
   17df8:	da458593          	addi	a1,a1,-604 # 20b98 <_exit+0x1e8>
   17dfc:	0c010113          	addi	sp,sp,192
   17e00:	50c0106f          	j	1930c <__nrv_alloc_D2A>
   17e04:	00012503          	lw	a0,0(sp)
   17e08:	00078593          	mv	a1,a5
   17e0c:	231010ef          	jal	1983c <_Bfree>
   17e10:	0ac12983          	lw	s3,172(sp)
   17e14:	0a812a03          	lw	s4,168(sp)
   17e18:	0a412a83          	lw	s5,164(sp)
   17e1c:	09412c83          	lw	s9,148(sp)
   17e20:	09012d03          	lw	s10,144(sp)
   17e24:	08c12d83          	lw	s11,140(sp)
   17e28:	00412703          	lw	a4,4(sp)
   17e2c:	00812603          	lw	a2,8(sp)
   17e30:	00012503          	lw	a0,0(sp)
   17e34:	00100793          	li	a5,1
   17e38:	00f72023          	sw	a5,0(a4)
   17e3c:	00100693          	li	a3,1
   17e40:	00009597          	auipc	a1,0x9
   17e44:	d3858593          	addi	a1,a1,-712 # 20b78 <_exit+0x1c8>
   17e48:	4c4010ef          	jal	1930c <__nrv_alloc_D2A>
   17e4c:	00050f13          	mv	t5,a0
   17e50:	0bc12083          	lw	ra,188(sp)
   17e54:	09812c03          	lw	s8,152(sp)
   17e58:	0b012903          	lw	s2,176(sp)
   17e5c:	0a012b03          	lw	s6,160(sp)
   17e60:	000f0513          	mv	a0,t5
   17e64:	0c010113          	addi	sp,sp,192
   17e68:	00008067          	ret
   17e6c:	00412703          	lw	a4,4(sp)
   17e70:	00812603          	lw	a2,8(sp)
   17e74:	0bc12083          	lw	ra,188(sp)
   17e78:	0b012903          	lw	s2,176(sp)
   17e7c:	0a012b03          	lw	s6,160(sp)
   17e80:	ffff87b7          	lui	a5,0xffff8
   17e84:	00f72023          	sw	a5,0(a4)
   17e88:	00800693          	li	a3,8
   17e8c:	00009597          	auipc	a1,0x9
   17e90:	d0058593          	addi	a1,a1,-768 # 20b8c <_exit+0x1dc>
   17e94:	0c010113          	addi	sp,sp,192
   17e98:	4740106f          	j	1930c <__nrv_alloc_D2A>
   17e9c:	00012503          	lw	a0,0(sp)
   17ea0:	00030593          	mv	a1,t1
   17ea4:	01e12823          	sw	t5,16(sp)
   17ea8:	1c8020ef          	jal	1a070 <__lshift>
   17eac:	01012f03          	lw	t5,16(sp)
   17eb0:	00050313          	mv	t1,a0
   17eb4:	ca051ee3          	bnez	a0,17b70 <__gdtoa+0x530>
   17eb8:	0b812403          	lw	s0,184(sp)
   17ebc:	0b412483          	lw	s1,180(sp)
   17ec0:	0ac12983          	lw	s3,172(sp)
   17ec4:	0a812a03          	lw	s4,168(sp)
   17ec8:	0a412a83          	lw	s5,164(sp)
   17ecc:	09c12b83          	lw	s7,156(sp)
   17ed0:	09412c83          	lw	s9,148(sp)
   17ed4:	09012d03          	lw	s10,144(sp)
   17ed8:	08c12d83          	lw	s11,140(sp)
   17edc:	00000f13          	li	t5,0
   17ee0:	f71ff06f          	j	17e50 <__gdtoa+0x810>
   17ee4:	00c12783          	lw	a5,12(sp)
   17ee8:	0007a823          	sw	zero,16(a5) # ffff8010 <__BSS_END__+0xfffd52e0>
   17eec:	891ff06f          	j	1777c <__gdtoa+0x13c>
   17ef0:	00100793          	li	a5,1
   17ef4:	02f12e23          	sw	a5,60(sp)
   17ef8:	02012223          	sw	zero,36(sp)
   17efc:	3c0bc463          	bltz	s7,182c4 <__gdtoa+0xc84>
   17f00:	01812783          	lw	a5,24(sp)
   17f04:	a607d4e3          	bgez	a5,1796c <__gdtoa+0x32c>
   17f08:	01812783          	lw	a5,24(sp)
   17f0c:	02412703          	lw	a4,36(sp)
   17f10:	00012c23          	sw	zero,24(sp)
   17f14:	02f12823          	sw	a5,48(sp)
   17f18:	40f70733          	sub	a4,a4,a5
   17f1c:	02e12223          	sw	a4,36(sp)
   17f20:	40f00733          	neg	a4,a5
   17f24:	02e12623          	sw	a4,44(sp)
   17f28:	a55ff06f          	j	1797c <__gdtoa+0x33c>
   17f2c:	00050593          	mv	a1,a0
   17f30:	00c12503          	lw	a0,12(sp)
   17f34:	69c010ef          	jal	195d0 <__rshift_D2A>
   17f38:	07c12683          	lw	a3,124(sp)
   17f3c:	01012783          	lw	a5,16(sp)
   17f40:	40dc8cb3          	sub	s9,s9,a3
   17f44:	00f687b3          	add	a5,a3,a5
   17f48:	02f12023          	sw	a5,32(sp)
   17f4c:	849ff06f          	j	17794 <__gdtoa+0x154>
   17f50:	01c12783          	lw	a5,28(sp)
   17f54:	00000d13          	li	s10,0
   17f58:	3fd78793          	addi	a5,a5,1021
   17f5c:	7f87ba13          	sltiu	s4,a5,2040
   17f60:	000a8513          	mv	a0,s5
   17f64:	405050ef          	jal	1db68 <__floatsidf>
   17f68:	0000b697          	auipc	a3,0xb
   17f6c:	96868693          	addi	a3,a3,-1688 # 228d0 <__SDATA_BEGIN__+0x28>
   17f70:	0006a603          	lw	a2,0(a3)
   17f74:	0046a683          	lw	a3,4(a3)
   17f78:	5d9040ef          	jal	1cd50 <__muldf3>
   17f7c:	36d050ef          	jal	1dae8 <__fixdfsi>
   17f80:	00100793          	li	a5,1
   17f84:	00350593          	addi	a1,a0,3
   17f88:	02f12a23          	sw	a5,52(sp)
   17f8c:	fff00793          	li	a5,-1
   17f90:	00058693          	mv	a3,a1
   17f94:	00012a23          	sw	zero,20(sp)
   17f98:	04f12623          	sw	a5,76(sp)
   17f9c:	00f12e23          	sw	a5,28(sp)
   17fa0:	a49ff06f          	j	179e8 <__gdtoa+0x3a8>
   17fa4:	02812783          	lw	a5,40(sp)
   17fa8:	00300693          	li	a3,3
   17fac:	40f687b3          	sub	a5,a3,a5
   17fb0:	02f12423          	sw	a5,40(sp)
   17fb4:	01c12483          	lw	s1,28(sp)
   17fb8:	00e00693          	li	a3,14
   17fbc:	a496eee3          	bltu	a3,s1,17a18 <__gdtoa+0x3d8>
   17fc0:	a40a0ce3          	beqz	s4,17a18 <__gdtoa+0x3d8>
   17fc4:	03012783          	lw	a5,48(sp)
   17fc8:	02812703          	lw	a4,40(sp)
   17fcc:	00e7e6b3          	or	a3,a5,a4
   17fd0:	a40694e3          	bnez	a3,17a18 <__gdtoa+0x3d8>
   17fd4:	03812403          	lw	s0,56(sp)
   17fd8:	04012a03          	lw	s4,64(sp)
   17fdc:	03c12783          	lw	a5,60(sp)
   17fe0:	06012e23          	sw	zero,124(sp)
   17fe4:	02812423          	sw	s0,40(sp)
   17fe8:	03412c23          	sw	s4,56(sp)
   17fec:	02078863          	beqz	a5,1801c <__gdtoa+0x9dc>
   17ff0:	0000b697          	auipc	a3,0xb
   17ff4:	8e868693          	addi	a3,a3,-1816 # 228d8 <__SDATA_BEGIN__+0x30>
   17ff8:	0006a603          	lw	a2,0(a3)
   17ffc:	0046a683          	lw	a3,4(a3)
   18000:	00040513          	mv	a0,s0
   18004:	000a0593          	mv	a1,s4
   18008:	05e12823          	sw	t5,80(sp)
   1800c:	469040ef          	jal	1cc74 <__ledf2>
   18010:	05012f03          	lw	t5,80(sp)
   18014:	00055463          	bgez	a0,1801c <__gdtoa+0x9dc>
   18018:	6ed0006f          	j	18f04 <__gdtoa+0x18c4>
   1801c:	02812783          	lw	a5,40(sp)
   18020:	05e12823          	sw	t5,80(sp)
   18024:	00078613          	mv	a2,a5
   18028:	00078513          	mv	a0,a5
   1802c:	03812783          	lw	a5,56(sp)
   18030:	00078693          	mv	a3,a5
   18034:	00078593          	mv	a1,a5
   18038:	461030ef          	jal	1bc98 <__adddf3>
   1803c:	0000b697          	auipc	a3,0xb
   18040:	8b468693          	addi	a3,a3,-1868 # 228f0 <__SDATA_BEGIN__+0x48>
   18044:	0006a603          	lw	a2,0(a3)
   18048:	0046a683          	lw	a3,4(a3)
   1804c:	44d030ef          	jal	1bc98 <__adddf3>
   18050:	01c12783          	lw	a5,28(sp)
   18054:	fcc00737          	lui	a4,0xfcc00
   18058:	05012f03          	lw	t5,80(sp)
   1805c:	00050b13          	mv	s6,a0
   18060:	00b70a33          	add	s4,a4,a1
   18064:	2a078ce3          	beqz	a5,18b1c <__gdtoa+0x14dc>
   18068:	01c12783          	lw	a5,28(sp)
   1806c:	02812e83          	lw	t4,40(sp)
   18070:	03812803          	lw	a6,56(sp)
   18074:	04012a23          	sw	zero,84(sp)
   18078:	04f12823          	sw	a5,80(sp)
   1807c:	05012783          	lw	a5,80(sp)
   18080:	03412703          	lw	a4,52(sp)
   18084:	00009317          	auipc	t1,0x9
   18088:	08c30313          	addi	t1,t1,140 # 21110 <__mprec_tens>
   1808c:	fff78693          	addi	a3,a5,-1
   18090:	00369693          	slli	a3,a3,0x3
   18094:	00d306b3          	add	a3,t1,a3
   18098:	05012e23          	sw	a6,92(sp)
   1809c:	05d12c23          	sw	t4,88(sp)
   180a0:	0006a603          	lw	a2,0(a3)
   180a4:	000b0493          	mv	s1,s6
   180a8:	0046a683          	lw	a3,4(a3)
   180ac:	480704e3          	beqz	a4,18d34 <__gdtoa+0x16f4>
   180b0:	0000b597          	auipc	a1,0xb
   180b4:	85058593          	addi	a1,a1,-1968 # 22900 <__SDATA_BEGIN__+0x58>
   180b8:	0005a503          	lw	a0,0(a1)
   180bc:	0045a583          	lw	a1,4(a1)
   180c0:	06612423          	sw	t1,104(sp)
   180c4:	001f0b13          	addi	s6,t5,1
   180c8:	07e12023          	sw	t5,96(sp)
   180cc:	360040ef          	jal	1c42c <__divdf3>
   180d0:	00048613          	mv	a2,s1
   180d4:	000a0693          	mv	a3,s4
   180d8:	268050ef          	jal	1d340 <__subdf3>
   180dc:	05812e83          	lw	t4,88(sp)
   180e0:	05c12803          	lw	a6,92(sp)
   180e4:	00050613          	mv	a2,a0
   180e8:	00058693          	mv	a3,a1
   180ec:	000e8513          	mv	a0,t4
   180f0:	00080593          	mv	a1,a6
   180f4:	05d12423          	sw	t4,72(sp)
   180f8:	05012023          	sw	a6,64(sp)
   180fc:	04c12c23          	sw	a2,88(sp)
   18100:	04d12e23          	sw	a3,92(sp)
   18104:	1e5050ef          	jal	1dae8 <__fixdfsi>
   18108:	00050413          	mv	s0,a0
   1810c:	25d050ef          	jal	1db68 <__floatsidf>
   18110:	04012803          	lw	a6,64(sp)
   18114:	04812e83          	lw	t4,72(sp)
   18118:	00050613          	mv	a2,a0
   1811c:	00058693          	mv	a3,a1
   18120:	000e8513          	mv	a0,t4
   18124:	00080593          	mv	a1,a6
   18128:	218050ef          	jal	1d340 <__subdf3>
   1812c:	06012f03          	lw	t5,96(sp)
   18130:	00050613          	mv	a2,a0
   18134:	00058693          	mv	a3,a1
   18138:	00050493          	mv	s1,a0
   1813c:	00058a13          	mv	s4,a1
   18140:	05812503          	lw	a0,88(sp)
   18144:	05c12583          	lw	a1,92(sp)
   18148:	03040793          	addi	a5,s0,48
   1814c:	00ff0023          	sb	a5,0(t5)
   18150:	05e12023          	sw	t5,64(sp)
   18154:	245040ef          	jal	1cb98 <__gedf2>
   18158:	04012f03          	lw	t5,64(sp)
   1815c:	00a05463          	blez	a0,18164 <__gdtoa+0xb24>
   18160:	71d0006f          	j	1907c <__gdtoa+0x1a3c>
   18164:	0000a697          	auipc	a3,0xa
   18168:	77468693          	addi	a3,a3,1908 # 228d8 <__SDATA_BEGIN__+0x30>
   1816c:	0006a783          	lw	a5,0(a3)
   18170:	0046a803          	lw	a6,4(a3)
   18174:	07712023          	sw	s7,96(sp)
   18178:	07512223          	sw	s5,100(sp)
   1817c:	05012b83          	lw	s7,80(sp)
   18180:	05812403          	lw	s0,88(sp)
   18184:	05c12a83          	lw	s5,92(sp)
   18188:	04f12023          	sw	a5,64(sp)
   1818c:	05012223          	sw	a6,68(sp)
   18190:	05e12423          	sw	t5,72(sp)
   18194:	05212823          	sw	s2,80(sp)
   18198:	0940006f          	j	1822c <__gdtoa+0xbec>
   1819c:	07c12783          	lw	a5,124(sp)
   181a0:	00178793          	addi	a5,a5,1
   181a4:	06f12e23          	sw	a5,124(sp)
   181a8:	0177c463          	blt	a5,s7,181b0 <__gdtoa+0xb70>
   181ac:	7450006f          	j	190f0 <__gdtoa+0x1ab0>
   181b0:	0000a917          	auipc	s2,0xa
   181b4:	73090913          	addi	s2,s2,1840 # 228e0 <__SDATA_BEGIN__+0x38>
   181b8:	00092603          	lw	a2,0(s2)
   181bc:	00492683          	lw	a3,4(s2)
   181c0:	001b0b13          	addi	s6,s6,1
   181c4:	38d040ef          	jal	1cd50 <__muldf3>
   181c8:	00092603          	lw	a2,0(s2)
   181cc:	00492683          	lw	a3,4(s2)
   181d0:	00050413          	mv	s0,a0
   181d4:	00058a93          	mv	s5,a1
   181d8:	00048513          	mv	a0,s1
   181dc:	000a0593          	mv	a1,s4
   181e0:	371040ef          	jal	1cd50 <__muldf3>
   181e4:	00058a13          	mv	s4,a1
   181e8:	00050913          	mv	s2,a0
   181ec:	0fd050ef          	jal	1dae8 <__fixdfsi>
   181f0:	00050493          	mv	s1,a0
   181f4:	175050ef          	jal	1db68 <__floatsidf>
   181f8:	00050613          	mv	a2,a0
   181fc:	00058693          	mv	a3,a1
   18200:	00090513          	mv	a0,s2
   18204:	000a0593          	mv	a1,s4
   18208:	138050ef          	jal	1d340 <__subdf3>
   1820c:	03048793          	addi	a5,s1,48
   18210:	00040613          	mv	a2,s0
   18214:	000a8693          	mv	a3,s5
   18218:	fefb0fa3          	sb	a5,-1(s6)
   1821c:	00050493          	mv	s1,a0
   18220:	00058a13          	mv	s4,a1
   18224:	251040ef          	jal	1cc74 <__ledf2>
   18228:	640546e3          	bltz	a0,19074 <__gdtoa+0x1a34>
   1822c:	04012503          	lw	a0,64(sp)
   18230:	04412583          	lw	a1,68(sp)
   18234:	00048613          	mv	a2,s1
   18238:	000a0693          	mv	a3,s4
   1823c:	104050ef          	jal	1d340 <__subdf3>
   18240:	00050613          	mv	a2,a0
   18244:	00058693          	mv	a3,a1
   18248:	00040513          	mv	a0,s0
   1824c:	000a8593          	mv	a1,s5
   18250:	149040ef          	jal	1cb98 <__gedf2>
   18254:	00050793          	mv	a5,a0
   18258:	000a8593          	mv	a1,s5
   1825c:	00040513          	mv	a0,s0
   18260:	f2f05ee3          	blez	a5,1819c <__gdtoa+0xb5c>
   18264:	05412783          	lw	a5,84(sp)
   18268:	04812f03          	lw	t5,72(sp)
   1826c:	05012903          	lw	s2,80(sp)
   18270:	fffb4703          	lbu	a4,-1(s6)
   18274:	00178413          	addi	s0,a5,1
   18278:	03900693          	li	a3,57
   1827c:	0100006f          	j	1828c <__gdtoa+0xc4c>
   18280:	2aff04e3          	beq	t5,a5,18d28 <__gdtoa+0x16e8>
   18284:	fff7c703          	lbu	a4,-1(a5)
   18288:	00078b13          	mv	s6,a5
   1828c:	fffb0793          	addi	a5,s6,-1
   18290:	fed708e3          	beq	a4,a3,18280 <__gdtoa+0xc40>
   18294:	00170693          	addi	a3,a4,1 # fcc00001 <__BSS_END__+0xfcbdd2d1>
   18298:	0ff6f693          	zext.b	a3,a3
   1829c:	00d78023          	sb	a3,0(a5)
   182a0:	00040493          	mv	s1,s0
   182a4:	02000c13          	li	s8,32
   182a8:	1f80006f          	j	184a0 <__gdtoa+0xe60>
   182ac:	0ac12983          	lw	s3,172(sp)
   182b0:	0a412a83          	lw	s5,164(sp)
   182b4:	09012d03          	lw	s10,144(sp)
   182b8:	08c12d83          	lw	s11,140(sp)
   182bc:	00000f13          	li	t5,0
   182c0:	b91ff06f          	j	17e50 <__gdtoa+0x810>
   182c4:	00100693          	li	a3,1
   182c8:	408687b3          	sub	a5,a3,s0
   182cc:	02f12223          	sw	a5,36(sp)
   182d0:	00000b93          	li	s7,0
   182d4:	c2dff06f          	j	17f00 <__gdtoa+0x8c0>
   182d8:	02c12483          	lw	s1,44(sp)
   182dc:	02412a83          	lw	s5,36(sp)
   182e0:	00000a13          	li	s4,0
   182e4:	fd4ff06f          	j	17ab8 <__gdtoa+0x478>
   182e8:	00200793          	li	a5,2
   182ec:	02f12423          	sw	a5,40(sp)
   182f0:	f20ff06f          	j	17a10 <__gdtoa+0x3d0>
   182f4:	01812783          	lw	a5,24(sp)
   182f8:	02012e23          	sw	zero,60(sp)
   182fc:	fff78793          	addi	a5,a5,-1
   18300:	00f12c23          	sw	a5,24(sp)
   18304:	bf5ff06f          	j	17ef8 <__gdtoa+0x8b8>
   18308:	ffcd0d13          	addi	s10,s10,-4
   1830c:	00400613          	li	a2,4
   18310:	22cd0063          	beq	s10,a2,18530 <__gdtoa+0xef0>
   18314:	70dd0863          	beq	s10,a3,18a24 <__gdtoa+0x13e4>
   18318:	00200693          	li	a3,2
   1831c:	02012a23          	sw	zero,52(sp)
   18320:	00000a13          	li	s4,0
   18324:	20dd0c63          	beq	s10,a3,1853c <__gdtoa+0xefc>
   18328:	00300d13          	li	s10,3
   1832c:	e98ff06f          	j	179c4 <__gdtoa+0x384>
   18330:	00379693          	slli	a3,a5,0x3
   18334:	00009797          	auipc	a5,0x9
   18338:	ddc78793          	addi	a5,a5,-548 # 21110 <__mprec_tens>
   1833c:	00d787b3          	add	a5,a5,a3
   18340:	0007ad03          	lw	s10,0(a5)
   18344:	0047ad83          	lw	s11,4(a5)
   18348:	01412783          	lw	a5,20(sp)
   1834c:	5607c463          	bltz	a5,188b4 <__gdtoa+0x1274>
   18350:	04812403          	lw	s0,72(sp)
   18354:	04012a03          	lw	s4,64(sp)
   18358:	00100793          	li	a5,1
   1835c:	000d0613          	mv	a2,s10
   18360:	000d8693          	mv	a3,s11
   18364:	00040513          	mv	a0,s0
   18368:	000a0593          	mv	a1,s4
   1836c:	01e12823          	sw	t5,16(sp)
   18370:	06f12e23          	sw	a5,124(sp)
   18374:	0b8040ef          	jal	1c42c <__divdf3>
   18378:	770050ef          	jal	1dae8 <__fixdfsi>
   1837c:	00050993          	mv	s3,a0
   18380:	7e8050ef          	jal	1db68 <__floatsidf>
   18384:	000d0613          	mv	a2,s10
   18388:	000d8693          	mv	a3,s11
   1838c:	1c5040ef          	jal	1cd50 <__muldf3>
   18390:	00050613          	mv	a2,a0
   18394:	00058693          	mv	a3,a1
   18398:	00040513          	mv	a0,s0
   1839c:	000a0593          	mv	a1,s4
   183a0:	7a1040ef          	jal	1d340 <__subdf3>
   183a4:	01012f03          	lw	t5,16(sp)
   183a8:	03012703          	lw	a4,48(sp)
   183ac:	03098793          	addi	a5,s3,48
   183b0:	00ff0023          	sb	a5,0(t5)
   183b4:	00000613          	li	a2,0
   183b8:	00000693          	li	a3,0
   183bc:	00170413          	addi	s0,a4,1
   183c0:	001f0b13          	addi	s6,t5,1
   183c4:	00050a93          	mv	s5,a0
   183c8:	00058a13          	mv	s4,a1
   183cc:	740040ef          	jal	1cb0c <__eqdf2>
   183d0:	01012f03          	lw	t5,16(sp)
   183d4:	00040493          	mv	s1,s0
   183d8:	0c050463          	beqz	a0,184a0 <__gdtoa+0xe60>
   183dc:	01812823          	sw	s8,16(sp)
   183e0:	01c12c83          	lw	s9,28(sp)
   183e4:	000a0c13          	mv	s8,s4
   183e8:	0000ab97          	auipc	s7,0xa
   183ec:	4f8b8b93          	addi	s7,s7,1272 # 228e0 <__SDATA_BEGIN__+0x38>
   183f0:	00040a13          	mv	s4,s0
   183f4:	000f0413          	mv	s0,t5
   183f8:	0780006f          	j	18470 <__gdtoa+0xe30>
   183fc:	000ba603          	lw	a2,0(s7)
   18400:	004ba683          	lw	a3,4(s7)
   18404:	07012e23          	sw	a6,124(sp)
   18408:	001b0b13          	addi	s6,s6,1
   1840c:	145040ef          	jal	1cd50 <__muldf3>
   18410:	000d0613          	mv	a2,s10
   18414:	000d8693          	mv	a3,s11
   18418:	00050c13          	mv	s8,a0
   1841c:	00058a93          	mv	s5,a1
   18420:	00c040ef          	jal	1c42c <__divdf3>
   18424:	6c4050ef          	jal	1dae8 <__fixdfsi>
   18428:	00050993          	mv	s3,a0
   1842c:	73c050ef          	jal	1db68 <__floatsidf>
   18430:	000d0613          	mv	a2,s10
   18434:	000d8693          	mv	a3,s11
   18438:	119040ef          	jal	1cd50 <__muldf3>
   1843c:	00050613          	mv	a2,a0
   18440:	00058693          	mv	a3,a1
   18444:	000c0513          	mv	a0,s8
   18448:	000a8593          	mv	a1,s5
   1844c:	6f5040ef          	jal	1d340 <__subdf3>
   18450:	03098793          	addi	a5,s3,48
   18454:	fefb0fa3          	sb	a5,-1(s6)
   18458:	00000613          	li	a2,0
   1845c:	00000693          	li	a3,0
   18460:	00050a93          	mv	s5,a0
   18464:	00058c13          	mv	s8,a1
   18468:	6a4040ef          	jal	1cb0c <__eqdf2>
   1846c:	42050e63          	beqz	a0,188a8 <__gdtoa+0x1268>
   18470:	07c12703          	lw	a4,124(sp)
   18474:	000a8513          	mv	a0,s5
   18478:	000c0593          	mv	a1,s8
   1847c:	00170813          	addi	a6,a4,1
   18480:	f7971ee3          	bne	a4,s9,183fc <__gdtoa+0xdbc>
   18484:	02812703          	lw	a4,40(sp)
   18488:	00040f13          	mv	t5,s0
   1848c:	000a0413          	mv	s0,s4
   18490:	02070ae3          	beqz	a4,18cc4 <__gdtoa+0x1684>
   18494:	00100793          	li	a5,1
   18498:	01000c13          	li	s8,16
   1849c:	26f700e3          	beq	a4,a5,18efc <__gdtoa+0x18bc>
   184a0:	00c12583          	lw	a1,12(sp)
   184a4:	00012503          	lw	a0,0(sp)
   184a8:	01e12823          	sw	t5,16(sp)
   184ac:	390010ef          	jal	1983c <_Bfree>
   184b0:	00412783          	lw	a5,4(sp)
   184b4:	000b0023          	sb	zero,0(s6)
   184b8:	01012f03          	lw	t5,16(sp)
   184bc:	0097a023          	sw	s1,0(a5)
   184c0:	00812783          	lw	a5,8(sp)
   184c4:	00078463          	beqz	a5,184cc <__gdtoa+0xe8c>
   184c8:	0167a023          	sw	s6,0(a5)
   184cc:	00092783          	lw	a5,0(s2)
   184d0:	0b812403          	lw	s0,184(sp)
   184d4:	0b412483          	lw	s1,180(sp)
   184d8:	0187e7b3          	or	a5,a5,s8
   184dc:	0ac12983          	lw	s3,172(sp)
   184e0:	0a812a03          	lw	s4,168(sp)
   184e4:	0a412a83          	lw	s5,164(sp)
   184e8:	09c12b83          	lw	s7,156(sp)
   184ec:	09412c83          	lw	s9,148(sp)
   184f0:	09012d03          	lw	s10,144(sp)
   184f4:	08c12d83          	lw	s11,140(sp)
   184f8:	00f92023          	sw	a5,0(s2)
   184fc:	955ff06f          	j	17e50 <__gdtoa+0x810>
   18500:	e4dc9263          	bne	s9,a3,17b44 <__gdtoa+0x504>
   18504:	004da783          	lw	a5,4(s11)
   18508:	01012703          	lw	a4,16(sp)
   1850c:	00178793          	addi	a5,a5,1
   18510:	e2e7da63          	bge	a5,a4,17b44 <__gdtoa+0x504>
   18514:	02412783          	lw	a5,36(sp)
   18518:	001b8b93          	addi	s7,s7,1
   1851c:	00178793          	addi	a5,a5,1
   18520:	02f12223          	sw	a5,36(sp)
   18524:	00100793          	li	a5,1
   18528:	00f12c23          	sw	a5,24(sp)
   1852c:	e18ff06f          	j	17b44 <__gdtoa+0x504>
   18530:	00100793          	li	a5,1
   18534:	00000a13          	li	s4,0
   18538:	02f12a23          	sw	a5,52(sp)
   1853c:	01412583          	lw	a1,20(sp)
   18540:	00b04463          	bgtz	a1,18548 <__gdtoa+0xf08>
   18544:	00100593          	li	a1,1
   18548:	00058693          	mv	a3,a1
   1854c:	04b12623          	sw	a1,76(sp)
   18550:	00b12e23          	sw	a1,28(sp)
   18554:	00b12a23          	sw	a1,20(sp)
   18558:	c90ff06f          	j	179e8 <__gdtoa+0x3a8>
   1855c:	304010ef          	jal	19860 <__multadd>
   18560:	00050a13          	mv	s4,a0
   18564:	94050ae3          	beqz	a0,17eb8 <__gdtoa+0x878>
   18568:	00050a93          	mv	s5,a0
   1856c:	efcff06f          	j	17c68 <__gdtoa+0x628>
   18570:	e80b5463          	bgez	s6,17bf8 <__gdtoa+0x5b8>
   18574:	02812783          	lw	a5,40(sp)
   18578:	00912623          	sw	s1,12(sp)
   1857c:	000c0313          	mv	t1,s8
   18580:	01012f03          	lw	t5,16(sp)
   18584:	01412903          	lw	s2,20(sp)
   18588:	00070c13          	mv	s8,a4
   1858c:	46079ae3          	bnez	a5,19200 <__gdtoa+0x1bc0>
   18590:	00c12783          	lw	a5,12(sp)
   18594:	00100693          	li	a3,1
   18598:	01000c13          	li	s8,16
   1859c:	0107a603          	lw	a2,16(a5)
   185a0:	001d8993          	addi	s3,s11,1
   185a4:	3ec6d6e3          	bge	a3,a2,19190 <__gdtoa+0x1b50>
   185a8:	000a0413          	mv	s0,s4
   185ac:	00098b13          	mv	s6,s3
   185b0:	019d8023          	sb	s9,0(s11)
   185b4:	000a8a13          	mv	s4,s5
   185b8:	00012983          	lw	s3,0(sp)
   185bc:	00030593          	mv	a1,t1
   185c0:	01e12823          	sw	t5,16(sp)
   185c4:	00098513          	mv	a0,s3
   185c8:	274010ef          	jal	1983c <_Bfree>
   185cc:	03012783          	lw	a5,48(sp)
   185d0:	01012f03          	lw	t5,16(sp)
   185d4:	00178493          	addi	s1,a5,1
   185d8:	ec0a04e3          	beqz	s4,184a0 <__gdtoa+0xe60>
   185dc:	00040c63          	beqz	s0,185f4 <__gdtoa+0xfb4>
   185e0:	01440a63          	beq	s0,s4,185f4 <__gdtoa+0xfb4>
   185e4:	00040593          	mv	a1,s0
   185e8:	00098513          	mv	a0,s3
   185ec:	250010ef          	jal	1983c <_Bfree>
   185f0:	01012f03          	lw	t5,16(sp)
   185f4:	00012503          	lw	a0,0(sp)
   185f8:	000a0593          	mv	a1,s4
   185fc:	01e12823          	sw	t5,16(sp)
   18600:	23c010ef          	jal	1983c <_Bfree>
   18604:	01012f03          	lw	t5,16(sp)
   18608:	e99ff06f          	j	184a0 <__gdtoa+0xe60>
   1860c:	00c12583          	lw	a1,12(sp)
   18610:	00012503          	lw	a0,0(sp)
   18614:	03e12023          	sw	t5,32(sp)
   18618:	00612823          	sw	t1,16(sp)
   1861c:	255010ef          	jal	1a070 <__lshift>
   18620:	00a12623          	sw	a0,12(sp)
   18624:	88050ae3          	beqz	a0,17eb8 <__gdtoa+0x878>
   18628:	07c12783          	lw	a5,124(sp)
   1862c:	02012f03          	lw	t5,32(sp)
   18630:	01012303          	lw	t1,16(sp)
   18634:	d34ff06f          	j	17b68 <__gdtoa+0x528>
   18638:	00c12503          	lw	a0,12(sp)
   1863c:	00030593          	mv	a1,t1
   18640:	03e12023          	sw	t5,32(sp)
   18644:	00612823          	sw	t1,16(sp)
   18648:	3b9010ef          	jal	1a200 <__mcmp>
   1864c:	01012303          	lw	t1,16(sp)
   18650:	02012f03          	lw	t5,32(sp)
   18654:	d2055263          	bgez	a0,17b78 <__gdtoa+0x538>
   18658:	03012783          	lw	a5,48(sp)
   1865c:	00c12583          	lw	a1,12(sp)
   18660:	00012503          	lw	a0,0(sp)
   18664:	fff78793          	addi	a5,a5,-1
   18668:	00000693          	li	a3,0
   1866c:	00a00613          	li	a2,10
   18670:	01e12e23          	sw	t5,28(sp)
   18674:	02f12823          	sw	a5,48(sp)
   18678:	1e8010ef          	jal	19860 <__multadd>
   1867c:	00a12623          	sw	a0,12(sp)
   18680:	82050ce3          	beqz	a0,17eb8 <__gdtoa+0x878>
   18684:	03412783          	lw	a5,52(sp)
   18688:	01012303          	lw	t1,16(sp)
   1868c:	01c12f03          	lw	t5,28(sp)
   18690:	18079ce3          	bnez	a5,19028 <__gdtoa+0x19e8>
   18694:	04c12783          	lw	a5,76(sp)
   18698:	1ef05e63          	blez	a5,18894 <__gdtoa+0x1254>
   1869c:	00f12e23          	sw	a5,28(sp)
   186a0:	01c12483          	lw	s1,28(sp)
   186a4:	00c12403          	lw	s0,12(sp)
   186a8:	00012a83          	lw	s5,0(sp)
   186ac:	000f0d93          	mv	s11,t5
   186b0:	00100793          	li	a5,1
   186b4:	00030993          	mv	s3,t1
   186b8:	000f0b13          	mv	s6,t5
   186bc:	0180006f          	j	186d4 <__gdtoa+0x1094>
   186c0:	1a0010ef          	jal	19860 <__multadd>
   186c4:	00050413          	mv	s0,a0
   186c8:	fe050863          	beqz	a0,17eb8 <__gdtoa+0x878>
   186cc:	07c12783          	lw	a5,124(sp)
   186d0:	00178793          	addi	a5,a5,1
   186d4:	00098593          	mv	a1,s3
   186d8:	00040513          	mv	a0,s0
   186dc:	06f12e23          	sw	a5,124(sp)
   186e0:	4f9000ef          	jal	193d8 <__quorem_D2A>
   186e4:	03050c93          	addi	s9,a0,48
   186e8:	019d8023          	sb	s9,0(s11)
   186ec:	07c12783          	lw	a5,124(sp)
   186f0:	00000693          	li	a3,0
   186f4:	00a00613          	li	a2,10
   186f8:	00040593          	mv	a1,s0
   186fc:	000a8513          	mv	a0,s5
   18700:	001d8d93          	addi	s11,s11,1
   18704:	fa97cee3          	blt	a5,s1,186c0 <__gdtoa+0x1080>
   18708:	00812623          	sw	s0,12(sp)
   1870c:	00098313          	mv	t1,s3
   18710:	000b0f13          	mv	t5,s6
   18714:	00000413          	li	s0,0
   18718:	02812703          	lw	a4,40(sp)
   1871c:	4a070063          	beqz	a4,18bbc <__gdtoa+0x157c>
   18720:	00c12603          	lw	a2,12(sp)
   18724:	00200793          	li	a5,2
   18728:	01062683          	lw	a3,16(a2)
   1872c:	4cf70a63          	beq	a4,a5,18c00 <__gdtoa+0x15c0>
   18730:	00100793          	li	a5,1
   18734:	28d7ce63          	blt	a5,a3,189d0 <__gdtoa+0x1390>
   18738:	01462783          	lw	a5,20(a2)
   1873c:	28079a63          	bnez	a5,189d0 <__gdtoa+0x1390>
   18740:	00f037b3          	snez	a5,a5
   18744:	00479c13          	slli	s8,a5,0x4
   18748:	03000693          	li	a3,48
   1874c:	fffdc783          	lbu	a5,-1(s11)
   18750:	000d8b13          	mv	s6,s11
   18754:	fffd8d93          	addi	s11,s11,-1
   18758:	fed78ae3          	beq	a5,a3,1874c <__gdtoa+0x110c>
   1875c:	e5dff06f          	j	185b8 <__gdtoa+0xf78>
   18760:	01c12783          	lw	a5,28(sp)
   18764:	02c12703          	lw	a4,44(sp)
   18768:	fff78693          	addi	a3,a5,-1
   1876c:	1ad74663          	blt	a4,a3,18918 <__gdtoa+0x12d8>
   18770:	40d704b3          	sub	s1,a4,a3
   18774:	0007dee3          	bgez	a5,18f90 <__gdtoa+0x1950>
   18778:	02412783          	lw	a5,36(sp)
   1877c:	01c12703          	lw	a4,28(sp)
   18780:	06012e23          	sw	zero,124(sp)
   18784:	40e78ab3          	sub	s5,a5,a4
   18788:	b14ff06f          	j	17a9c <__gdtoa+0x45c>
   1878c:	00200793          	li	a5,2
   18790:	bfa7d863          	bge	a5,s10,17b80 <__gdtoa+0x540>
   18794:	00012503          	lw	a0,0(sp)
   18798:	00030593          	mv	a1,t1
   1879c:	00000693          	li	a3,0
   187a0:	00500613          	li	a2,5
   187a4:	01e12823          	sw	t5,16(sp)
   187a8:	0b8010ef          	jal	19860 <__multadd>
   187ac:	00050593          	mv	a1,a0
   187b0:	f0050463          	beqz	a0,17eb8 <__gdtoa+0x878>
   187b4:	01c12783          	lw	a5,28(sp)
   187b8:	01012f03          	lw	t5,16(sp)
   187bc:	14079463          	bnez	a5,18904 <__gdtoa+0x12c4>
   187c0:	00a12823          	sw	a0,16(sp)
   187c4:	00c12503          	lw	a0,12(sp)
   187c8:	01e12c23          	sw	t5,24(sp)
   187cc:	235010ef          	jal	1a200 <__mcmp>
   187d0:	01012583          	lw	a1,16(sp)
   187d4:	01812f03          	lw	t5,24(sp)
   187d8:	12a05663          	blez	a0,18904 <__gdtoa+0x12c4>
   187dc:	03012783          	lw	a5,48(sp)
   187e0:	00278493          	addi	s1,a5,2
   187e4:	03100793          	li	a5,49
   187e8:	001f0b13          	addi	s6,t5,1
   187ec:	00ff0023          	sb	a5,0(t5)
   187f0:	02000c13          	li	s8,32
   187f4:	00012503          	lw	a0,0(sp)
   187f8:	01e12823          	sw	t5,16(sp)
   187fc:	040010ef          	jal	1983c <_Bfree>
   18800:	01012f03          	lw	t5,16(sp)
   18804:	c80a0ee3          	beqz	s4,184a0 <__gdtoa+0xe60>
   18808:	dedff06f          	j	185f4 <__gdtoa+0xfb4>
   1880c:	00100693          	li	a3,1
   18810:	00100593          	li	a1,1
   18814:	9d4ff06f          	j	179e8 <__gdtoa+0x3a8>
   18818:	00050593          	mv	a1,a0
   1881c:	00012503          	lw	a0,0(sp)
   18820:	00078613          	mv	a2,a5
   18824:	6fc010ef          	jal	19f20 <__pow5mult>
   18828:	00050313          	mv	t1,a0
   1882c:	e8050663          	beqz	a0,17eb8 <__gdtoa+0x878>
   18830:	00100693          	li	a3,1
   18834:	02012f03          	lw	t5,32(sp)
   18838:	21a6de63          	bge	a3,s10,18a54 <__gdtoa+0x1414>
   1883c:	01032783          	lw	a5,16(t1)
   18840:	03e12023          	sw	t5,32(sp)
   18844:	00612823          	sw	t1,16(sp)
   18848:	00378793          	addi	a5,a5,3
   1884c:	00279793          	slli	a5,a5,0x2
   18850:	00f307b3          	add	a5,t1,a5
   18854:	0047a503          	lw	a0,4(a5)
   18858:	28c010ef          	jal	19ae4 <__hi0bits>
   1885c:	02012f03          	lw	t5,32(sp)
   18860:	01012303          	lw	t1,16(sp)
   18864:	00050b13          	mv	s6,a0
   18868:	01812c23          	sw	s8,24(sp)
   1886c:	adcff06f          	j	17b48 <__gdtoa+0x508>
   18870:	00100693          	li	a3,1
   18874:	efa6c6e3          	blt	a3,s10,18760 <__gdtoa+0x1120>
   18878:	02412783          	lw	a5,36(sp)
   1887c:	02c12483          	lw	s1,44(sp)
   18880:	00cb8bb3          	add	s7,s7,a2
   18884:	00078a93          	mv	s5,a5
   18888:	00f607b3          	add	a5,a2,a5
   1888c:	02f12223          	sw	a5,36(sp)
   18890:	a0cff06f          	j	17a9c <__gdtoa+0x45c>
   18894:	00200793          	li	a5,2
   18898:	13a7c0e3          	blt	a5,s10,191b8 <__gdtoa+0x1b78>
   1889c:	04c12783          	lw	a5,76(sp)
   188a0:	00f12e23          	sw	a5,28(sp)
   188a4:	dfdff06f          	j	186a0 <__gdtoa+0x1060>
   188a8:	01012c03          	lw	s8,16(sp)
   188ac:	00040f13          	mv	t5,s0
   188b0:	bf1ff06f          	j	184a0 <__gdtoa+0xe60>
   188b4:	01c12783          	lw	a5,28(sp)
   188b8:	a8f04ce3          	bgtz	a5,18350 <__gdtoa+0xd10>
   188bc:	04079063          	bnez	a5,188fc <__gdtoa+0x12bc>
   188c0:	0000a797          	auipc	a5,0xa
   188c4:	03878793          	addi	a5,a5,56 # 228f8 <__SDATA_BEGIN__+0x50>
   188c8:	0007a603          	lw	a2,0(a5)
   188cc:	0047a683          	lw	a3,4(a5)
   188d0:	000d0513          	mv	a0,s10
   188d4:	000d8593          	mv	a1,s11
   188d8:	01e12823          	sw	t5,16(sp)
   188dc:	474040ef          	jal	1cd50 <__muldf3>
   188e0:	04812883          	lw	a7,72(sp)
   188e4:	04012783          	lw	a5,64(sp)
   188e8:	00088613          	mv	a2,a7
   188ec:	00078693          	mv	a3,a5
   188f0:	2a8040ef          	jal	1cb98 <__gedf2>
   188f4:	01012f03          	lw	t5,16(sp)
   188f8:	6e054c63          	bltz	a0,18ff0 <__gdtoa+0x19b0>
   188fc:	00000593          	li	a1,0
   18900:	00000a13          	li	s4,0
   18904:	01412783          	lw	a5,20(sp)
   18908:	000f0b13          	mv	s6,t5
   1890c:	01000c13          	li	s8,16
   18910:	40f004b3          	neg	s1,a5
   18914:	ee1ff06f          	j	187f4 <__gdtoa+0x11b4>
   18918:	02c12783          	lw	a5,44(sp)
   1891c:	02412703          	lw	a4,36(sp)
   18920:	00000493          	li	s1,0
   18924:	40f68633          	sub	a2,a3,a5
   18928:	01812783          	lw	a5,24(sp)
   1892c:	00070a93          	mv	s5,a4
   18930:	02d12623          	sw	a3,44(sp)
   18934:	00c787b3          	add	a5,a5,a2
   18938:	00f12c23          	sw	a5,24(sp)
   1893c:	01c12783          	lw	a5,28(sp)
   18940:	06f12e23          	sw	a5,124(sp)
   18944:	00fb8bb3          	add	s7,s7,a5
   18948:	00f707b3          	add	a5,a4,a5
   1894c:	02f12223          	sw	a5,36(sp)
   18950:	94cff06f          	j	17a9c <__gdtoa+0x45c>
   18954:	00012b03          	lw	s6,0(sp)
   18958:	000a0593          	mv	a1,s4
   1895c:	00048613          	mv	a2,s1
   18960:	000b0513          	mv	a0,s6
   18964:	03e12023          	sw	t5,32(sp)
   18968:	5b8010ef          	jal	19f20 <__pow5mult>
   1896c:	00050a13          	mv	s4,a0
   18970:	d4050463          	beqz	a0,17eb8 <__gdtoa+0x878>
   18974:	00c12603          	lw	a2,12(sp)
   18978:	00050593          	mv	a1,a0
   1897c:	000b0513          	mv	a0,s6
   18980:	36c010ef          	jal	19cec <__multiply>
   18984:	00050413          	mv	s0,a0
   18988:	d2050863          	beqz	a0,17eb8 <__gdtoa+0x878>
   1898c:	00c12583          	lw	a1,12(sp)
   18990:	000b0513          	mv	a0,s6
   18994:	6a9000ef          	jal	1983c <_Bfree>
   18998:	02c12783          	lw	a5,44(sp)
   1899c:	00812623          	sw	s0,12(sp)
   189a0:	02012f03          	lw	t5,32(sp)
   189a4:	409787b3          	sub	a5,a5,s1
   189a8:	02f12623          	sw	a5,44(sp)
   189ac:	96078663          	beqz	a5,17b18 <__gdtoa+0x4d8>
   189b0:	948ff06f          	j	17af8 <__gdtoa+0x4b8>
   189b4:	000a0413          	mv	s0,s4
   189b8:	000d8793          	mv	a5,s11
   189bc:	001d8993          	addi	s3,s11,1
   189c0:	000a8a13          	mv	s4,s5
   189c4:	03900693          	li	a3,57
   189c8:	00098d93          	mv	s11,s3
   189cc:	00d78023          	sb	a3,0(a5)
   189d0:	03900693          	li	a3,57
   189d4:	0080006f          	j	189dc <__gdtoa+0x139c>
   189d8:	25bf0463          	beq	t5,s11,18c20 <__gdtoa+0x15e0>
   189dc:	fffdc783          	lbu	a5,-1(s11)
   189e0:	000d8b13          	mv	s6,s11
   189e4:	fffd8d93          	addi	s11,s11,-1
   189e8:	fed788e3          	beq	a5,a3,189d8 <__gdtoa+0x1398>
   189ec:	00178793          	addi	a5,a5,1
   189f0:	00fd8023          	sb	a5,0(s11)
   189f4:	02000c13          	li	s8,32
   189f8:	bc1ff06f          	j	185b8 <__gdtoa+0xf78>
   189fc:	00012503          	lw	a0,0(sp)
   18a00:	000a0593          	mv	a1,s4
   18a04:	01e12a23          	sw	t5,20(sp)
   18a08:	00612823          	sw	t1,16(sp)
   18a0c:	664010ef          	jal	1a070 <__lshift>
   18a10:	01012303          	lw	t1,16(sp)
   18a14:	01412f03          	lw	t5,20(sp)
   18a18:	00050a13          	mv	s4,a0
   18a1c:	96051a63          	bnez	a0,17b90 <__gdtoa+0x550>
   18a20:	c98ff06f          	j	17eb8 <__gdtoa+0x878>
   18a24:	00100793          	li	a5,1
   18a28:	00000a13          	li	s4,0
   18a2c:	02f12a23          	sw	a5,52(sp)
   18a30:	f95fe06f          	j	179c4 <__gdtoa+0x384>
   18a34:	00100793          	li	a5,1
   18a38:	02f12a23          	sw	a5,52(sp)
   18a3c:	f89fe06f          	j	179c4 <__gdtoa+0x384>
   18a40:	00100793          	li	a5,1
   18a44:	02f12a23          	sw	a5,52(sp)
   18a48:	af5ff06f          	j	1853c <__gdtoa+0xefc>
   18a4c:	02012a23          	sw	zero,52(sp)
   18a50:	aedff06f          	j	1853c <__gdtoa+0xefc>
   18a54:	dedc94e3          	bne	s9,a3,1883c <__gdtoa+0x11fc>
   18a58:	004da783          	lw	a5,4(s11)
   18a5c:	01012703          	lw	a4,16(sp)
   18a60:	00178793          	addi	a5,a5,1
   18a64:	dce7dce3          	bge	a5,a4,1883c <__gdtoa+0x11fc>
   18a68:	02412783          	lw	a5,36(sp)
   18a6c:	001b8b93          	addi	s7,s7,1
   18a70:	00100c13          	li	s8,1
   18a74:	00178793          	addi	a5,a5,1
   18a78:	02f12223          	sw	a5,36(sp)
   18a7c:	dc1ff06f          	j	1883c <__gdtoa+0x11fc>
   18a80:	00012403          	lw	s0,0(sp)
   18a84:	004a2583          	lw	a1,4(s4)
   18a88:	01e12a23          	sw	t5,20(sp)
   18a8c:	00040513          	mv	a0,s0
   18a90:	00612823          	sw	t1,16(sp)
   18a94:	4f5000ef          	jal	19788 <_Balloc>
   18a98:	00050a93          	mv	s5,a0
   18a9c:	c0050e63          	beqz	a0,17eb8 <__gdtoa+0x878>
   18aa0:	010a2603          	lw	a2,16(s4)
   18aa4:	00ca0593          	addi	a1,s4,12
   18aa8:	00c50513          	addi	a0,a0,12
   18aac:	00260613          	addi	a2,a2,2
   18ab0:	00261613          	slli	a2,a2,0x2
   18ab4:	a3cfe0ef          	jal	16cf0 <memcpy>
   18ab8:	000a8593          	mv	a1,s5
   18abc:	00100613          	li	a2,1
   18ac0:	00040513          	mv	a0,s0
   18ac4:	5ac010ef          	jal	1a070 <__lshift>
   18ac8:	01012303          	lw	t1,16(sp)
   18acc:	01412f03          	lw	t5,20(sp)
   18ad0:	00050a93          	mv	s5,a0
   18ad4:	00050463          	beqz	a0,18adc <__gdtoa+0x149c>
   18ad8:	8c8ff06f          	j	17ba0 <__gdtoa+0x560>
   18adc:	bdcff06f          	j	17eb8 <__gdtoa+0x878>
   18ae0:	000a0693          	mv	a3,s4
   18ae4:	00040613          	mv	a2,s0
   18ae8:	000a0593          	mv	a1,s4
   18aec:	00040513          	mv	a0,s0
   18af0:	05e12023          	sw	t5,64(sp)
   18af4:	1a4030ef          	jal	1bc98 <__adddf3>
   18af8:	0000a697          	auipc	a3,0xa
   18afc:	df868693          	addi	a3,a3,-520 # 228f0 <__SDATA_BEGIN__+0x48>
   18b00:	0006a603          	lw	a2,0(a3)
   18b04:	0046a683          	lw	a3,4(a3)
   18b08:	190030ef          	jal	1bc98 <__adddf3>
   18b0c:	04012f03          	lw	t5,64(sp)
   18b10:	fcc00737          	lui	a4,0xfcc00
   18b14:	00050b13          	mv	s6,a0
   18b18:	00b70a33          	add	s4,a4,a1
   18b1c:	0000a697          	auipc	a3,0xa
   18b20:	ddc68693          	addi	a3,a3,-548 # 228f8 <__SDATA_BEGIN__+0x50>
   18b24:	0006a603          	lw	a2,0(a3)
   18b28:	02812503          	lw	a0,40(sp)
   18b2c:	0046a683          	lw	a3,4(a3)
   18b30:	03812583          	lw	a1,56(sp)
   18b34:	05e12023          	sw	t5,64(sp)
   18b38:	009040ef          	jal	1d340 <__subdf3>
   18b3c:	000b0613          	mv	a2,s6
   18b40:	000a0693          	mv	a3,s4
   18b44:	00050493          	mv	s1,a0
   18b48:	00058413          	mv	s0,a1
   18b4c:	04c040ef          	jal	1cb98 <__gedf2>
   18b50:	04012f03          	lw	t5,64(sp)
   18b54:	44a04c63          	bgtz	a0,18fac <__gdtoa+0x196c>
   18b58:	800008b7          	lui	a7,0x80000
   18b5c:	0148c8b3          	xor	a7,a7,s4
   18b60:	000b0613          	mv	a2,s6
   18b64:	00048513          	mv	a0,s1
   18b68:	00088693          	mv	a3,a7
   18b6c:	00040593          	mv	a1,s0
   18b70:	104040ef          	jal	1cc74 <__ledf2>
   18b74:	04012f03          	lw	t5,64(sp)
   18b78:	d80542e3          	bltz	a0,188fc <__gdtoa+0x12bc>
   18b7c:	02812783          	lw	a5,40(sp)
   18b80:	00008317          	auipc	t1,0x8
   18b84:	59030313          	addi	t1,t1,1424 # 21110 <__mprec_tens>
   18b88:	04f12423          	sw	a5,72(sp)
   18b8c:	03812783          	lw	a5,56(sp)
   18b90:	04f12023          	sw	a5,64(sp)
   18b94:	02012783          	lw	a5,32(sp)
   18b98:	5207c063          	bltz	a5,190b8 <__gdtoa+0x1a78>
   18b9c:	01412783          	lw	a5,20(sp)
   18ba0:	02012423          	sw	zero,40(sp)
   18ba4:	00032d03          	lw	s10,0(t1)
   18ba8:	00432d83          	lw	s11,4(t1)
   18bac:	fa07d263          	bgez	a5,18350 <__gdtoa+0xd10>
   18bb0:	01c12783          	lw	a5,28(sp)
   18bb4:	f8079e63          	bnez	a5,18350 <__gdtoa+0xd10>
   18bb8:	d09ff06f          	j	188c0 <__gdtoa+0x1280>
   18bbc:	00c12583          	lw	a1,12(sp)
   18bc0:	00012503          	lw	a0,0(sp)
   18bc4:	00100613          	li	a2,1
   18bc8:	01e12a23          	sw	t5,20(sp)
   18bcc:	00612823          	sw	t1,16(sp)
   18bd0:	4a0010ef          	jal	1a070 <__lshift>
   18bd4:	00a12623          	sw	a0,12(sp)
   18bd8:	ae050063          	beqz	a0,17eb8 <__gdtoa+0x878>
   18bdc:	01012303          	lw	t1,16(sp)
   18be0:	00030593          	mv	a1,t1
   18be4:	61c010ef          	jal	1a200 <__mcmp>
   18be8:	01012303          	lw	t1,16(sp)
   18bec:	01412f03          	lw	t5,20(sp)
   18bf0:	dea040e3          	bgtz	a0,189d0 <__gdtoa+0x1390>
   18bf4:	00051663          	bnez	a0,18c00 <__gdtoa+0x15c0>
   18bf8:	001cf793          	andi	a5,s9,1
   18bfc:	dc079ae3          	bnez	a5,189d0 <__gdtoa+0x1390>
   18c00:	00c12783          	lw	a5,12(sp)
   18c04:	01000c13          	li	s8,16
   18c08:	0107a683          	lw	a3,16(a5)
   18c0c:	00100793          	li	a5,1
   18c10:	b2d7cce3          	blt	a5,a3,18748 <__gdtoa+0x1108>
   18c14:	00c12783          	lw	a5,12(sp)
   18c18:	0147a783          	lw	a5,20(a5)
   18c1c:	b25ff06f          	j	18740 <__gdtoa+0x1100>
   18c20:	03012783          	lw	a5,48(sp)
   18c24:	02000c13          	li	s8,32
   18c28:	00178793          	addi	a5,a5,1
   18c2c:	02f12823          	sw	a5,48(sp)
   18c30:	03100793          	li	a5,49
   18c34:	00ff0023          	sb	a5,0(t5)
   18c38:	981ff06f          	j	185b8 <__gdtoa+0xf78>
   18c3c:	02812783          	lw	a5,40(sp)
   18c40:	00912623          	sw	s1,12(sp)
   18c44:	01012f03          	lw	t5,16(sp)
   18c48:	01412903          	lw	s2,20(sp)
   18c4c:	000c0313          	mv	t1,s8
   18c50:	02078263          	beqz	a5,18c74 <__gdtoa+0x1634>
   18c54:	00c12783          	lw	a5,12(sp)
   18c58:	00100693          	li	a3,1
   18c5c:	0107a603          	lw	a2,16(a5)
   18c60:	00c6d463          	bge	a3,a2,18c68 <__gdtoa+0x1628>
   18c64:	8c8ff06f          	j	17d2c <__gdtoa+0x6ec>
   18c68:	0147a683          	lw	a3,20(a5)
   18c6c:	00068463          	beqz	a3,18c74 <__gdtoa+0x1634>
   18c70:	8bcff06f          	j	17d2c <__gdtoa+0x6ec>
   18c74:	00c12583          	lw	a1,12(sp)
   18c78:	00012503          	lw	a0,0(sp)
   18c7c:	00100613          	li	a2,1
   18c80:	01e12a23          	sw	t5,20(sp)
   18c84:	00612823          	sw	t1,16(sp)
   18c88:	3e8010ef          	jal	1a070 <__lshift>
   18c8c:	00a12623          	sw	a0,12(sp)
   18c90:	a2050463          	beqz	a0,17eb8 <__gdtoa+0x878>
   18c94:	01012303          	lw	t1,16(sp)
   18c98:	00030593          	mv	a1,t1
   18c9c:	564010ef          	jal	1a200 <__mcmp>
   18ca0:	01012303          	lw	t1,16(sp)
   18ca4:	01412f03          	lw	t5,20(sp)
   18ca8:	4ea05c63          	blez	a0,191a0 <__gdtoa+0x1b60>
   18cac:	03900693          	li	a3,57
   18cb0:	d0dc82e3          	beq	s9,a3,189b4 <__gdtoa+0x1374>
   18cb4:	02000793          	li	a5,32
   18cb8:	031b8c93          	addi	s9,s7,49
   18cbc:	02f12423          	sw	a5,40(sp)
   18cc0:	8d1ff06f          	j	18590 <__gdtoa+0xf50>
   18cc4:	000a8613          	mv	a2,s5
   18cc8:	000c0693          	mv	a3,s8
   18ccc:	01e12a23          	sw	t5,20(sp)
   18cd0:	7c9020ef          	jal	1bc98 <__adddf3>
   18cd4:	fffb4703          	lbu	a4,-1(s6)
   18cd8:	000d0613          	mv	a2,s10
   18cdc:	000d8693          	mv	a3,s11
   18ce0:	00e12823          	sw	a4,16(sp)
   18ce4:	00050a93          	mv	s5,a0
   18ce8:	00058a13          	mv	s4,a1
   18cec:	6ad030ef          	jal	1cb98 <__gedf2>
   18cf0:	01012703          	lw	a4,16(sp)
   18cf4:	01412f03          	lw	t5,20(sp)
   18cf8:	d8a04063          	bgtz	a0,18278 <__gdtoa+0xc38>
   18cfc:	000a8513          	mv	a0,s5
   18d00:	000a0593          	mv	a1,s4
   18d04:	000d0613          	mv	a2,s10
   18d08:	000d8693          	mv	a3,s11
   18d0c:	601030ef          	jal	1cb0c <__eqdf2>
   18d10:	01412f03          	lw	t5,20(sp)
   18d14:	2e051863          	bnez	a0,19004 <__gdtoa+0x19c4>
   18d18:	0019f993          	andi	s3,s3,1
   18d1c:	2e098463          	beqz	s3,19004 <__gdtoa+0x19c4>
   18d20:	01012703          	lw	a4,16(sp)
   18d24:	d54ff06f          	j	18278 <__gdtoa+0xc38>
   18d28:	03100693          	li	a3,49
   18d2c:	00140413          	addi	s0,s0,1
   18d30:	d6cff06f          	j	1829c <__gdtoa+0xc5c>
   18d34:	000b0513          	mv	a0,s6
   18d38:	000a0593          	mv	a1,s4
   18d3c:	06612623          	sw	t1,108(sp)
   18d40:	07e12023          	sw	t5,96(sp)
   18d44:	00c040ef          	jal	1cd50 <__muldf3>
   18d48:	04812603          	lw	a2,72(sp)
   18d4c:	04012683          	lw	a3,64(sp)
   18d50:	06012f03          	lw	t5,96(sp)
   18d54:	06c12303          	lw	t1,108(sp)
   18d58:	00060a93          	mv	s5,a2
   18d5c:	04b12e23          	sw	a1,92(sp)
   18d60:	04c12023          	sw	a2,64(sp)
   18d64:	00100593          	li	a1,1
   18d68:	00000613          	li	a2,0
   18d6c:	07312223          	sw	s3,100(sp)
   18d70:	07212423          	sw	s2,104(sp)
   18d74:	04a12c23          	sw	a0,88(sp)
   18d78:	05012903          	lw	s2,80(sp)
   18d7c:	06b12e23          	sw	a1,124(sp)
   18d80:	04d12423          	sw	a3,72(sp)
   18d84:	00068a13          	mv	s4,a3
   18d88:	000f0b13          	mv	s6,t5
   18d8c:	0000a497          	auipc	s1,0xa
   18d90:	b5448493          	addi	s1,s1,-1196 # 228e0 <__SDATA_BEGIN__+0x38>
   18d94:	00060413          	mv	s0,a2
   18d98:	04612823          	sw	t1,80(sp)
   18d9c:	000a8993          	mv	s3,s5
   18da0:	0200006f          	j	18dc0 <__gdtoa+0x1780>
   18da4:	0004a603          	lw	a2,0(s1)
   18da8:	0044a683          	lw	a3,4(s1)
   18dac:	06612e23          	sw	t1,124(sp)
   18db0:	00100413          	li	s0,1
   18db4:	79d030ef          	jal	1cd50 <__muldf3>
   18db8:	00050993          	mv	s3,a0
   18dbc:	00058a13          	mv	s4,a1
   18dc0:	00098513          	mv	a0,s3
   18dc4:	000a0593          	mv	a1,s4
   18dc8:	521040ef          	jal	1dae8 <__fixdfsi>
   18dcc:	00050a93          	mv	s5,a0
   18dd0:	02050463          	beqz	a0,18df8 <__gdtoa+0x17b8>
   18dd4:	595040ef          	jal	1db68 <__floatsidf>
   18dd8:	00050613          	mv	a2,a0
   18ddc:	00058693          	mv	a3,a1
   18de0:	00098513          	mv	a0,s3
   18de4:	000a0593          	mv	a1,s4
   18de8:	558040ef          	jal	1d340 <__subdf3>
   18dec:	00050993          	mv	s3,a0
   18df0:	00058a13          	mv	s4,a1
   18df4:	00100413          	li	s0,1
   18df8:	030a8793          	addi	a5,s5,48
   18dfc:	0ff7f713          	zext.b	a4,a5
   18e00:	00eb0023          	sb	a4,0(s6)
   18e04:	07c12783          	lw	a5,124(sp)
   18e08:	001b0b13          	addi	s6,s6,1
   18e0c:	00098513          	mv	a0,s3
   18e10:	000a0593          	mv	a1,s4
   18e14:	00178313          	addi	t1,a5,1
   18e18:	f92796e3          	bne	a5,s2,18da4 <__gdtoa+0x1764>
   18e1c:	06012f03          	lw	t5,96(sp)
   18e20:	06812903          	lw	s2,104(sp)
   18e24:	05012303          	lw	t1,80(sp)
   18e28:	06412983          	lw	s3,100(sp)
   18e2c:	00040663          	beqz	s0,18e38 <__gdtoa+0x17f8>
   18e30:	04a12023          	sw	a0,64(sp)
   18e34:	05412423          	sw	s4,72(sp)
   18e38:	0000aa17          	auipc	s4,0xa
   18e3c:	ac8a0a13          	addi	s4,s4,-1336 # 22900 <__SDATA_BEGIN__+0x58>
   18e40:	05c12a83          	lw	s5,92(sp)
   18e44:	000a2603          	lw	a2,0(s4)
   18e48:	004a2683          	lw	a3,4(s4)
   18e4c:	05812503          	lw	a0,88(sp)
   18e50:	000a8593          	mv	a1,s5
   18e54:	07e12023          	sw	t5,96(sp)
   18e58:	04e12823          	sw	a4,80(sp)
   18e5c:	06612223          	sw	t1,100(sp)
   18e60:	639020ef          	jal	1bc98 <__adddf3>
   18e64:	04012483          	lw	s1,64(sp)
   18e68:	04812403          	lw	s0,72(sp)
   18e6c:	00048613          	mv	a2,s1
   18e70:	00040693          	mv	a3,s0
   18e74:	601030ef          	jal	1cc74 <__ledf2>
   18e78:	05012703          	lw	a4,80(sp)
   18e7c:	06012f03          	lw	t5,96(sp)
   18e80:	22054663          	bltz	a0,190ac <__gdtoa+0x1a6c>
   18e84:	05812603          	lw	a2,88(sp)
   18e88:	000a2503          	lw	a0,0(s4)
   18e8c:	004a2583          	lw	a1,4(s4)
   18e90:	000a8693          	mv	a3,s5
   18e94:	05e12823          	sw	t5,80(sp)
   18e98:	4a8040ef          	jal	1d340 <__subdf3>
   18e9c:	00048613          	mv	a2,s1
   18ea0:	00040693          	mv	a3,s0
   18ea4:	4f5030ef          	jal	1cb98 <__gedf2>
   18ea8:	05012f03          	lw	t5,80(sp)
   18eac:	06412303          	lw	t1,100(sp)
   18eb0:	2aa04663          	bgtz	a0,1915c <__gdtoa+0x1b1c>
   18eb4:	02812783          	lw	a5,40(sp)
   18eb8:	04f12423          	sw	a5,72(sp)
   18ebc:	03812783          	lw	a5,56(sp)
   18ec0:	04f12023          	sw	a5,64(sp)
   18ec4:	02012783          	lw	a5,32(sp)
   18ec8:	cc07dae3          	bgez	a5,18b9c <__gdtoa+0x155c>
   18ecc:	02c12483          	lw	s1,44(sp)
   18ed0:	02412a83          	lw	s5,36(sp)
   18ed4:	02012423          	sw	zero,40(sp)
   18ed8:	00000a13          	li	s4,0
   18edc:	bddfe06f          	j	17ab8 <__gdtoa+0x478>
   18ee0:	000a0413          	mv	s0,s4
   18ee4:	01012f03          	lw	t5,16(sp)
   18ee8:	01412903          	lw	s2,20(sp)
   18eec:	00912623          	sw	s1,12(sp)
   18ef0:	000c0313          	mv	t1,s8
   18ef4:	000a8a13          	mv	s4,s5
   18ef8:	821ff06f          	j	18718 <__gdtoa+0x10d8>
   18efc:	fffb4703          	lbu	a4,-1(s6)
   18f00:	b78ff06f          	j	18278 <__gdtoa+0xc38>
   18f04:	bc048ee3          	beqz	s1,18ae0 <__gdtoa+0x14a0>
   18f08:	04c12483          	lw	s1,76(sp)
   18f0c:	c69058e3          	blez	s1,18b7c <__gdtoa+0x153c>
   18f10:	0000a697          	auipc	a3,0xa
   18f14:	9d068693          	addi	a3,a3,-1584 # 228e0 <__SDATA_BEGIN__+0x38>
   18f18:	0006a603          	lw	a2,0(a3)
   18f1c:	0046a683          	lw	a3,4(a3)
   18f20:	000a0593          	mv	a1,s4
   18f24:	00040513          	mv	a0,s0
   18f28:	07e12023          	sw	t5,96(sp)
   18f2c:	625030ef          	jal	1cd50 <__muldf3>
   18f30:	0000a697          	auipc	a3,0xa
   18f34:	9b868693          	addi	a3,a3,-1608 # 228e8 <__SDATA_BEGIN__+0x40>
   18f38:	0006a603          	lw	a2,0(a3)
   18f3c:	0046a683          	lw	a3,4(a3)
   18f40:	04a12423          	sw	a0,72(sp)
   18f44:	04a12e23          	sw	a0,92(sp)
   18f48:	04b12023          	sw	a1,64(sp)
   18f4c:	04b12c23          	sw	a1,88(sp)
   18f50:	601030ef          	jal	1cd50 <__muldf3>
   18f54:	0000a697          	auipc	a3,0xa
   18f58:	99c68693          	addi	a3,a3,-1636 # 228f0 <__SDATA_BEGIN__+0x48>
   18f5c:	0006a603          	lw	a2,0(a3)
   18f60:	0046a683          	lw	a3,4(a3)
   18f64:	535020ef          	jal	1bc98 <__adddf3>
   18f68:	fcc00737          	lui	a4,0xfcc00
   18f6c:	fff00793          	li	a5,-1
   18f70:	05812803          	lw	a6,88(sp)
   18f74:	05c12e83          	lw	t4,92(sp)
   18f78:	06012f03          	lw	t5,96(sp)
   18f7c:	00050b13          	mv	s6,a0
   18f80:	00b70a33          	add	s4,a4,a1
   18f84:	04912823          	sw	s1,80(sp)
   18f88:	04f12a23          	sw	a5,84(sp)
   18f8c:	8f0ff06f          	j	1807c <__gdtoa+0xa3c>
   18f90:	02412703          	lw	a4,36(sp)
   18f94:	06f12e23          	sw	a5,124(sp)
   18f98:	00fb8bb3          	add	s7,s7,a5
   18f9c:	00f707b3          	add	a5,a4,a5
   18fa0:	00070a93          	mv	s5,a4
   18fa4:	02f12223          	sw	a5,36(sp)
   18fa8:	af5fe06f          	j	17a9c <__gdtoa+0x45c>
   18fac:	00200493          	li	s1,2
   18fb0:	00000593          	li	a1,0
   18fb4:	00000a13          	li	s4,0
   18fb8:	82dff06f          	j	187e4 <__gdtoa+0x11a4>
   18fbc:	00912623          	sw	s1,12(sp)
   18fc0:	03900693          	li	a3,57
   18fc4:	01012f03          	lw	t5,16(sp)
   18fc8:	01412903          	lw	s2,20(sp)
   18fcc:	000c0313          	mv	t1,s8
   18fd0:	9edc82e3          	beq	s9,a3,189b4 <__gdtoa+0x1374>
   18fd4:	001c8793          	addi	a5,s9,1
   18fd8:	000a0413          	mv	s0,s4
   18fdc:	001d8b13          	addi	s6,s11,1
   18fe0:	00fd8023          	sb	a5,0(s11)
   18fe4:	000a8a13          	mv	s4,s5
   18fe8:	02000c13          	li	s8,32
   18fec:	dccff06f          	j	185b8 <__gdtoa+0xf78>
   18ff0:	03012783          	lw	a5,48(sp)
   18ff4:	00000593          	li	a1,0
   18ff8:	00000a13          	li	s4,0
   18ffc:	00278493          	addi	s1,a5,2
   19000:	fe4ff06f          	j	187e4 <__gdtoa+0x11a4>
   19004:	000b0793          	mv	a5,s6
   19008:	01000c13          	li	s8,16
   1900c:	03000613          	li	a2,48
   19010:	fff7c703          	lbu	a4,-1(a5)
   19014:	00078b13          	mv	s6,a5
   19018:	fff78793          	addi	a5,a5,-1
   1901c:	fec70ae3          	beq	a4,a2,19010 <__gdtoa+0x19d0>
   19020:	00040493          	mv	s1,s0
   19024:	c7cff06f          	j	184a0 <__gdtoa+0xe60>
   19028:	00012503          	lw	a0,0(sp)
   1902c:	000a0593          	mv	a1,s4
   19030:	00000693          	li	a3,0
   19034:	00a00613          	li	a2,10
   19038:	029000ef          	jal	19860 <__multadd>
   1903c:	00050a13          	mv	s4,a0
   19040:	00051463          	bnez	a0,19048 <__gdtoa+0x1a08>
   19044:	e75fe06f          	j	17eb8 <__gdtoa+0x878>
   19048:	04c12783          	lw	a5,76(sp)
   1904c:	01012303          	lw	t1,16(sp)
   19050:	01c12f03          	lw	t5,28(sp)
   19054:	00f05663          	blez	a5,19060 <__gdtoa+0x1a20>
   19058:	00f12e23          	sw	a5,28(sp)
   1905c:	b2dfe06f          	j	17b88 <__gdtoa+0x548>
   19060:	00200793          	li	a5,2
   19064:	15a7ca63          	blt	a5,s10,191b8 <__gdtoa+0x1b78>
   19068:	04c12783          	lw	a5,76(sp)
   1906c:	00f12e23          	sw	a5,28(sp)
   19070:	b19fe06f          	j	17b88 <__gdtoa+0x548>
   19074:	04812f03          	lw	t5,72(sp)
   19078:	05012903          	lw	s2,80(sp)
   1907c:	00048513          	mv	a0,s1
   19080:	000a0593          	mv	a1,s4
   19084:	00000613          	li	a2,0
   19088:	00000693          	li	a3,0
   1908c:	01e12823          	sw	t5,16(sp)
   19090:	27d030ef          	jal	1cb0c <__eqdf2>
   19094:	00a037b3          	snez	a5,a0
   19098:	00479c13          	slli	s8,a5,0x4
   1909c:	05412783          	lw	a5,84(sp)
   190a0:	01012f03          	lw	t5,16(sp)
   190a4:	00178493          	addi	s1,a5,1
   190a8:	bf8ff06f          	j	184a0 <__gdtoa+0xe60>
   190ac:	05412783          	lw	a5,84(sp)
   190b0:	00178413          	addi	s0,a5,1
   190b4:	9c4ff06f          	j	18278 <__gdtoa+0xc38>
   190b8:	02012423          	sw	zero,40(sp)
   190bc:	971fe06f          	j	17a2c <__gdtoa+0x3ec>
   190c0:	01c12783          	lw	a5,28(sp)
   190c4:	02c12703          	lw	a4,44(sp)
   190c8:	fff78693          	addi	a3,a5,-1
   190cc:	84d746e3          	blt	a4,a3,18918 <__gdtoa+0x12d8>
   190d0:	02412603          	lw	a2,36(sp)
   190d4:	00fb8bb3          	add	s7,s7,a5
   190d8:	06f12e23          	sw	a5,124(sp)
   190dc:	00f607b3          	add	a5,a2,a5
   190e0:	00060a93          	mv	s5,a2
   190e4:	40d704b3          	sub	s1,a4,a3
   190e8:	02f12223          	sw	a5,36(sp)
   190ec:	9b1fe06f          	j	17a9c <__gdtoa+0x45c>
   190f0:	02812783          	lw	a5,40(sp)
   190f4:	04812f03          	lw	t5,72(sp)
   190f8:	06012b83          	lw	s7,96(sp)
   190fc:	04f12423          	sw	a5,72(sp)
   19100:	03812783          	lw	a5,56(sp)
   19104:	06412a83          	lw	s5,100(sp)
   19108:	05012903          	lw	s2,80(sp)
   1910c:	04f12023          	sw	a5,64(sp)
   19110:	02012783          	lw	a5,32(sp)
   19114:	06812303          	lw	t1,104(sp)
   19118:	a807d2e3          	bgez	a5,18b9c <__gdtoa+0x155c>
   1911c:	02012423          	sw	zero,40(sp)
   19120:	915fe06f          	j	17a34 <__gdtoa+0x3f4>
   19124:	00912623          	sw	s1,12(sp)
   19128:	03900693          	li	a3,57
   1912c:	01012f03          	lw	t5,16(sp)
   19130:	01412903          	lw	s2,20(sp)
   19134:	000c0313          	mv	t1,s8
   19138:	86dc8ee3          	beq	s9,a3,189b4 <__gdtoa+0x1374>
   1913c:	0f605663          	blez	s6,19228 <__gdtoa+0x1be8>
   19140:	031b8c93          	addi	s9,s7,49
   19144:	02000c13          	li	s8,32
   19148:	000a0413          	mv	s0,s4
   1914c:	001d8b13          	addi	s6,s11,1
   19150:	019d8023          	sb	s9,0(s11)
   19154:	000a8a13          	mv	s4,s5
   19158:	c60ff06f          	j	185b8 <__gdtoa+0xf78>
   1915c:	00040593          	mv	a1,s0
   19160:	00048513          	mv	a0,s1
   19164:	00000613          	li	a2,0
   19168:	00000693          	li	a3,0
   1916c:	01e12823          	sw	t5,16(sp)
   19170:	19d030ef          	jal	1cb0c <__eqdf2>
   19174:	05412703          	lw	a4,84(sp)
   19178:	00a037b3          	snez	a5,a0
   1917c:	00479c13          	slli	s8,a5,0x4
   19180:	01012f03          	lw	t5,16(sp)
   19184:	000b0793          	mv	a5,s6
   19188:	00170413          	addi	s0,a4,1 # fcc00001 <__BSS_END__+0xfcbdd2d1>
   1918c:	e81ff06f          	j	1900c <__gdtoa+0x19cc>
   19190:	0147a683          	lw	a3,20(a5)
   19194:	c0069a63          	bnez	a3,185a8 <__gdtoa+0xf68>
   19198:	02812c03          	lw	s8,40(sp)
   1919c:	c0cff06f          	j	185a8 <__gdtoa+0xf68>
   191a0:	00051663          	bnez	a0,191ac <__gdtoa+0x1b6c>
   191a4:	001cf693          	andi	a3,s9,1
   191a8:	b00692e3          	bnez	a3,18cac <__gdtoa+0x166c>
   191ac:	02000793          	li	a5,32
   191b0:	02f12423          	sw	a5,40(sp)
   191b4:	bdcff06f          	j	18590 <__gdtoa+0xf50>
   191b8:	04c12783          	lw	a5,76(sp)
   191bc:	00f12e23          	sw	a5,28(sp)
   191c0:	dd4ff06f          	j	18794 <__gdtoa+0x1154>
   191c4:	00912623          	sw	s1,12(sp)
   191c8:	03900693          	li	a3,57
   191cc:	000b0313          	mv	t1,s6
   191d0:	000c0f13          	mv	t5,s8
   191d4:	00dc8e63          	beq	s9,a3,191f0 <__gdtoa+0x1bb0>
   191d8:	001c8c93          	addi	s9,s9,1
   191dc:	02000c13          	li	s8,32
   191e0:	bc8ff06f          	j	185a8 <__gdtoa+0xf68>
   191e4:	01000c13          	li	s8,16
   191e8:	001d8993          	addi	s3,s11,1
   191ec:	bbcff06f          	j	185a8 <__gdtoa+0xf68>
   191f0:	000a0413          	mv	s0,s4
   191f4:	000d8793          	mv	a5,s11
   191f8:	000a8a13          	mv	s4,s5
   191fc:	fc8ff06f          	j	189c4 <__gdtoa+0x1384>
   19200:	0104a603          	lw	a2,16(s1)
   19204:	00100693          	li	a3,1
   19208:	00048793          	mv	a5,s1
   1920c:	00c6d463          	bge	a3,a2,19214 <__gdtoa+0x1bd4>
   19210:	b1dfe06f          	j	17d2c <__gdtoa+0x6ec>
   19214:	0147a683          	lw	a3,20(a5)
   19218:	00068463          	beqz	a3,19220 <__gdtoa+0x1be0>
   1921c:	b11fe06f          	j	17d2c <__gdtoa+0x6ec>
   19220:	001d8993          	addi	s3,s11,1
   19224:	b84ff06f          	j	185a8 <__gdtoa+0xf68>
   19228:	00c12783          	lw	a5,12(sp)
   1922c:	00100693          	li	a3,1
   19230:	01000c13          	li	s8,16
   19234:	0107a603          	lw	a2,16(a5)
   19238:	f0c6c8e3          	blt	a3,a2,19148 <__gdtoa+0x1b08>
   1923c:	0147a683          	lw	a3,20(a5)
   19240:	00d036b3          	snez	a3,a3
   19244:	00469c13          	slli	s8,a3,0x4
   19248:	f01ff06f          	j	19148 <__gdtoa+0x1b08>
   1924c:	020b4263          	bltz	s6,19270 <__gdtoa+0x1c30>
   19250:	016d6b33          	or	s6,s10,s6
   19254:	000b1863          	bnez	s6,19264 <__gdtoa+0x1c24>
   19258:	0009a783          	lw	a5,0(s3)
   1925c:	0017f793          	andi	a5,a5,1
   19260:	00078863          	beqz	a5,19270 <__gdtoa+0x1c30>
   19264:	00d04463          	bgtz	a3,1926c <__gdtoa+0x1c2c>
   19268:	991fe06f          	j	17bf8 <__gdtoa+0x5b8>
   1926c:	981fe06f          	j	17bec <__gdtoa+0x5ac>
   19270:	02812703          	lw	a4,40(sp)
   19274:	00912623          	sw	s1,12(sp)
   19278:	01012f03          	lw	t5,16(sp)
   1927c:	01412903          	lw	s2,20(sp)
   19280:	000c0313          	mv	t1,s8
   19284:	02070863          	beqz	a4,192b4 <__gdtoa+0x1c74>
   19288:	0104a583          	lw	a1,16(s1)
   1928c:	00100613          	li	a2,1
   19290:	00b65463          	bge	a2,a1,19298 <__gdtoa+0x1c58>
   19294:	a99fe06f          	j	17d2c <__gdtoa+0x6ec>
   19298:	0144a603          	lw	a2,20(s1)
   1929c:	00060463          	beqz	a2,192a4 <__gdtoa+0x1c64>
   192a0:	a8dfe06f          	j	17d2c <__gdtoa+0x6ec>
   192a4:	9cd048e3          	bgtz	a3,18c74 <__gdtoa+0x1634>
   192a8:	00000c13          	li	s8,0
   192ac:	001d8993          	addi	s3,s11,1
   192b0:	af8ff06f          	j	185a8 <__gdtoa+0xf68>
   192b4:	acd05e63          	blez	a3,18590 <__gdtoa+0xf50>
   192b8:	9bdff06f          	j	18c74 <__gdtoa+0x1634>

000192bc <__rv_alloc_D2A>:
   192bc:	ff010113          	addi	sp,sp,-16
   192c0:	00812423          	sw	s0,8(sp)
   192c4:	00112623          	sw	ra,12(sp)
   192c8:	01300793          	li	a5,19
   192cc:	00000413          	li	s0,0
   192d0:	00b7fc63          	bgeu	a5,a1,192e8 <__rv_alloc_D2A+0x2c>
   192d4:	00400793          	li	a5,4
   192d8:	00179793          	slli	a5,a5,0x1
   192dc:	01078713          	addi	a4,a5,16
   192e0:	00140413          	addi	s0,s0,1
   192e4:	fee5fae3          	bgeu	a1,a4,192d8 <__rv_alloc_D2A+0x1c>
   192e8:	00040593          	mv	a1,s0
   192ec:	49c000ef          	jal	19788 <_Balloc>
   192f0:	00050663          	beqz	a0,192fc <__rv_alloc_D2A+0x40>
   192f4:	00852023          	sw	s0,0(a0)
   192f8:	00450513          	addi	a0,a0,4
   192fc:	00c12083          	lw	ra,12(sp)
   19300:	00812403          	lw	s0,8(sp)
   19304:	01010113          	addi	sp,sp,16
   19308:	00008067          	ret

0001930c <__nrv_alloc_D2A>:
   1930c:	ff010113          	addi	sp,sp,-16
   19310:	00812423          	sw	s0,8(sp)
   19314:	01212023          	sw	s2,0(sp)
   19318:	00112623          	sw	ra,12(sp)
   1931c:	00912223          	sw	s1,4(sp)
   19320:	01300793          	li	a5,19
   19324:	00058413          	mv	s0,a1
   19328:	00060913          	mv	s2,a2
   1932c:	06d7fe63          	bgeu	a5,a3,193a8 <__nrv_alloc_D2A+0x9c>
   19330:	00400793          	li	a5,4
   19334:	00000493          	li	s1,0
   19338:	00179793          	slli	a5,a5,0x1
   1933c:	01078713          	addi	a4,a5,16
   19340:	00148493          	addi	s1,s1,1
   19344:	fee6fae3          	bgeu	a3,a4,19338 <__nrv_alloc_D2A+0x2c>
   19348:	00048593          	mv	a1,s1
   1934c:	43c000ef          	jal	19788 <_Balloc>
   19350:	00050793          	mv	a5,a0
   19354:	04050e63          	beqz	a0,193b0 <__nrv_alloc_D2A+0xa4>
   19358:	00952023          	sw	s1,0(a0)
   1935c:	00044703          	lbu	a4,0(s0)
   19360:	00450513          	addi	a0,a0,4
   19364:	00140593          	addi	a1,s0,1
   19368:	00e78223          	sb	a4,4(a5)
   1936c:	00050793          	mv	a5,a0
   19370:	00070c63          	beqz	a4,19388 <__nrv_alloc_D2A+0x7c>
   19374:	0005c703          	lbu	a4,0(a1)
   19378:	00178793          	addi	a5,a5,1
   1937c:	00158593          	addi	a1,a1,1
   19380:	00e78023          	sb	a4,0(a5)
   19384:	fe0718e3          	bnez	a4,19374 <__nrv_alloc_D2A+0x68>
   19388:	00090463          	beqz	s2,19390 <__nrv_alloc_D2A+0x84>
   1938c:	00f92023          	sw	a5,0(s2)
   19390:	00c12083          	lw	ra,12(sp)
   19394:	00812403          	lw	s0,8(sp)
   19398:	00412483          	lw	s1,4(sp)
   1939c:	00012903          	lw	s2,0(sp)
   193a0:	01010113          	addi	sp,sp,16
   193a4:	00008067          	ret
   193a8:	00000493          	li	s1,0
   193ac:	f9dff06f          	j	19348 <__nrv_alloc_D2A+0x3c>
   193b0:	00000513          	li	a0,0
   193b4:	fddff06f          	j	19390 <__nrv_alloc_D2A+0x84>

000193b8 <__freedtoa>:
   193b8:	ffc5a683          	lw	a3,-4(a1)
   193bc:	00100713          	li	a4,1
   193c0:	00058793          	mv	a5,a1
   193c4:	00d71733          	sll	a4,a4,a3
   193c8:	ffc58593          	addi	a1,a1,-4
   193cc:	00d7a023          	sw	a3,0(a5)
   193d0:	00e7a223          	sw	a4,4(a5)
   193d4:	4680006f          	j	1983c <_Bfree>

000193d8 <__quorem_D2A>:
   193d8:	fe010113          	addi	sp,sp,-32
   193dc:	00912a23          	sw	s1,20(sp)
   193e0:	01052783          	lw	a5,16(a0)
   193e4:	0105a483          	lw	s1,16(a1)
   193e8:	00112e23          	sw	ra,28(sp)
   193ec:	1c97c863          	blt	a5,s1,195bc <__quorem_D2A+0x1e4>
   193f0:	fff48493          	addi	s1,s1,-1
   193f4:	00249313          	slli	t1,s1,0x2
   193f8:	00812c23          	sw	s0,24(sp)
   193fc:	01458413          	addi	s0,a1,20
   19400:	01312623          	sw	s3,12(sp)
   19404:	01412423          	sw	s4,8(sp)
   19408:	006409b3          	add	s3,s0,t1
   1940c:	01450a13          	addi	s4,a0,20
   19410:	0009a783          	lw	a5,0(s3)
   19414:	006a0333          	add	t1,s4,t1
   19418:	00032703          	lw	a4,0(t1)
   1941c:	01212823          	sw	s2,16(sp)
   19420:	01512223          	sw	s5,4(sp)
   19424:	00178793          	addi	a5,a5,1
   19428:	02f75933          	divu	s2,a4,a5
   1942c:	00050a93          	mv	s5,a0
   19430:	0af76e63          	bltu	a4,a5,194ec <__quorem_D2A+0x114>
   19434:	00010537          	lui	a0,0x10
   19438:	00040893          	mv	a7,s0
   1943c:	000a0813          	mv	a6,s4
   19440:	00000f13          	li	t5,0
   19444:	00000e93          	li	t4,0
   19448:	fff50513          	addi	a0,a0,-1 # ffff <exit-0xb5>
   1944c:	0008a783          	lw	a5,0(a7) # 80000000 <__BSS_END__+0x7ffdd2d0>
   19450:	00082603          	lw	a2,0(a6)
   19454:	00480813          	addi	a6,a6,4
   19458:	00a7f6b3          	and	a3,a5,a0
   1945c:	0107d793          	srli	a5,a5,0x10
   19460:	00a67733          	and	a4,a2,a0
   19464:	01065e13          	srli	t3,a2,0x10
   19468:	00488893          	addi	a7,a7,4
   1946c:	032686b3          	mul	a3,a3,s2
   19470:	032787b3          	mul	a5,a5,s2
   19474:	01e686b3          	add	a3,a3,t5
   19478:	00a6f633          	and	a2,a3,a0
   1947c:	40c70733          	sub	a4,a4,a2
   19480:	41d70733          	sub	a4,a4,t4
   19484:	0106d693          	srli	a3,a3,0x10
   19488:	01075613          	srli	a2,a4,0x10
   1948c:	00167613          	andi	a2,a2,1
   19490:	00a77733          	and	a4,a4,a0
   19494:	00d787b3          	add	a5,a5,a3
   19498:	00a7f6b3          	and	a3,a5,a0
   1949c:	00d60633          	add	a2,a2,a3
   194a0:	40ce06b3          	sub	a3,t3,a2
   194a4:	01069613          	slli	a2,a3,0x10
   194a8:	00e66733          	or	a4,a2,a4
   194ac:	0106d693          	srli	a3,a3,0x10
   194b0:	fee82e23          	sw	a4,-4(a6)
   194b4:	0107df13          	srli	t5,a5,0x10
   194b8:	0016fe93          	andi	t4,a3,1
   194bc:	f919f8e3          	bgeu	s3,a7,1944c <__quorem_D2A+0x74>
   194c0:	00032783          	lw	a5,0(t1)
   194c4:	02079463          	bnez	a5,194ec <__quorem_D2A+0x114>
   194c8:	ffc30313          	addi	t1,t1,-4
   194cc:	006a6863          	bltu	s4,t1,194dc <__quorem_D2A+0x104>
   194d0:	0180006f          	j	194e8 <__quorem_D2A+0x110>
   194d4:	fff48493          	addi	s1,s1,-1
   194d8:	006a7863          	bgeu	s4,t1,194e8 <__quorem_D2A+0x110>
   194dc:	00032783          	lw	a5,0(t1)
   194e0:	ffc30313          	addi	t1,t1,-4
   194e4:	fe0788e3          	beqz	a5,194d4 <__quorem_D2A+0xfc>
   194e8:	009aa823          	sw	s1,16(s5)
   194ec:	000a8513          	mv	a0,s5
   194f0:	511000ef          	jal	1a200 <__mcmp>
   194f4:	0a054063          	bltz	a0,19594 <__quorem_D2A+0x1bc>
   194f8:	00010537          	lui	a0,0x10
   194fc:	000a0593          	mv	a1,s4
   19500:	00000693          	li	a3,0
   19504:	fff50513          	addi	a0,a0,-1 # ffff <exit-0xb5>
   19508:	0005a783          	lw	a5,0(a1)
   1950c:	00042603          	lw	a2,0(s0)
   19510:	00458593          	addi	a1,a1,4
   19514:	00a7f733          	and	a4,a5,a0
   19518:	00a67833          	and	a6,a2,a0
   1951c:	41070733          	sub	a4,a4,a6
   19520:	40d70733          	sub	a4,a4,a3
   19524:	01075693          	srli	a3,a4,0x10
   19528:	0016f693          	andi	a3,a3,1
   1952c:	01065613          	srli	a2,a2,0x10
   19530:	00c686b3          	add	a3,a3,a2
   19534:	0107d793          	srli	a5,a5,0x10
   19538:	40d787b3          	sub	a5,a5,a3
   1953c:	01079693          	slli	a3,a5,0x10
   19540:	00a77733          	and	a4,a4,a0
   19544:	00e6e733          	or	a4,a3,a4
   19548:	0107d793          	srli	a5,a5,0x10
   1954c:	00440413          	addi	s0,s0,4
   19550:	fee5ae23          	sw	a4,-4(a1)
   19554:	0017f693          	andi	a3,a5,1
   19558:	fa89f8e3          	bgeu	s3,s0,19508 <__quorem_D2A+0x130>
   1955c:	00249793          	slli	a5,s1,0x2
   19560:	00fa07b3          	add	a5,s4,a5
   19564:	0007a703          	lw	a4,0(a5)
   19568:	02071463          	bnez	a4,19590 <__quorem_D2A+0x1b8>
   1956c:	ffc78793          	addi	a5,a5,-4
   19570:	00fa6863          	bltu	s4,a5,19580 <__quorem_D2A+0x1a8>
   19574:	0180006f          	j	1958c <__quorem_D2A+0x1b4>
   19578:	fff48493          	addi	s1,s1,-1
   1957c:	00fa7863          	bgeu	s4,a5,1958c <__quorem_D2A+0x1b4>
   19580:	0007a703          	lw	a4,0(a5)
   19584:	ffc78793          	addi	a5,a5,-4
   19588:	fe0708e3          	beqz	a4,19578 <__quorem_D2A+0x1a0>
   1958c:	009aa823          	sw	s1,16(s5)
   19590:	00190913          	addi	s2,s2,1
   19594:	01812403          	lw	s0,24(sp)
   19598:	01c12083          	lw	ra,28(sp)
   1959c:	00c12983          	lw	s3,12(sp)
   195a0:	00812a03          	lw	s4,8(sp)
   195a4:	00412a83          	lw	s5,4(sp)
   195a8:	01412483          	lw	s1,20(sp)
   195ac:	00090513          	mv	a0,s2
   195b0:	01012903          	lw	s2,16(sp)
   195b4:	02010113          	addi	sp,sp,32
   195b8:	00008067          	ret
   195bc:	01c12083          	lw	ra,28(sp)
   195c0:	01412483          	lw	s1,20(sp)
   195c4:	00000513          	li	a0,0
   195c8:	02010113          	addi	sp,sp,32
   195cc:	00008067          	ret

000195d0 <__rshift_D2A>:
   195d0:	01052803          	lw	a6,16(a0)
   195d4:	4055de13          	srai	t3,a1,0x5
   195d8:	010e4863          	blt	t3,a6,195e8 <__rshift_D2A+0x18>
   195dc:	00052823          	sw	zero,16(a0)
   195e0:	00052a23          	sw	zero,20(a0)
   195e4:	00008067          	ret
   195e8:	01450313          	addi	t1,a0,20
   195ec:	00281613          	slli	a2,a6,0x2
   195f0:	002e1793          	slli	a5,t3,0x2
   195f4:	01f5f593          	andi	a1,a1,31
   195f8:	00c30633          	add	a2,t1,a2
   195fc:	00f307b3          	add	a5,t1,a5
   19600:	06058263          	beqz	a1,19664 <__rshift_D2A+0x94>
   19604:	0007a683          	lw	a3,0(a5)
   19608:	02000e93          	li	t4,32
   1960c:	00478793          	addi	a5,a5,4
   19610:	40be8eb3          	sub	t4,t4,a1
   19614:	00b6d6b3          	srl	a3,a3,a1
   19618:	08c7f463          	bgeu	a5,a2,196a0 <__rshift_D2A+0xd0>
   1961c:	00030893          	mv	a7,t1
   19620:	0007a703          	lw	a4,0(a5)
   19624:	00488893          	addi	a7,a7,4
   19628:	00478793          	addi	a5,a5,4
   1962c:	01d71733          	sll	a4,a4,t4
   19630:	00d76733          	or	a4,a4,a3
   19634:	fee8ae23          	sw	a4,-4(a7)
   19638:	ffc7a683          	lw	a3,-4(a5)
   1963c:	00b6d6b3          	srl	a3,a3,a1
   19640:	fec7e0e3          	bltu	a5,a2,19620 <__rshift_D2A+0x50>
   19644:	41c80833          	sub	a6,a6,t3
   19648:	00281813          	slli	a6,a6,0x2
   1964c:	ffc80813          	addi	a6,a6,-4
   19650:	01030833          	add	a6,t1,a6
   19654:	00d82023          	sw	a3,0(a6)
   19658:	02068a63          	beqz	a3,1968c <__rshift_D2A+0xbc>
   1965c:	00480813          	addi	a6,a6,4
   19660:	02c0006f          	j	1968c <__rshift_D2A+0xbc>
   19664:	00030713          	mv	a4,t1
   19668:	f6c7fae3          	bgeu	a5,a2,195dc <__rshift_D2A+0xc>
   1966c:	0007a683          	lw	a3,0(a5)
   19670:	00478793          	addi	a5,a5,4
   19674:	00470713          	addi	a4,a4,4
   19678:	fed72e23          	sw	a3,-4(a4)
   1967c:	fec7e8e3          	bltu	a5,a2,1966c <__rshift_D2A+0x9c>
   19680:	41c80833          	sub	a6,a6,t3
   19684:	00281813          	slli	a6,a6,0x2
   19688:	01030833          	add	a6,t1,a6
   1968c:	406807b3          	sub	a5,a6,t1
   19690:	4027d793          	srai	a5,a5,0x2
   19694:	00f52823          	sw	a5,16(a0)
   19698:	f46804e3          	beq	a6,t1,195e0 <__rshift_D2A+0x10>
   1969c:	00008067          	ret
   196a0:	00d52a23          	sw	a3,20(a0)
   196a4:	f2068ce3          	beqz	a3,195dc <__rshift_D2A+0xc>
   196a8:	00030813          	mv	a6,t1
   196ac:	00480813          	addi	a6,a6,4
   196b0:	fddff06f          	j	1968c <__rshift_D2A+0xbc>

000196b4 <__trailz_D2A>:
   196b4:	01052703          	lw	a4,16(a0)
   196b8:	fe010113          	addi	sp,sp,-32
   196bc:	01450513          	addi	a0,a0,20
   196c0:	00271713          	slli	a4,a4,0x2
   196c4:	00812c23          	sw	s0,24(sp)
   196c8:	00112e23          	sw	ra,28(sp)
   196cc:	00e50733          	add	a4,a0,a4
   196d0:	00000413          	li	s0,0
   196d4:	00e56a63          	bltu	a0,a4,196e8 <__trailz_D2A+0x34>
   196d8:	02c0006f          	j	19704 <__trailz_D2A+0x50>
   196dc:	00450513          	addi	a0,a0,4
   196e0:	02040413          	addi	s0,s0,32
   196e4:	02e57063          	bgeu	a0,a4,19704 <__trailz_D2A+0x50>
   196e8:	00052783          	lw	a5,0(a0)
   196ec:	fe0788e3          	beqz	a5,196dc <__trailz_D2A+0x28>
   196f0:	00e57a63          	bgeu	a0,a4,19704 <__trailz_D2A+0x50>
   196f4:	00c10513          	addi	a0,sp,12
   196f8:	00f12623          	sw	a5,12(sp)
   196fc:	460000ef          	jal	19b5c <__lo0bits>
   19700:	00a40433          	add	s0,s0,a0
   19704:	01c12083          	lw	ra,28(sp)
   19708:	00040513          	mv	a0,s0
   1970c:	01812403          	lw	s0,24(sp)
   19710:	02010113          	addi	sp,sp,32
   19714:	00008067          	ret

00019718 <_mbtowc_r>:
   19718:	f9c1a783          	lw	a5,-100(gp) # 2281c <__global_locale+0xe4>
   1971c:	00078067          	jr	a5

00019720 <__ascii_mbtowc>:
   19720:	02058063          	beqz	a1,19740 <__ascii_mbtowc+0x20>
   19724:	04060263          	beqz	a2,19768 <__ascii_mbtowc+0x48>
   19728:	04068863          	beqz	a3,19778 <__ascii_mbtowc+0x58>
   1972c:	00064783          	lbu	a5,0(a2)
   19730:	00f5a023          	sw	a5,0(a1)
   19734:	00064503          	lbu	a0,0(a2)
   19738:	00a03533          	snez	a0,a0
   1973c:	00008067          	ret
   19740:	ff010113          	addi	sp,sp,-16
   19744:	00c10593          	addi	a1,sp,12
   19748:	02060463          	beqz	a2,19770 <__ascii_mbtowc+0x50>
   1974c:	02068a63          	beqz	a3,19780 <__ascii_mbtowc+0x60>
   19750:	00064783          	lbu	a5,0(a2)
   19754:	00f5a023          	sw	a5,0(a1)
   19758:	00064503          	lbu	a0,0(a2)
   1975c:	00a03533          	snez	a0,a0
   19760:	01010113          	addi	sp,sp,16
   19764:	00008067          	ret
   19768:	00000513          	li	a0,0
   1976c:	00008067          	ret
   19770:	00000513          	li	a0,0
   19774:	fedff06f          	j	19760 <__ascii_mbtowc+0x40>
   19778:	ffe00513          	li	a0,-2
   1977c:	00008067          	ret
   19780:	ffe00513          	li	a0,-2
   19784:	fddff06f          	j	19760 <__ascii_mbtowc+0x40>

00019788 <_Balloc>:
   19788:	04452783          	lw	a5,68(a0)
   1978c:	ff010113          	addi	sp,sp,-16
   19790:	00812423          	sw	s0,8(sp)
   19794:	00912223          	sw	s1,4(sp)
   19798:	00112623          	sw	ra,12(sp)
   1979c:	00050413          	mv	s0,a0
   197a0:	00058493          	mv	s1,a1
   197a4:	02078c63          	beqz	a5,197dc <_Balloc+0x54>
   197a8:	00249713          	slli	a4,s1,0x2
   197ac:	00e787b3          	add	a5,a5,a4
   197b0:	0007a503          	lw	a0,0(a5)
   197b4:	04050463          	beqz	a0,197fc <_Balloc+0x74>
   197b8:	00052703          	lw	a4,0(a0)
   197bc:	00e7a023          	sw	a4,0(a5)
   197c0:	00052823          	sw	zero,16(a0)
   197c4:	00052623          	sw	zero,12(a0)
   197c8:	00c12083          	lw	ra,12(sp)
   197cc:	00812403          	lw	s0,8(sp)
   197d0:	00412483          	lw	s1,4(sp)
   197d4:	01010113          	addi	sp,sp,16
   197d8:	00008067          	ret
   197dc:	02100613          	li	a2,33
   197e0:	00400593          	li	a1,4
   197e4:	591010ef          	jal	1b574 <_calloc_r>
   197e8:	04a42223          	sw	a0,68(s0)
   197ec:	00050793          	mv	a5,a0
   197f0:	fa051ce3          	bnez	a0,197a8 <_Balloc+0x20>
   197f4:	00000513          	li	a0,0
   197f8:	fd1ff06f          	j	197c8 <_Balloc+0x40>
   197fc:	01212023          	sw	s2,0(sp)
   19800:	00100913          	li	s2,1
   19804:	00991933          	sll	s2,s2,s1
   19808:	00590613          	addi	a2,s2,5
   1980c:	00261613          	slli	a2,a2,0x2
   19810:	00100593          	li	a1,1
   19814:	00040513          	mv	a0,s0
   19818:	55d010ef          	jal	1b574 <_calloc_r>
   1981c:	00050a63          	beqz	a0,19830 <_Balloc+0xa8>
   19820:	01252423          	sw	s2,8(a0)
   19824:	00952223          	sw	s1,4(a0)
   19828:	00012903          	lw	s2,0(sp)
   1982c:	f95ff06f          	j	197c0 <_Balloc+0x38>
   19830:	00012903          	lw	s2,0(sp)
   19834:	00000513          	li	a0,0
   19838:	f91ff06f          	j	197c8 <_Balloc+0x40>

0001983c <_Bfree>:
   1983c:	02058063          	beqz	a1,1985c <_Bfree+0x20>
   19840:	0045a703          	lw	a4,4(a1)
   19844:	04452783          	lw	a5,68(a0)
   19848:	00271713          	slli	a4,a4,0x2
   1984c:	00e787b3          	add	a5,a5,a4
   19850:	0007a703          	lw	a4,0(a5)
   19854:	00e5a023          	sw	a4,0(a1)
   19858:	00b7a023          	sw	a1,0(a5)
   1985c:	00008067          	ret

00019860 <__multadd>:
   19860:	fe010113          	addi	sp,sp,-32
   19864:	00912a23          	sw	s1,20(sp)
   19868:	0105a483          	lw	s1,16(a1)
   1986c:	00010337          	lui	t1,0x10
   19870:	00812c23          	sw	s0,24(sp)
   19874:	01212823          	sw	s2,16(sp)
   19878:	01312623          	sw	s3,12(sp)
   1987c:	00112e23          	sw	ra,28(sp)
   19880:	00058913          	mv	s2,a1
   19884:	00050993          	mv	s3,a0
   19888:	00068413          	mv	s0,a3
   1988c:	01458813          	addi	a6,a1,20
   19890:	00000893          	li	a7,0
   19894:	fff30313          	addi	t1,t1,-1 # ffff <exit-0xb5>
   19898:	00082783          	lw	a5,0(a6)
   1989c:	00480813          	addi	a6,a6,4
   198a0:	00188893          	addi	a7,a7,1
   198a4:	0067f733          	and	a4,a5,t1
   198a8:	02c70733          	mul	a4,a4,a2
   198ac:	0107d693          	srli	a3,a5,0x10
   198b0:	02c686b3          	mul	a3,a3,a2
   198b4:	008707b3          	add	a5,a4,s0
   198b8:	0107d713          	srli	a4,a5,0x10
   198bc:	0067f7b3          	and	a5,a5,t1
   198c0:	00e686b3          	add	a3,a3,a4
   198c4:	01069713          	slli	a4,a3,0x10
   198c8:	00f707b3          	add	a5,a4,a5
   198cc:	fef82e23          	sw	a5,-4(a6)
   198d0:	0106d413          	srli	s0,a3,0x10
   198d4:	fc98c2e3          	blt	a7,s1,19898 <__multadd+0x38>
   198d8:	02040263          	beqz	s0,198fc <__multadd+0x9c>
   198dc:	00892783          	lw	a5,8(s2)
   198e0:	02f4de63          	bge	s1,a5,1991c <__multadd+0xbc>
   198e4:	00448793          	addi	a5,s1,4
   198e8:	00279793          	slli	a5,a5,0x2
   198ec:	00f907b3          	add	a5,s2,a5
   198f0:	0087a223          	sw	s0,4(a5)
   198f4:	00148493          	addi	s1,s1,1
   198f8:	00992823          	sw	s1,16(s2)
   198fc:	01c12083          	lw	ra,28(sp)
   19900:	01812403          	lw	s0,24(sp)
   19904:	01412483          	lw	s1,20(sp)
   19908:	00c12983          	lw	s3,12(sp)
   1990c:	00090513          	mv	a0,s2
   19910:	01012903          	lw	s2,16(sp)
   19914:	02010113          	addi	sp,sp,32
   19918:	00008067          	ret
   1991c:	00492583          	lw	a1,4(s2)
   19920:	00098513          	mv	a0,s3
   19924:	01412423          	sw	s4,8(sp)
   19928:	00158593          	addi	a1,a1,1
   1992c:	e5dff0ef          	jal	19788 <_Balloc>
   19930:	00050a13          	mv	s4,a0
   19934:	04050e63          	beqz	a0,19990 <__multadd+0x130>
   19938:	01092603          	lw	a2,16(s2)
   1993c:	00c90593          	addi	a1,s2,12
   19940:	00c50513          	addi	a0,a0,12
   19944:	00260613          	addi	a2,a2,2
   19948:	00261613          	slli	a2,a2,0x2
   1994c:	ba4fd0ef          	jal	16cf0 <memcpy>
   19950:	00492703          	lw	a4,4(s2)
   19954:	0449a783          	lw	a5,68(s3)
   19958:	00271713          	slli	a4,a4,0x2
   1995c:	00e787b3          	add	a5,a5,a4
   19960:	0007a703          	lw	a4,0(a5)
   19964:	00e92023          	sw	a4,0(s2)
   19968:	0127a023          	sw	s2,0(a5)
   1996c:	00448793          	addi	a5,s1,4
   19970:	000a0913          	mv	s2,s4
   19974:	00279793          	slli	a5,a5,0x2
   19978:	00f907b3          	add	a5,s2,a5
   1997c:	00812a03          	lw	s4,8(sp)
   19980:	00148493          	addi	s1,s1,1
   19984:	0087a223          	sw	s0,4(a5)
   19988:	00992823          	sw	s1,16(s2)
   1998c:	f71ff06f          	j	198fc <__multadd+0x9c>
   19990:	00007697          	auipc	a3,0x7
   19994:	20c68693          	addi	a3,a3,524 # 20b9c <_exit+0x1ec>
   19998:	00000613          	li	a2,0
   1999c:	0ba00593          	li	a1,186
   199a0:	00007517          	auipc	a0,0x7
   199a4:	21050513          	addi	a0,a0,528 # 20bb0 <_exit+0x200>
   199a8:	365010ef          	jal	1b50c <__assert_func>

000199ac <__s2b>:
   199ac:	fe010113          	addi	sp,sp,-32
   199b0:	00812c23          	sw	s0,24(sp)
   199b4:	00912a23          	sw	s1,20(sp)
   199b8:	01212823          	sw	s2,16(sp)
   199bc:	01312623          	sw	s3,12(sp)
   199c0:	01412423          	sw	s4,8(sp)
   199c4:	00068993          	mv	s3,a3
   199c8:	00900793          	li	a5,9
   199cc:	00868693          	addi	a3,a3,8
   199d0:	00112e23          	sw	ra,28(sp)
   199d4:	02f6c6b3          	div	a3,a3,a5
   199d8:	00050913          	mv	s2,a0
   199dc:	00058413          	mv	s0,a1
   199e0:	00060a13          	mv	s4,a2
   199e4:	00070493          	mv	s1,a4
   199e8:	0d37da63          	bge	a5,s3,19abc <__s2b+0x110>
   199ec:	00100793          	li	a5,1
   199f0:	00000593          	li	a1,0
   199f4:	00179793          	slli	a5,a5,0x1
   199f8:	00158593          	addi	a1,a1,1
   199fc:	fed7cce3          	blt	a5,a3,199f4 <__s2b+0x48>
   19a00:	00090513          	mv	a0,s2
   19a04:	d85ff0ef          	jal	19788 <_Balloc>
   19a08:	00050593          	mv	a1,a0
   19a0c:	0a050c63          	beqz	a0,19ac4 <__s2b+0x118>
   19a10:	00100793          	li	a5,1
   19a14:	00f52823          	sw	a5,16(a0)
   19a18:	00952a23          	sw	s1,20(a0)
   19a1c:	00900793          	li	a5,9
   19a20:	0947d863          	bge	a5,s4,19ab0 <__s2b+0x104>
   19a24:	01512223          	sw	s5,4(sp)
   19a28:	00940a93          	addi	s5,s0,9
   19a2c:	000a8493          	mv	s1,s5
   19a30:	01440433          	add	s0,s0,s4
   19a34:	0004c683          	lbu	a3,0(s1)
   19a38:	00a00613          	li	a2,10
   19a3c:	00090513          	mv	a0,s2
   19a40:	fd068693          	addi	a3,a3,-48
   19a44:	e1dff0ef          	jal	19860 <__multadd>
   19a48:	00148493          	addi	s1,s1,1
   19a4c:	00050593          	mv	a1,a0
   19a50:	fe8492e3          	bne	s1,s0,19a34 <__s2b+0x88>
   19a54:	ff8a0413          	addi	s0,s4,-8
   19a58:	008a8433          	add	s0,s5,s0
   19a5c:	00412a83          	lw	s5,4(sp)
   19a60:	033a5663          	bge	s4,s3,19a8c <__s2b+0xe0>
   19a64:	414989b3          	sub	s3,s3,s4
   19a68:	013409b3          	add	s3,s0,s3
   19a6c:	00044683          	lbu	a3,0(s0)
   19a70:	00a00613          	li	a2,10
   19a74:	00090513          	mv	a0,s2
   19a78:	fd068693          	addi	a3,a3,-48
   19a7c:	de5ff0ef          	jal	19860 <__multadd>
   19a80:	00140413          	addi	s0,s0,1
   19a84:	00050593          	mv	a1,a0
   19a88:	ff3412e3          	bne	s0,s3,19a6c <__s2b+0xc0>
   19a8c:	01c12083          	lw	ra,28(sp)
   19a90:	01812403          	lw	s0,24(sp)
   19a94:	01412483          	lw	s1,20(sp)
   19a98:	01012903          	lw	s2,16(sp)
   19a9c:	00c12983          	lw	s3,12(sp)
   19aa0:	00812a03          	lw	s4,8(sp)
   19aa4:	00058513          	mv	a0,a1
   19aa8:	02010113          	addi	sp,sp,32
   19aac:	00008067          	ret
   19ab0:	00a40413          	addi	s0,s0,10
   19ab4:	00900a13          	li	s4,9
   19ab8:	fa9ff06f          	j	19a60 <__s2b+0xb4>
   19abc:	00000593          	li	a1,0
   19ac0:	f41ff06f          	j	19a00 <__s2b+0x54>
   19ac4:	00007697          	auipc	a3,0x7
   19ac8:	0d868693          	addi	a3,a3,216 # 20b9c <_exit+0x1ec>
   19acc:	00000613          	li	a2,0
   19ad0:	0d300593          	li	a1,211
   19ad4:	00007517          	auipc	a0,0x7
   19ad8:	0dc50513          	addi	a0,a0,220 # 20bb0 <_exit+0x200>
   19adc:	01512223          	sw	s5,4(sp)
   19ae0:	22d010ef          	jal	1b50c <__assert_func>

00019ae4 <__hi0bits>:
   19ae4:	00050793          	mv	a5,a0
   19ae8:	00010737          	lui	a4,0x10
   19aec:	00000513          	li	a0,0
   19af0:	00e7f663          	bgeu	a5,a4,19afc <__hi0bits+0x18>
   19af4:	01079793          	slli	a5,a5,0x10
   19af8:	01000513          	li	a0,16
   19afc:	01000737          	lui	a4,0x1000
   19b00:	00e7f663          	bgeu	a5,a4,19b0c <__hi0bits+0x28>
   19b04:	00850513          	addi	a0,a0,8
   19b08:	00879793          	slli	a5,a5,0x8
   19b0c:	10000737          	lui	a4,0x10000
   19b10:	00e7f663          	bgeu	a5,a4,19b1c <__hi0bits+0x38>
   19b14:	00450513          	addi	a0,a0,4
   19b18:	00479793          	slli	a5,a5,0x4
   19b1c:	40000737          	lui	a4,0x40000
   19b20:	00e7ea63          	bltu	a5,a4,19b34 <__hi0bits+0x50>
   19b24:	fff7c793          	not	a5,a5
   19b28:	01f7d793          	srli	a5,a5,0x1f
   19b2c:	00f50533          	add	a0,a0,a5
   19b30:	00008067          	ret
   19b34:	00279793          	slli	a5,a5,0x2
   19b38:	0007ca63          	bltz	a5,19b4c <__hi0bits+0x68>
   19b3c:	00179713          	slli	a4,a5,0x1
   19b40:	00074a63          	bltz	a4,19b54 <__hi0bits+0x70>
   19b44:	02000513          	li	a0,32
   19b48:	00008067          	ret
   19b4c:	00250513          	addi	a0,a0,2
   19b50:	00008067          	ret
   19b54:	00350513          	addi	a0,a0,3
   19b58:	00008067          	ret

00019b5c <__lo0bits>:
   19b5c:	00052783          	lw	a5,0(a0)
   19b60:	00050713          	mv	a4,a0
   19b64:	0077f693          	andi	a3,a5,7
   19b68:	02068463          	beqz	a3,19b90 <__lo0bits+0x34>
   19b6c:	0017f693          	andi	a3,a5,1
   19b70:	00000513          	li	a0,0
   19b74:	04069e63          	bnez	a3,19bd0 <__lo0bits+0x74>
   19b78:	0027f693          	andi	a3,a5,2
   19b7c:	0a068863          	beqz	a3,19c2c <__lo0bits+0xd0>
   19b80:	0017d793          	srli	a5,a5,0x1
   19b84:	00f72023          	sw	a5,0(a4) # 40000000 <__BSS_END__+0x3ffdd2d0>
   19b88:	00100513          	li	a0,1
   19b8c:	00008067          	ret
   19b90:	01079693          	slli	a3,a5,0x10
   19b94:	0106d693          	srli	a3,a3,0x10
   19b98:	00000513          	li	a0,0
   19b9c:	06068e63          	beqz	a3,19c18 <__lo0bits+0xbc>
   19ba0:	0ff7f693          	zext.b	a3,a5
   19ba4:	06068063          	beqz	a3,19c04 <__lo0bits+0xa8>
   19ba8:	00f7f693          	andi	a3,a5,15
   19bac:	04068263          	beqz	a3,19bf0 <__lo0bits+0x94>
   19bb0:	0037f693          	andi	a3,a5,3
   19bb4:	02068463          	beqz	a3,19bdc <__lo0bits+0x80>
   19bb8:	0017f693          	andi	a3,a5,1
   19bbc:	00069c63          	bnez	a3,19bd4 <__lo0bits+0x78>
   19bc0:	0017d793          	srli	a5,a5,0x1
   19bc4:	00150513          	addi	a0,a0,1
   19bc8:	00079663          	bnez	a5,19bd4 <__lo0bits+0x78>
   19bcc:	02000513          	li	a0,32
   19bd0:	00008067          	ret
   19bd4:	00f72023          	sw	a5,0(a4)
   19bd8:	00008067          	ret
   19bdc:	0027d793          	srli	a5,a5,0x2
   19be0:	0017f693          	andi	a3,a5,1
   19be4:	00250513          	addi	a0,a0,2
   19be8:	fe0696e3          	bnez	a3,19bd4 <__lo0bits+0x78>
   19bec:	fd5ff06f          	j	19bc0 <__lo0bits+0x64>
   19bf0:	0047d793          	srli	a5,a5,0x4
   19bf4:	0037f693          	andi	a3,a5,3
   19bf8:	00450513          	addi	a0,a0,4
   19bfc:	fa069ee3          	bnez	a3,19bb8 <__lo0bits+0x5c>
   19c00:	fddff06f          	j	19bdc <__lo0bits+0x80>
   19c04:	0087d793          	srli	a5,a5,0x8
   19c08:	00f7f693          	andi	a3,a5,15
   19c0c:	00850513          	addi	a0,a0,8
   19c10:	fa0690e3          	bnez	a3,19bb0 <__lo0bits+0x54>
   19c14:	fddff06f          	j	19bf0 <__lo0bits+0x94>
   19c18:	0107d793          	srli	a5,a5,0x10
   19c1c:	0ff7f693          	zext.b	a3,a5
   19c20:	01000513          	li	a0,16
   19c24:	f80692e3          	bnez	a3,19ba8 <__lo0bits+0x4c>
   19c28:	fddff06f          	j	19c04 <__lo0bits+0xa8>
   19c2c:	0027d793          	srli	a5,a5,0x2
   19c30:	00f72023          	sw	a5,0(a4)
   19c34:	00200513          	li	a0,2
   19c38:	00008067          	ret

00019c3c <__i2b>:
   19c3c:	04452783          	lw	a5,68(a0)
   19c40:	ff010113          	addi	sp,sp,-16
   19c44:	00812423          	sw	s0,8(sp)
   19c48:	00912223          	sw	s1,4(sp)
   19c4c:	00112623          	sw	ra,12(sp)
   19c50:	00050413          	mv	s0,a0
   19c54:	00058493          	mv	s1,a1
   19c58:	02078c63          	beqz	a5,19c90 <__i2b+0x54>
   19c5c:	0047a503          	lw	a0,4(a5)
   19c60:	06050263          	beqz	a0,19cc4 <__i2b+0x88>
   19c64:	00052703          	lw	a4,0(a0)
   19c68:	00e7a223          	sw	a4,4(a5)
   19c6c:	00c12083          	lw	ra,12(sp)
   19c70:	00812403          	lw	s0,8(sp)
   19c74:	00100793          	li	a5,1
   19c78:	00952a23          	sw	s1,20(a0)
   19c7c:	00052623          	sw	zero,12(a0)
   19c80:	00f52823          	sw	a5,16(a0)
   19c84:	00412483          	lw	s1,4(sp)
   19c88:	01010113          	addi	sp,sp,16
   19c8c:	00008067          	ret
   19c90:	02100613          	li	a2,33
   19c94:	00400593          	li	a1,4
   19c98:	0dd010ef          	jal	1b574 <_calloc_r>
   19c9c:	04a42223          	sw	a0,68(s0)
   19ca0:	00050793          	mv	a5,a0
   19ca4:	fa051ce3          	bnez	a0,19c5c <__i2b+0x20>
   19ca8:	00007697          	auipc	a3,0x7
   19cac:	ef468693          	addi	a3,a3,-268 # 20b9c <_exit+0x1ec>
   19cb0:	00000613          	li	a2,0
   19cb4:	14500593          	li	a1,325
   19cb8:	00007517          	auipc	a0,0x7
   19cbc:	ef850513          	addi	a0,a0,-264 # 20bb0 <_exit+0x200>
   19cc0:	04d010ef          	jal	1b50c <__assert_func>
   19cc4:	01c00613          	li	a2,28
   19cc8:	00100593          	li	a1,1
   19ccc:	00040513          	mv	a0,s0
   19cd0:	0a5010ef          	jal	1b574 <_calloc_r>
   19cd4:	fc050ae3          	beqz	a0,19ca8 <__i2b+0x6c>
   19cd8:	00100793          	li	a5,1
   19cdc:	00f52223          	sw	a5,4(a0)
   19ce0:	00200793          	li	a5,2
   19ce4:	00f52423          	sw	a5,8(a0)
   19ce8:	f85ff06f          	j	19c6c <__i2b+0x30>

00019cec <__multiply>:
   19cec:	fe010113          	addi	sp,sp,-32
   19cf0:	01212823          	sw	s2,16(sp)
   19cf4:	01312623          	sw	s3,12(sp)
   19cf8:	0105a903          	lw	s2,16(a1)
   19cfc:	01062983          	lw	s3,16(a2)
   19d00:	00912a23          	sw	s1,20(sp)
   19d04:	01412423          	sw	s4,8(sp)
   19d08:	00112e23          	sw	ra,28(sp)
   19d0c:	00812c23          	sw	s0,24(sp)
   19d10:	00058a13          	mv	s4,a1
   19d14:	00060493          	mv	s1,a2
   19d18:	01394c63          	blt	s2,s3,19d30 <__multiply+0x44>
   19d1c:	00098713          	mv	a4,s3
   19d20:	00058493          	mv	s1,a1
   19d24:	00090993          	mv	s3,s2
   19d28:	00060a13          	mv	s4,a2
   19d2c:	00070913          	mv	s2,a4
   19d30:	0084a783          	lw	a5,8(s1)
   19d34:	0044a583          	lw	a1,4(s1)
   19d38:	01298433          	add	s0,s3,s2
   19d3c:	0087a7b3          	slt	a5,a5,s0
   19d40:	00f585b3          	add	a1,a1,a5
   19d44:	a45ff0ef          	jal	19788 <_Balloc>
   19d48:	1a050e63          	beqz	a0,19f04 <__multiply+0x218>
   19d4c:	01450313          	addi	t1,a0,20
   19d50:	00241893          	slli	a7,s0,0x2
   19d54:	011308b3          	add	a7,t1,a7
   19d58:	00030793          	mv	a5,t1
   19d5c:	01137863          	bgeu	t1,a7,19d6c <__multiply+0x80>
   19d60:	0007a023          	sw	zero,0(a5)
   19d64:	00478793          	addi	a5,a5,4
   19d68:	ff17ece3          	bltu	a5,a7,19d60 <__multiply+0x74>
   19d6c:	014a0813          	addi	a6,s4,20
   19d70:	00291e13          	slli	t3,s2,0x2
   19d74:	01448e93          	addi	t4,s1,20
   19d78:	00299593          	slli	a1,s3,0x2
   19d7c:	01c80e33          	add	t3,a6,t3
   19d80:	00be85b3          	add	a1,t4,a1
   19d84:	13c87663          	bgeu	a6,t3,19eb0 <__multiply+0x1c4>
   19d88:	01548793          	addi	a5,s1,21
   19d8c:	00400f13          	li	t5,4
   19d90:	16f5f063          	bgeu	a1,a5,19ef0 <__multiply+0x204>
   19d94:	000106b7          	lui	a3,0x10
   19d98:	fff68693          	addi	a3,a3,-1 # ffff <exit-0xb5>
   19d9c:	0180006f          	j	19db4 <__multiply+0xc8>
   19da0:	010fdf93          	srli	t6,t6,0x10
   19da4:	080f9863          	bnez	t6,19e34 <__multiply+0x148>
   19da8:	00480813          	addi	a6,a6,4
   19dac:	00430313          	addi	t1,t1,4
   19db0:	11c87063          	bgeu	a6,t3,19eb0 <__multiply+0x1c4>
   19db4:	00082f83          	lw	t6,0(a6)
   19db8:	00dff3b3          	and	t2,t6,a3
   19dbc:	fe0382e3          	beqz	t2,19da0 <__multiply+0xb4>
   19dc0:	00030293          	mv	t0,t1
   19dc4:	000e8f93          	mv	t6,t4
   19dc8:	00000493          	li	s1,0
   19dcc:	000fa783          	lw	a5,0(t6)
   19dd0:	0002a603          	lw	a2,0(t0)
   19dd4:	00428293          	addi	t0,t0,4
   19dd8:	00d7f733          	and	a4,a5,a3
   19ddc:	02770733          	mul	a4,a4,t2
   19de0:	0107d793          	srli	a5,a5,0x10
   19de4:	00d67933          	and	s2,a2,a3
   19de8:	01065613          	srli	a2,a2,0x10
   19dec:	004f8f93          	addi	t6,t6,4
   19df0:	027787b3          	mul	a5,a5,t2
   19df4:	01270733          	add	a4,a4,s2
   19df8:	00970733          	add	a4,a4,s1
   19dfc:	01075493          	srli	s1,a4,0x10
   19e00:	00d77733          	and	a4,a4,a3
   19e04:	00c787b3          	add	a5,a5,a2
   19e08:	009787b3          	add	a5,a5,s1
   19e0c:	01079613          	slli	a2,a5,0x10
   19e10:	00e66733          	or	a4,a2,a4
   19e14:	fee2ae23          	sw	a4,-4(t0)
   19e18:	0107d493          	srli	s1,a5,0x10
   19e1c:	fabfe8e3          	bltu	t6,a1,19dcc <__multiply+0xe0>
   19e20:	01e307b3          	add	a5,t1,t5
   19e24:	0097a023          	sw	s1,0(a5)
   19e28:	00082f83          	lw	t6,0(a6)
   19e2c:	010fdf93          	srli	t6,t6,0x10
   19e30:	f60f8ce3          	beqz	t6,19da8 <__multiply+0xbc>
   19e34:	00032703          	lw	a4,0(t1)
   19e38:	00030293          	mv	t0,t1
   19e3c:	000e8613          	mv	a2,t4
   19e40:	00070493          	mv	s1,a4
   19e44:	00000393          	li	t2,0
   19e48:	00062783          	lw	a5,0(a2)
   19e4c:	0104d993          	srli	s3,s1,0x10
   19e50:	00d77733          	and	a4,a4,a3
   19e54:	00d7f7b3          	and	a5,a5,a3
   19e58:	03f787b3          	mul	a5,a5,t6
   19e5c:	0042a483          	lw	s1,4(t0)
   19e60:	00428293          	addi	t0,t0,4
   19e64:	00460613          	addi	a2,a2,4
   19e68:	00d4f933          	and	s2,s1,a3
   19e6c:	007787b3          	add	a5,a5,t2
   19e70:	013787b3          	add	a5,a5,s3
   19e74:	01079393          	slli	t2,a5,0x10
   19e78:	00e3e733          	or	a4,t2,a4
   19e7c:	fee2ae23          	sw	a4,-4(t0)
   19e80:	ffe65703          	lhu	a4,-2(a2)
   19e84:	0107d793          	srli	a5,a5,0x10
   19e88:	03f70733          	mul	a4,a4,t6
   19e8c:	01270733          	add	a4,a4,s2
   19e90:	00f70733          	add	a4,a4,a5
   19e94:	01075393          	srli	t2,a4,0x10
   19e98:	fab668e3          	bltu	a2,a1,19e48 <__multiply+0x15c>
   19e9c:	01e307b3          	add	a5,t1,t5
   19ea0:	00e7a023          	sw	a4,0(a5)
   19ea4:	00480813          	addi	a6,a6,4
   19ea8:	00430313          	addi	t1,t1,4
   19eac:	f1c864e3          	bltu	a6,t3,19db4 <__multiply+0xc8>
   19eb0:	00804863          	bgtz	s0,19ec0 <__multiply+0x1d4>
   19eb4:	0180006f          	j	19ecc <__multiply+0x1e0>
   19eb8:	fff40413          	addi	s0,s0,-1
   19ebc:	00040863          	beqz	s0,19ecc <__multiply+0x1e0>
   19ec0:	ffc8a783          	lw	a5,-4(a7)
   19ec4:	ffc88893          	addi	a7,a7,-4
   19ec8:	fe0788e3          	beqz	a5,19eb8 <__multiply+0x1cc>
   19ecc:	01c12083          	lw	ra,28(sp)
   19ed0:	00852823          	sw	s0,16(a0)
   19ed4:	01812403          	lw	s0,24(sp)
   19ed8:	01412483          	lw	s1,20(sp)
   19edc:	01012903          	lw	s2,16(sp)
   19ee0:	00c12983          	lw	s3,12(sp)
   19ee4:	00812a03          	lw	s4,8(sp)
   19ee8:	02010113          	addi	sp,sp,32
   19eec:	00008067          	ret
   19ef0:	409587b3          	sub	a5,a1,s1
   19ef4:	feb78793          	addi	a5,a5,-21
   19ef8:	ffc7f793          	andi	a5,a5,-4
   19efc:	00478f13          	addi	t5,a5,4
   19f00:	e95ff06f          	j	19d94 <__multiply+0xa8>
   19f04:	00007697          	auipc	a3,0x7
   19f08:	c9868693          	addi	a3,a3,-872 # 20b9c <_exit+0x1ec>
   19f0c:	00000613          	li	a2,0
   19f10:	16200593          	li	a1,354
   19f14:	00007517          	auipc	a0,0x7
   19f18:	c9c50513          	addi	a0,a0,-868 # 20bb0 <_exit+0x200>
   19f1c:	5f0010ef          	jal	1b50c <__assert_func>

00019f20 <__pow5mult>:
   19f20:	fe010113          	addi	sp,sp,-32
   19f24:	00812c23          	sw	s0,24(sp)
   19f28:	01212823          	sw	s2,16(sp)
   19f2c:	01312623          	sw	s3,12(sp)
   19f30:	00112e23          	sw	ra,28(sp)
   19f34:	00367793          	andi	a5,a2,3
   19f38:	00060413          	mv	s0,a2
   19f3c:	00050993          	mv	s3,a0
   19f40:	00058913          	mv	s2,a1
   19f44:	0a079c63          	bnez	a5,19ffc <__pow5mult+0xdc>
   19f48:	40245413          	srai	s0,s0,0x2
   19f4c:	06040a63          	beqz	s0,19fc0 <__pow5mult+0xa0>
   19f50:	00912a23          	sw	s1,20(sp)
   19f54:	0409a483          	lw	s1,64(s3)
   19f58:	0c048663          	beqz	s1,1a024 <__pow5mult+0x104>
   19f5c:	00147793          	andi	a5,s0,1
   19f60:	02079063          	bnez	a5,19f80 <__pow5mult+0x60>
   19f64:	40145413          	srai	s0,s0,0x1
   19f68:	04040a63          	beqz	s0,19fbc <__pow5mult+0x9c>
   19f6c:	0004a503          	lw	a0,0(s1)
   19f70:	06050663          	beqz	a0,19fdc <__pow5mult+0xbc>
   19f74:	00050493          	mv	s1,a0
   19f78:	00147793          	andi	a5,s0,1
   19f7c:	fe0784e3          	beqz	a5,19f64 <__pow5mult+0x44>
   19f80:	00048613          	mv	a2,s1
   19f84:	00090593          	mv	a1,s2
   19f88:	00098513          	mv	a0,s3
   19f8c:	d61ff0ef          	jal	19cec <__multiply>
   19f90:	02090063          	beqz	s2,19fb0 <__pow5mult+0x90>
   19f94:	00492703          	lw	a4,4(s2)
   19f98:	0449a783          	lw	a5,68(s3)
   19f9c:	00271713          	slli	a4,a4,0x2
   19fa0:	00e787b3          	add	a5,a5,a4
   19fa4:	0007a703          	lw	a4,0(a5)
   19fa8:	00e92023          	sw	a4,0(s2)
   19fac:	0127a023          	sw	s2,0(a5)
   19fb0:	40145413          	srai	s0,s0,0x1
   19fb4:	00050913          	mv	s2,a0
   19fb8:	fa041ae3          	bnez	s0,19f6c <__pow5mult+0x4c>
   19fbc:	01412483          	lw	s1,20(sp)
   19fc0:	01c12083          	lw	ra,28(sp)
   19fc4:	01812403          	lw	s0,24(sp)
   19fc8:	00c12983          	lw	s3,12(sp)
   19fcc:	00090513          	mv	a0,s2
   19fd0:	01012903          	lw	s2,16(sp)
   19fd4:	02010113          	addi	sp,sp,32
   19fd8:	00008067          	ret
   19fdc:	00048613          	mv	a2,s1
   19fe0:	00048593          	mv	a1,s1
   19fe4:	00098513          	mv	a0,s3
   19fe8:	d05ff0ef          	jal	19cec <__multiply>
   19fec:	00a4a023          	sw	a0,0(s1)
   19ff0:	00052023          	sw	zero,0(a0)
   19ff4:	00050493          	mv	s1,a0
   19ff8:	f81ff06f          	j	19f78 <__pow5mult+0x58>
   19ffc:	fff78793          	addi	a5,a5,-1
   1a000:	00007717          	auipc	a4,0x7
   1a004:	0b070713          	addi	a4,a4,176 # 210b0 <p05.0>
   1a008:	00279793          	slli	a5,a5,0x2
   1a00c:	00f707b3          	add	a5,a4,a5
   1a010:	0007a603          	lw	a2,0(a5)
   1a014:	00000693          	li	a3,0
   1a018:	849ff0ef          	jal	19860 <__multadd>
   1a01c:	00050913          	mv	s2,a0
   1a020:	f29ff06f          	j	19f48 <__pow5mult+0x28>
   1a024:	00100593          	li	a1,1
   1a028:	00098513          	mv	a0,s3
   1a02c:	f5cff0ef          	jal	19788 <_Balloc>
   1a030:	00050493          	mv	s1,a0
   1a034:	02050063          	beqz	a0,1a054 <__pow5mult+0x134>
   1a038:	27100793          	li	a5,625
   1a03c:	00f52a23          	sw	a5,20(a0)
   1a040:	00100793          	li	a5,1
   1a044:	00f52823          	sw	a5,16(a0)
   1a048:	04a9a023          	sw	a0,64(s3)
   1a04c:	00052023          	sw	zero,0(a0)
   1a050:	f0dff06f          	j	19f5c <__pow5mult+0x3c>
   1a054:	00007697          	auipc	a3,0x7
   1a058:	b4868693          	addi	a3,a3,-1208 # 20b9c <_exit+0x1ec>
   1a05c:	00000613          	li	a2,0
   1a060:	14500593          	li	a1,325
   1a064:	00007517          	auipc	a0,0x7
   1a068:	b4c50513          	addi	a0,a0,-1204 # 20bb0 <_exit+0x200>
   1a06c:	4a0010ef          	jal	1b50c <__assert_func>

0001a070 <__lshift>:
   1a070:	fe010113          	addi	sp,sp,-32
   1a074:	01512223          	sw	s5,4(sp)
   1a078:	0105aa83          	lw	s5,16(a1)
   1a07c:	0085a783          	lw	a5,8(a1)
   1a080:	01312623          	sw	s3,12(sp)
   1a084:	40565993          	srai	s3,a2,0x5
   1a088:	01598ab3          	add	s5,s3,s5
   1a08c:	00812c23          	sw	s0,24(sp)
   1a090:	00912a23          	sw	s1,20(sp)
   1a094:	01212823          	sw	s2,16(sp)
   1a098:	01412423          	sw	s4,8(sp)
   1a09c:	00112e23          	sw	ra,28(sp)
   1a0a0:	001a8913          	addi	s2,s5,1
   1a0a4:	00058493          	mv	s1,a1
   1a0a8:	00060413          	mv	s0,a2
   1a0ac:	0045a583          	lw	a1,4(a1)
   1a0b0:	00050a13          	mv	s4,a0
   1a0b4:	0127d863          	bge	a5,s2,1a0c4 <__lshift+0x54>
   1a0b8:	00179793          	slli	a5,a5,0x1
   1a0bc:	00158593          	addi	a1,a1,1
   1a0c0:	ff27cce3          	blt	a5,s2,1a0b8 <__lshift+0x48>
   1a0c4:	000a0513          	mv	a0,s4
   1a0c8:	ec0ff0ef          	jal	19788 <_Balloc>
   1a0cc:	10050c63          	beqz	a0,1a1e4 <__lshift+0x174>
   1a0d0:	01450813          	addi	a6,a0,20
   1a0d4:	03305463          	blez	s3,1a0fc <__lshift+0x8c>
   1a0d8:	00598993          	addi	s3,s3,5
   1a0dc:	00299993          	slli	s3,s3,0x2
   1a0e0:	01350733          	add	a4,a0,s3
   1a0e4:	00080793          	mv	a5,a6
   1a0e8:	00478793          	addi	a5,a5,4
   1a0ec:	fe07ae23          	sw	zero,-4(a5)
   1a0f0:	fee79ce3          	bne	a5,a4,1a0e8 <__lshift+0x78>
   1a0f4:	fec98993          	addi	s3,s3,-20
   1a0f8:	01380833          	add	a6,a6,s3
   1a0fc:	0104a883          	lw	a7,16(s1)
   1a100:	01448793          	addi	a5,s1,20
   1a104:	01f47613          	andi	a2,s0,31
   1a108:	00289893          	slli	a7,a7,0x2
   1a10c:	011788b3          	add	a7,a5,a7
   1a110:	0a060463          	beqz	a2,1a1b8 <__lshift+0x148>
   1a114:	02000593          	li	a1,32
   1a118:	40c585b3          	sub	a1,a1,a2
   1a11c:	00080313          	mv	t1,a6
   1a120:	00000693          	li	a3,0
   1a124:	0007a703          	lw	a4,0(a5)
   1a128:	00430313          	addi	t1,t1,4
   1a12c:	00478793          	addi	a5,a5,4
   1a130:	00c71733          	sll	a4,a4,a2
   1a134:	00d76733          	or	a4,a4,a3
   1a138:	fee32e23          	sw	a4,-4(t1)
   1a13c:	ffc7a683          	lw	a3,-4(a5)
   1a140:	00b6d6b3          	srl	a3,a3,a1
   1a144:	ff17e0e3          	bltu	a5,a7,1a124 <__lshift+0xb4>
   1a148:	01548793          	addi	a5,s1,21
   1a14c:	00400713          	li	a4,4
   1a150:	00f8ea63          	bltu	a7,a5,1a164 <__lshift+0xf4>
   1a154:	409887b3          	sub	a5,a7,s1
   1a158:	feb78793          	addi	a5,a5,-21
   1a15c:	ffc7f793          	andi	a5,a5,-4
   1a160:	00478713          	addi	a4,a5,4
   1a164:	00e80833          	add	a6,a6,a4
   1a168:	00d82023          	sw	a3,0(a6)
   1a16c:	00069463          	bnez	a3,1a174 <__lshift+0x104>
   1a170:	000a8913          	mv	s2,s5
   1a174:	0044a703          	lw	a4,4(s1)
   1a178:	044a2783          	lw	a5,68(s4)
   1a17c:	01c12083          	lw	ra,28(sp)
   1a180:	00271713          	slli	a4,a4,0x2
   1a184:	00e787b3          	add	a5,a5,a4
   1a188:	0007a703          	lw	a4,0(a5)
   1a18c:	01252823          	sw	s2,16(a0)
   1a190:	01812403          	lw	s0,24(sp)
   1a194:	00e4a023          	sw	a4,0(s1)
   1a198:	0097a023          	sw	s1,0(a5)
   1a19c:	01012903          	lw	s2,16(sp)
   1a1a0:	01412483          	lw	s1,20(sp)
   1a1a4:	00c12983          	lw	s3,12(sp)
   1a1a8:	00812a03          	lw	s4,8(sp)
   1a1ac:	00412a83          	lw	s5,4(sp)
   1a1b0:	02010113          	addi	sp,sp,32
   1a1b4:	00008067          	ret
   1a1b8:	0007a703          	lw	a4,0(a5)
   1a1bc:	00478793          	addi	a5,a5,4
   1a1c0:	00480813          	addi	a6,a6,4
   1a1c4:	fee82e23          	sw	a4,-4(a6)
   1a1c8:	fb17f4e3          	bgeu	a5,a7,1a170 <__lshift+0x100>
   1a1cc:	0007a703          	lw	a4,0(a5)
   1a1d0:	00478793          	addi	a5,a5,4
   1a1d4:	00480813          	addi	a6,a6,4
   1a1d8:	fee82e23          	sw	a4,-4(a6)
   1a1dc:	fd17eee3          	bltu	a5,a7,1a1b8 <__lshift+0x148>
   1a1e0:	f91ff06f          	j	1a170 <__lshift+0x100>
   1a1e4:	00007697          	auipc	a3,0x7
   1a1e8:	9b868693          	addi	a3,a3,-1608 # 20b9c <_exit+0x1ec>
   1a1ec:	00000613          	li	a2,0
   1a1f0:	1de00593          	li	a1,478
   1a1f4:	00007517          	auipc	a0,0x7
   1a1f8:	9bc50513          	addi	a0,a0,-1604 # 20bb0 <_exit+0x200>
   1a1fc:	310010ef          	jal	1b50c <__assert_func>

0001a200 <__mcmp>:
   1a200:	01052703          	lw	a4,16(a0)
   1a204:	0105a783          	lw	a5,16(a1)
   1a208:	00050813          	mv	a6,a0
   1a20c:	40f70533          	sub	a0,a4,a5
   1a210:	04f71263          	bne	a4,a5,1a254 <__mcmp+0x54>
   1a214:	00279793          	slli	a5,a5,0x2
   1a218:	01480813          	addi	a6,a6,20
   1a21c:	01458593          	addi	a1,a1,20
   1a220:	00f80733          	add	a4,a6,a5
   1a224:	00f587b3          	add	a5,a1,a5
   1a228:	0080006f          	j	1a230 <__mcmp+0x30>
   1a22c:	02e87463          	bgeu	a6,a4,1a254 <__mcmp+0x54>
   1a230:	ffc72603          	lw	a2,-4(a4)
   1a234:	ffc7a683          	lw	a3,-4(a5)
   1a238:	ffc70713          	addi	a4,a4,-4
   1a23c:	ffc78793          	addi	a5,a5,-4
   1a240:	fed606e3          	beq	a2,a3,1a22c <__mcmp+0x2c>
   1a244:	00100513          	li	a0,1
   1a248:	00d67663          	bgeu	a2,a3,1a254 <__mcmp+0x54>
   1a24c:	fff00513          	li	a0,-1
   1a250:	00008067          	ret
   1a254:	00008067          	ret

0001a258 <__mdiff>:
   1a258:	0105a703          	lw	a4,16(a1)
   1a25c:	01062783          	lw	a5,16(a2)
   1a260:	ff010113          	addi	sp,sp,-16
   1a264:	00812423          	sw	s0,8(sp)
   1a268:	00912223          	sw	s1,4(sp)
   1a26c:	00112623          	sw	ra,12(sp)
   1a270:	01212023          	sw	s2,0(sp)
   1a274:	00058413          	mv	s0,a1
   1a278:	00060493          	mv	s1,a2
   1a27c:	40f706b3          	sub	a3,a4,a5
   1a280:	18f71e63          	bne	a4,a5,1a41c <__mdiff+0x1c4>
   1a284:	00279693          	slli	a3,a5,0x2
   1a288:	01458613          	addi	a2,a1,20
   1a28c:	01448713          	addi	a4,s1,20
   1a290:	00d607b3          	add	a5,a2,a3
   1a294:	00d70733          	add	a4,a4,a3
   1a298:	0080006f          	j	1a2a0 <__mdiff+0x48>
   1a29c:	18f67c63          	bgeu	a2,a5,1a434 <__mdiff+0x1dc>
   1a2a0:	ffc7a583          	lw	a1,-4(a5)
   1a2a4:	ffc72683          	lw	a3,-4(a4)
   1a2a8:	ffc78793          	addi	a5,a5,-4
   1a2ac:	ffc70713          	addi	a4,a4,-4
   1a2b0:	fed586e3          	beq	a1,a3,1a29c <__mdiff+0x44>
   1a2b4:	00100913          	li	s2,1
   1a2b8:	00d5ea63          	bltu	a1,a3,1a2cc <__mdiff+0x74>
   1a2bc:	00048793          	mv	a5,s1
   1a2c0:	00000913          	li	s2,0
   1a2c4:	00040493          	mv	s1,s0
   1a2c8:	00078413          	mv	s0,a5
   1a2cc:	0044a583          	lw	a1,4(s1)
   1a2d0:	cb8ff0ef          	jal	19788 <_Balloc>
   1a2d4:	1a050663          	beqz	a0,1a480 <__mdiff+0x228>
   1a2d8:	0104a883          	lw	a7,16(s1)
   1a2dc:	01042283          	lw	t0,16(s0)
   1a2e0:	01448f93          	addi	t6,s1,20
   1a2e4:	00289313          	slli	t1,a7,0x2
   1a2e8:	01440813          	addi	a6,s0,20
   1a2ec:	00229293          	slli	t0,t0,0x2
   1a2f0:	01450593          	addi	a1,a0,20
   1a2f4:	00010e37          	lui	t3,0x10
   1a2f8:	01252623          	sw	s2,12(a0)
   1a2fc:	006f8333          	add	t1,t6,t1
   1a300:	005802b3          	add	t0,a6,t0
   1a304:	00058f13          	mv	t5,a1
   1a308:	000f8e93          	mv	t4,t6
   1a30c:	00000693          	li	a3,0
   1a310:	fffe0e13          	addi	t3,t3,-1 # ffff <exit-0xb5>
   1a314:	000ea703          	lw	a4,0(t4)
   1a318:	00082603          	lw	a2,0(a6)
   1a31c:	004f0f13          	addi	t5,t5,4
   1a320:	01c777b3          	and	a5,a4,t3
   1a324:	01c673b3          	and	t2,a2,t3
   1a328:	407787b3          	sub	a5,a5,t2
   1a32c:	00d787b3          	add	a5,a5,a3
   1a330:	01075713          	srli	a4,a4,0x10
   1a334:	01065613          	srli	a2,a2,0x10
   1a338:	4107d693          	srai	a3,a5,0x10
   1a33c:	40c70733          	sub	a4,a4,a2
   1a340:	00d70733          	add	a4,a4,a3
   1a344:	01071693          	slli	a3,a4,0x10
   1a348:	01c7f7b3          	and	a5,a5,t3
   1a34c:	00d7e7b3          	or	a5,a5,a3
   1a350:	00480813          	addi	a6,a6,4
   1a354:	feff2e23          	sw	a5,-4(t5)
   1a358:	004e8e93          	addi	t4,t4,4
   1a35c:	41075693          	srai	a3,a4,0x10
   1a360:	fa586ae3          	bltu	a6,t0,1a314 <__mdiff+0xbc>
   1a364:	01540713          	addi	a4,s0,21
   1a368:	40828433          	sub	s0,t0,s0
   1a36c:	feb40413          	addi	s0,s0,-21
   1a370:	00e2b2b3          	sltu	t0,t0,a4
   1a374:	0012cf13          	xori	t5,t0,1
   1a378:	00245413          	srli	s0,s0,0x2
   1a37c:	00400713          	li	a4,4
   1a380:	0a028463          	beqz	t0,1a428 <__mdiff+0x1d0>
   1a384:	00ef8fb3          	add	t6,t6,a4
   1a388:	00e58833          	add	a6,a1,a4
   1a38c:	00010eb7          	lui	t4,0x10
   1a390:	00080e13          	mv	t3,a6
   1a394:	000f8613          	mv	a2,t6
   1a398:	fffe8e93          	addi	t4,t4,-1 # ffff <exit-0xb5>
   1a39c:	0c6ff463          	bgeu	t6,t1,1a464 <__mdiff+0x20c>
   1a3a0:	00062783          	lw	a5,0(a2)
   1a3a4:	004e0e13          	addi	t3,t3,4
   1a3a8:	00460613          	addi	a2,a2,4
   1a3ac:	01d7f733          	and	a4,a5,t4
   1a3b0:	00d70733          	add	a4,a4,a3
   1a3b4:	0107d593          	srli	a1,a5,0x10
   1a3b8:	41075713          	srai	a4,a4,0x10
   1a3bc:	00b70733          	add	a4,a4,a1
   1a3c0:	00d787b3          	add	a5,a5,a3
   1a3c4:	01d7f7b3          	and	a5,a5,t4
   1a3c8:	01071693          	slli	a3,a4,0x10
   1a3cc:	00d7e7b3          	or	a5,a5,a3
   1a3d0:	fefe2e23          	sw	a5,-4(t3)
   1a3d4:	41075693          	srai	a3,a4,0x10
   1a3d8:	fc6664e3          	bltu	a2,t1,1a3a0 <__mdiff+0x148>
   1a3dc:	fff30313          	addi	t1,t1,-1
   1a3e0:	41f30333          	sub	t1,t1,t6
   1a3e4:	ffc37313          	andi	t1,t1,-4
   1a3e8:	00680733          	add	a4,a6,t1
   1a3ec:	00079a63          	bnez	a5,1a400 <__mdiff+0x1a8>
   1a3f0:	ffc72783          	lw	a5,-4(a4)
   1a3f4:	fff88893          	addi	a7,a7,-1
   1a3f8:	ffc70713          	addi	a4,a4,-4
   1a3fc:	fe078ae3          	beqz	a5,1a3f0 <__mdiff+0x198>
   1a400:	00c12083          	lw	ra,12(sp)
   1a404:	00812403          	lw	s0,8(sp)
   1a408:	01152823          	sw	a7,16(a0)
   1a40c:	00412483          	lw	s1,4(sp)
   1a410:	00012903          	lw	s2,0(sp)
   1a414:	01010113          	addi	sp,sp,16
   1a418:	00008067          	ret
   1a41c:	00100913          	li	s2,1
   1a420:	e806dee3          	bgez	a3,1a2bc <__mdiff+0x64>
   1a424:	ea9ff06f          	j	1a2cc <__mdiff+0x74>
   1a428:	00140713          	addi	a4,s0,1
   1a42c:	00271713          	slli	a4,a4,0x2
   1a430:	f55ff06f          	j	1a384 <__mdiff+0x12c>
   1a434:	00000593          	li	a1,0
   1a438:	b50ff0ef          	jal	19788 <_Balloc>
   1a43c:	06050063          	beqz	a0,1a49c <__mdiff+0x244>
   1a440:	00c12083          	lw	ra,12(sp)
   1a444:	00812403          	lw	s0,8(sp)
   1a448:	00100793          	li	a5,1
   1a44c:	00f52823          	sw	a5,16(a0)
   1a450:	00052a23          	sw	zero,20(a0)
   1a454:	00412483          	lw	s1,4(sp)
   1a458:	00012903          	lw	s2,0(sp)
   1a45c:	01010113          	addi	sp,sp,16
   1a460:	00008067          	ret
   1a464:	00000713          	li	a4,0
   1a468:	000f1663          	bnez	t5,1a474 <__mdiff+0x21c>
   1a46c:	00e58733          	add	a4,a1,a4
   1a470:	f7dff06f          	j	1a3ec <__mdiff+0x194>
   1a474:	00241713          	slli	a4,s0,0x2
   1a478:	00e58733          	add	a4,a1,a4
   1a47c:	f71ff06f          	j	1a3ec <__mdiff+0x194>
   1a480:	00006697          	auipc	a3,0x6
   1a484:	71c68693          	addi	a3,a3,1820 # 20b9c <_exit+0x1ec>
   1a488:	00000613          	li	a2,0
   1a48c:	24500593          	li	a1,581
   1a490:	00006517          	auipc	a0,0x6
   1a494:	72050513          	addi	a0,a0,1824 # 20bb0 <_exit+0x200>
   1a498:	074010ef          	jal	1b50c <__assert_func>
   1a49c:	00006697          	auipc	a3,0x6
   1a4a0:	70068693          	addi	a3,a3,1792 # 20b9c <_exit+0x1ec>
   1a4a4:	00000613          	li	a2,0
   1a4a8:	23700593          	li	a1,567
   1a4ac:	00006517          	auipc	a0,0x6
   1a4b0:	70450513          	addi	a0,a0,1796 # 20bb0 <_exit+0x200>
   1a4b4:	058010ef          	jal	1b50c <__assert_func>

0001a4b8 <__ulp>:
   1a4b8:	7ff007b7          	lui	a5,0x7ff00
   1a4bc:	00b7f5b3          	and	a1,a5,a1
   1a4c0:	fcc007b7          	lui	a5,0xfcc00
   1a4c4:	00f585b3          	add	a1,a1,a5
   1a4c8:	00000793          	li	a5,0
   1a4cc:	00b05663          	blez	a1,1a4d8 <__ulp+0x20>
   1a4d0:	00078513          	mv	a0,a5
   1a4d4:	00008067          	ret
   1a4d8:	40b005b3          	neg	a1,a1
   1a4dc:	4145d593          	srai	a1,a1,0x14
   1a4e0:	01300793          	li	a5,19
   1a4e4:	00b7cc63          	blt	a5,a1,1a4fc <__ulp+0x44>
   1a4e8:	000807b7          	lui	a5,0x80
   1a4ec:	40b7d5b3          	sra	a1,a5,a1
   1a4f0:	00000793          	li	a5,0
   1a4f4:	00078513          	mv	a0,a5
   1a4f8:	00008067          	ret
   1a4fc:	fec58593          	addi	a1,a1,-20
   1a500:	01e00713          	li	a4,30
   1a504:	00100793          	li	a5,1
   1a508:	00b74663          	blt	a4,a1,1a514 <__ulp+0x5c>
   1a50c:	800007b7          	lui	a5,0x80000
   1a510:	00b7d7b3          	srl	a5,a5,a1
   1a514:	00000593          	li	a1,0
   1a518:	00078513          	mv	a0,a5
   1a51c:	00008067          	ret

0001a520 <__b2d>:
   1a520:	fe010113          	addi	sp,sp,-32
   1a524:	00912a23          	sw	s1,20(sp)
   1a528:	01052483          	lw	s1,16(a0)
   1a52c:	00812c23          	sw	s0,24(sp)
   1a530:	01450413          	addi	s0,a0,20
   1a534:	00249493          	slli	s1,s1,0x2
   1a538:	009404b3          	add	s1,s0,s1
   1a53c:	01212823          	sw	s2,16(sp)
   1a540:	ffc4a903          	lw	s2,-4(s1)
   1a544:	01312623          	sw	s3,12(sp)
   1a548:	01412423          	sw	s4,8(sp)
   1a54c:	00090513          	mv	a0,s2
   1a550:	00058993          	mv	s3,a1
   1a554:	00112e23          	sw	ra,28(sp)
   1a558:	d8cff0ef          	jal	19ae4 <__hi0bits>
   1a55c:	02000713          	li	a4,32
   1a560:	40a707b3          	sub	a5,a4,a0
   1a564:	00f9a023          	sw	a5,0(s3)
   1a568:	00a00793          	li	a5,10
   1a56c:	ffc48a13          	addi	s4,s1,-4
   1a570:	08a7dc63          	bge	a5,a0,1a608 <__b2d+0xe8>
   1a574:	ff550613          	addi	a2,a0,-11
   1a578:	05447063          	bgeu	s0,s4,1a5b8 <__b2d+0x98>
   1a57c:	ff84a783          	lw	a5,-8(s1)
   1a580:	04060c63          	beqz	a2,1a5d8 <__b2d+0xb8>
   1a584:	40c706b3          	sub	a3,a4,a2
   1a588:	00d7d733          	srl	a4,a5,a3
   1a58c:	00c91933          	sll	s2,s2,a2
   1a590:	00e96933          	or	s2,s2,a4
   1a594:	ff848593          	addi	a1,s1,-8
   1a598:	3ff00737          	lui	a4,0x3ff00
   1a59c:	00e96733          	or	a4,s2,a4
   1a5a0:	00c797b3          	sll	a5,a5,a2
   1a5a4:	02b47e63          	bgeu	s0,a1,1a5e0 <__b2d+0xc0>
   1a5a8:	ff44a603          	lw	a2,-12(s1)
   1a5ac:	00d656b3          	srl	a3,a2,a3
   1a5b0:	00d7e7b3          	or	a5,a5,a3
   1a5b4:	02c0006f          	j	1a5e0 <__b2d+0xc0>
   1a5b8:	00b00793          	li	a5,11
   1a5bc:	00f50c63          	beq	a0,a5,1a5d4 <__b2d+0xb4>
   1a5c0:	00c91933          	sll	s2,s2,a2
   1a5c4:	3ff00737          	lui	a4,0x3ff00
   1a5c8:	00e96733          	or	a4,s2,a4
   1a5cc:	00000793          	li	a5,0
   1a5d0:	0100006f          	j	1a5e0 <__b2d+0xc0>
   1a5d4:	00000793          	li	a5,0
   1a5d8:	3ff00737          	lui	a4,0x3ff00
   1a5dc:	00e96733          	or	a4,s2,a4
   1a5e0:	01c12083          	lw	ra,28(sp)
   1a5e4:	01812403          	lw	s0,24(sp)
   1a5e8:	01412483          	lw	s1,20(sp)
   1a5ec:	01012903          	lw	s2,16(sp)
   1a5f0:	00c12983          	lw	s3,12(sp)
   1a5f4:	00812a03          	lw	s4,8(sp)
   1a5f8:	00078513          	mv	a0,a5
   1a5fc:	00070593          	mv	a1,a4
   1a600:	02010113          	addi	sp,sp,32
   1a604:	00008067          	ret
   1a608:	00b00693          	li	a3,11
   1a60c:	40a686b3          	sub	a3,a3,a0
   1a610:	3ff007b7          	lui	a5,0x3ff00
   1a614:	00d95733          	srl	a4,s2,a3
   1a618:	00f76733          	or	a4,a4,a5
   1a61c:	00000793          	li	a5,0
   1a620:	01447663          	bgeu	s0,s4,1a62c <__b2d+0x10c>
   1a624:	ff84a783          	lw	a5,-8(s1)
   1a628:	00d7d7b3          	srl	a5,a5,a3
   1a62c:	01550513          	addi	a0,a0,21
   1a630:	00a91933          	sll	s2,s2,a0
   1a634:	00f967b3          	or	a5,s2,a5
   1a638:	fa9ff06f          	j	1a5e0 <__b2d+0xc0>

0001a63c <__d2b>:
   1a63c:	fd010113          	addi	sp,sp,-48
   1a640:	01512a23          	sw	s5,20(sp)
   1a644:	00058a93          	mv	s5,a1
   1a648:	00100593          	li	a1,1
   1a64c:	02912223          	sw	s1,36(sp)
   1a650:	01312e23          	sw	s3,28(sp)
   1a654:	01412c23          	sw	s4,24(sp)
   1a658:	02112623          	sw	ra,44(sp)
   1a65c:	02812423          	sw	s0,40(sp)
   1a660:	03212023          	sw	s2,32(sp)
   1a664:	00060493          	mv	s1,a2
   1a668:	00068a13          	mv	s4,a3
   1a66c:	00070993          	mv	s3,a4
   1a670:	918ff0ef          	jal	19788 <_Balloc>
   1a674:	10050263          	beqz	a0,1a778 <__d2b+0x13c>
   1a678:	00100737          	lui	a4,0x100
   1a67c:	0144d913          	srli	s2,s1,0x14
   1a680:	fff70793          	addi	a5,a4,-1 # fffff <__BSS_END__+0xdd2cf>
   1a684:	7ff97913          	andi	s2,s2,2047
   1a688:	00050413          	mv	s0,a0
   1a68c:	0097f7b3          	and	a5,a5,s1
   1a690:	00090463          	beqz	s2,1a698 <__d2b+0x5c>
   1a694:	00e7e7b3          	or	a5,a5,a4
   1a698:	00f12623          	sw	a5,12(sp)
   1a69c:	060a9263          	bnez	s5,1a700 <__d2b+0xc4>
   1a6a0:	00c10513          	addi	a0,sp,12
   1a6a4:	cb8ff0ef          	jal	19b5c <__lo0bits>
   1a6a8:	00c12703          	lw	a4,12(sp)
   1a6ac:	00100493          	li	s1,1
   1a6b0:	00942823          	sw	s1,16(s0)
   1a6b4:	00e42a23          	sw	a4,20(s0)
   1a6b8:	02050793          	addi	a5,a0,32
   1a6bc:	08090863          	beqz	s2,1a74c <__d2b+0x110>
   1a6c0:	bcd90913          	addi	s2,s2,-1075
   1a6c4:	00f90933          	add	s2,s2,a5
   1a6c8:	03500493          	li	s1,53
   1a6cc:	012a2023          	sw	s2,0(s4)
   1a6d0:	40f48533          	sub	a0,s1,a5
   1a6d4:	00a9a023          	sw	a0,0(s3)
   1a6d8:	02c12083          	lw	ra,44(sp)
   1a6dc:	00040513          	mv	a0,s0
   1a6e0:	02812403          	lw	s0,40(sp)
   1a6e4:	02412483          	lw	s1,36(sp)
   1a6e8:	02012903          	lw	s2,32(sp)
   1a6ec:	01c12983          	lw	s3,28(sp)
   1a6f0:	01812a03          	lw	s4,24(sp)
   1a6f4:	01412a83          	lw	s5,20(sp)
   1a6f8:	03010113          	addi	sp,sp,48
   1a6fc:	00008067          	ret
   1a700:	00810513          	addi	a0,sp,8
   1a704:	01512423          	sw	s5,8(sp)
   1a708:	c54ff0ef          	jal	19b5c <__lo0bits>
   1a70c:	00c12703          	lw	a4,12(sp)
   1a710:	00050793          	mv	a5,a0
   1a714:	04050e63          	beqz	a0,1a770 <__d2b+0x134>
   1a718:	00812603          	lw	a2,8(sp)
   1a71c:	02000693          	li	a3,32
   1a720:	40a686b3          	sub	a3,a3,a0
   1a724:	00d716b3          	sll	a3,a4,a3
   1a728:	00a75733          	srl	a4,a4,a0
   1a72c:	00c6e6b3          	or	a3,a3,a2
   1a730:	00e12623          	sw	a4,12(sp)
   1a734:	00e034b3          	snez	s1,a4
   1a738:	00148493          	addi	s1,s1,1
   1a73c:	00d42a23          	sw	a3,20(s0)
   1a740:	00e42c23          	sw	a4,24(s0)
   1a744:	00942823          	sw	s1,16(s0)
   1a748:	f6091ce3          	bnez	s2,1a6c0 <__d2b+0x84>
   1a74c:	00249713          	slli	a4,s1,0x2
   1a750:	00e40733          	add	a4,s0,a4
   1a754:	01072503          	lw	a0,16(a4)
   1a758:	bce78793          	addi	a5,a5,-1074 # 3feffbce <__BSS_END__+0x3fedce9e>
   1a75c:	00fa2023          	sw	a5,0(s4)
   1a760:	b84ff0ef          	jal	19ae4 <__hi0bits>
   1a764:	00549493          	slli	s1,s1,0x5
   1a768:	40a48533          	sub	a0,s1,a0
   1a76c:	f69ff06f          	j	1a6d4 <__d2b+0x98>
   1a770:	00812683          	lw	a3,8(sp)
   1a774:	fc1ff06f          	j	1a734 <__d2b+0xf8>
   1a778:	00006697          	auipc	a3,0x6
   1a77c:	42468693          	addi	a3,a3,1060 # 20b9c <_exit+0x1ec>
   1a780:	00000613          	li	a2,0
   1a784:	30f00593          	li	a1,783
   1a788:	00006517          	auipc	a0,0x6
   1a78c:	42850513          	addi	a0,a0,1064 # 20bb0 <_exit+0x200>
   1a790:	57d000ef          	jal	1b50c <__assert_func>

0001a794 <__ratio>:
   1a794:	fd010113          	addi	sp,sp,-48
   1a798:	03212023          	sw	s2,32(sp)
   1a79c:	00058913          	mv	s2,a1
   1a7a0:	00810593          	addi	a1,sp,8
   1a7a4:	02112623          	sw	ra,44(sp)
   1a7a8:	02812423          	sw	s0,40(sp)
   1a7ac:	02912223          	sw	s1,36(sp)
   1a7b0:	01312e23          	sw	s3,28(sp)
   1a7b4:	00050993          	mv	s3,a0
   1a7b8:	d69ff0ef          	jal	1a520 <__b2d>
   1a7bc:	00050493          	mv	s1,a0
   1a7c0:	00058413          	mv	s0,a1
   1a7c4:	00090513          	mv	a0,s2
   1a7c8:	00c10593          	addi	a1,sp,12
   1a7cc:	d55ff0ef          	jal	1a520 <__b2d>
   1a7d0:	01092703          	lw	a4,16(s2)
   1a7d4:	0109a783          	lw	a5,16(s3)
   1a7d8:	00c12683          	lw	a3,12(sp)
   1a7dc:	40e787b3          	sub	a5,a5,a4
   1a7e0:	00812703          	lw	a4,8(sp)
   1a7e4:	00579793          	slli	a5,a5,0x5
   1a7e8:	40d70733          	sub	a4,a4,a3
   1a7ec:	00e787b3          	add	a5,a5,a4
   1a7f0:	00050713          	mv	a4,a0
   1a7f4:	02f05e63          	blez	a5,1a830 <__ratio+0x9c>
   1a7f8:	01479793          	slli	a5,a5,0x14
   1a7fc:	00878433          	add	s0,a5,s0
   1a800:	00058693          	mv	a3,a1
   1a804:	00048513          	mv	a0,s1
   1a808:	00040593          	mv	a1,s0
   1a80c:	00070613          	mv	a2,a4
   1a810:	41d010ef          	jal	1c42c <__divdf3>
   1a814:	02c12083          	lw	ra,44(sp)
   1a818:	02812403          	lw	s0,40(sp)
   1a81c:	02412483          	lw	s1,36(sp)
   1a820:	02012903          	lw	s2,32(sp)
   1a824:	01c12983          	lw	s3,28(sp)
   1a828:	03010113          	addi	sp,sp,48
   1a82c:	00008067          	ret
   1a830:	01479793          	slli	a5,a5,0x14
   1a834:	40f585b3          	sub	a1,a1,a5
   1a838:	fc9ff06f          	j	1a800 <__ratio+0x6c>

0001a83c <_mprec_log10>:
   1a83c:	ff010113          	addi	sp,sp,-16
   1a840:	01212023          	sw	s2,0(sp)
   1a844:	00112623          	sw	ra,12(sp)
   1a848:	01700793          	li	a5,23
   1a84c:	00050913          	mv	s2,a0
   1a850:	06a7d263          	bge	a5,a0,1a8b4 <_mprec_log10+0x78>
   1a854:	00008717          	auipc	a4,0x8
   1a858:	08470713          	addi	a4,a4,132 # 228d8 <__SDATA_BEGIN__+0x30>
   1a85c:	00072783          	lw	a5,0(a4)
   1a860:	00472583          	lw	a1,4(a4)
   1a864:	00008717          	auipc	a4,0x8
   1a868:	07c70713          	addi	a4,a4,124 # 228e0 <__SDATA_BEGIN__+0x38>
   1a86c:	00812423          	sw	s0,8(sp)
   1a870:	00912223          	sw	s1,4(sp)
   1a874:	00072403          	lw	s0,0(a4)
   1a878:	00472483          	lw	s1,4(a4)
   1a87c:	00078513          	mv	a0,a5
   1a880:	00040613          	mv	a2,s0
   1a884:	00048693          	mv	a3,s1
   1a888:	4c8020ef          	jal	1cd50 <__muldf3>
   1a88c:	fff90913          	addi	s2,s2,-1
   1a890:	00050793          	mv	a5,a0
   1a894:	fe0914e3          	bnez	s2,1a87c <_mprec_log10+0x40>
   1a898:	00812403          	lw	s0,8(sp)
   1a89c:	00c12083          	lw	ra,12(sp)
   1a8a0:	00412483          	lw	s1,4(sp)
   1a8a4:	00012903          	lw	s2,0(sp)
   1a8a8:	00078513          	mv	a0,a5
   1a8ac:	01010113          	addi	sp,sp,16
   1a8b0:	00008067          	ret
   1a8b4:	00351913          	slli	s2,a0,0x3
   1a8b8:	00007717          	auipc	a4,0x7
   1a8bc:	85870713          	addi	a4,a4,-1960 # 21110 <__mprec_tens>
   1a8c0:	01270733          	add	a4,a4,s2
   1a8c4:	00072783          	lw	a5,0(a4)
   1a8c8:	00c12083          	lw	ra,12(sp)
   1a8cc:	00472583          	lw	a1,4(a4)
   1a8d0:	00012903          	lw	s2,0(sp)
   1a8d4:	00078513          	mv	a0,a5
   1a8d8:	01010113          	addi	sp,sp,16
   1a8dc:	00008067          	ret

0001a8e0 <__copybits>:
   1a8e0:	01062683          	lw	a3,16(a2)
   1a8e4:	fff58593          	addi	a1,a1,-1
   1a8e8:	4055d593          	srai	a1,a1,0x5
   1a8ec:	00158593          	addi	a1,a1,1
   1a8f0:	01460793          	addi	a5,a2,20
   1a8f4:	00269693          	slli	a3,a3,0x2
   1a8f8:	00259593          	slli	a1,a1,0x2
   1a8fc:	00d786b3          	add	a3,a5,a3
   1a900:	00b505b3          	add	a1,a0,a1
   1a904:	02d7f863          	bgeu	a5,a3,1a934 <__copybits+0x54>
   1a908:	00050713          	mv	a4,a0
   1a90c:	0007a803          	lw	a6,0(a5)
   1a910:	00478793          	addi	a5,a5,4
   1a914:	00470713          	addi	a4,a4,4
   1a918:	ff072e23          	sw	a6,-4(a4)
   1a91c:	fed7e8e3          	bltu	a5,a3,1a90c <__copybits+0x2c>
   1a920:	40c687b3          	sub	a5,a3,a2
   1a924:	feb78793          	addi	a5,a5,-21
   1a928:	ffc7f793          	andi	a5,a5,-4
   1a92c:	00478793          	addi	a5,a5,4
   1a930:	00f50533          	add	a0,a0,a5
   1a934:	00b57863          	bgeu	a0,a1,1a944 <__copybits+0x64>
   1a938:	00450513          	addi	a0,a0,4
   1a93c:	fe052e23          	sw	zero,-4(a0)
   1a940:	feb56ce3          	bltu	a0,a1,1a938 <__copybits+0x58>
   1a944:	00008067          	ret

0001a948 <__any_on>:
   1a948:	01052703          	lw	a4,16(a0)
   1a94c:	4055d613          	srai	a2,a1,0x5
   1a950:	01450693          	addi	a3,a0,20
   1a954:	02c75263          	bge	a4,a2,1a978 <__any_on+0x30>
   1a958:	00271713          	slli	a4,a4,0x2
   1a95c:	00e687b3          	add	a5,a3,a4
   1a960:	04f6f263          	bgeu	a3,a5,1a9a4 <__any_on+0x5c>
   1a964:	ffc7a703          	lw	a4,-4(a5)
   1a968:	ffc78793          	addi	a5,a5,-4
   1a96c:	fe070ae3          	beqz	a4,1a960 <__any_on+0x18>
   1a970:	00100513          	li	a0,1
   1a974:	00008067          	ret
   1a978:	00261793          	slli	a5,a2,0x2
   1a97c:	00f687b3          	add	a5,a3,a5
   1a980:	fee650e3          	bge	a2,a4,1a960 <__any_on+0x18>
   1a984:	01f5f593          	andi	a1,a1,31
   1a988:	fc058ce3          	beqz	a1,1a960 <__any_on+0x18>
   1a98c:	0007a603          	lw	a2,0(a5)
   1a990:	00100513          	li	a0,1
   1a994:	00b65733          	srl	a4,a2,a1
   1a998:	00b71733          	sll	a4,a4,a1
   1a99c:	fce602e3          	beq	a2,a4,1a960 <__any_on+0x18>
   1a9a0:	00008067          	ret
   1a9a4:	00000513          	li	a0,0
   1a9a8:	00008067          	ret

0001a9ac <_realloc_r>:
   1a9ac:	fd010113          	addi	sp,sp,-48
   1a9b0:	02912223          	sw	s1,36(sp)
   1a9b4:	02112623          	sw	ra,44(sp)
   1a9b8:	00060493          	mv	s1,a2
   1a9bc:	1e058863          	beqz	a1,1abac <_realloc_r+0x200>
   1a9c0:	02812423          	sw	s0,40(sp)
   1a9c4:	03212023          	sw	s2,32(sp)
   1a9c8:	00058413          	mv	s0,a1
   1a9cc:	01312e23          	sw	s3,28(sp)
   1a9d0:	01512a23          	sw	s5,20(sp)
   1a9d4:	01412c23          	sw	s4,24(sp)
   1a9d8:	00050913          	mv	s2,a0
   1a9dc:	b5cf70ef          	jal	11d38 <__malloc_lock>
   1a9e0:	ffc42703          	lw	a4,-4(s0)
   1a9e4:	00b48793          	addi	a5,s1,11
   1a9e8:	01600693          	li	a3,22
   1a9ec:	ff840a93          	addi	s5,s0,-8
   1a9f0:	ffc77993          	andi	s3,a4,-4
   1a9f4:	10f6f263          	bgeu	a3,a5,1aaf8 <_realloc_r+0x14c>
   1a9f8:	ff87fa13          	andi	s4,a5,-8
   1a9fc:	1007c263          	bltz	a5,1ab00 <_realloc_r+0x154>
   1aa00:	109a6063          	bltu	s4,s1,1ab00 <_realloc_r+0x154>
   1aa04:	1349d263          	bge	s3,s4,1ab28 <_realloc_r+0x17c>
   1aa08:	01812423          	sw	s8,8(sp)
   1aa0c:	00008c17          	auipc	s8,0x8
   1aa10:	924c0c13          	addi	s8,s8,-1756 # 22330 <__malloc_av_>
   1aa14:	008c2603          	lw	a2,8(s8)
   1aa18:	013a86b3          	add	a3,s5,s3
   1aa1c:	0046a783          	lw	a5,4(a3)
   1aa20:	1cd60863          	beq	a2,a3,1abf0 <_realloc_r+0x244>
   1aa24:	ffe7f613          	andi	a2,a5,-2
   1aa28:	00c68633          	add	a2,a3,a2
   1aa2c:	00462603          	lw	a2,4(a2)
   1aa30:	00167613          	andi	a2,a2,1
   1aa34:	14061a63          	bnez	a2,1ab88 <_realloc_r+0x1dc>
   1aa38:	ffc7f793          	andi	a5,a5,-4
   1aa3c:	00f98633          	add	a2,s3,a5
   1aa40:	0d465863          	bge	a2,s4,1ab10 <_realloc_r+0x164>
   1aa44:	00177713          	andi	a4,a4,1
   1aa48:	02071c63          	bnez	a4,1aa80 <_realloc_r+0xd4>
   1aa4c:	01712623          	sw	s7,12(sp)
   1aa50:	ff842b83          	lw	s7,-8(s0)
   1aa54:	01612823          	sw	s6,16(sp)
   1aa58:	417a8bb3          	sub	s7,s5,s7
   1aa5c:	004ba703          	lw	a4,4(s7)
   1aa60:	ffc77713          	andi	a4,a4,-4
   1aa64:	00e787b3          	add	a5,a5,a4
   1aa68:	01378b33          	add	s6,a5,s3
   1aa6c:	334b5c63          	bge	s6,s4,1ada4 <_realloc_r+0x3f8>
   1aa70:	00e98b33          	add	s6,s3,a4
   1aa74:	294b5863          	bge	s6,s4,1ad04 <_realloc_r+0x358>
   1aa78:	01012b03          	lw	s6,16(sp)
   1aa7c:	00c12b83          	lw	s7,12(sp)
   1aa80:	00048593          	mv	a1,s1
   1aa84:	00090513          	mv	a0,s2
   1aa88:	ae9f60ef          	jal	11570 <_malloc_r>
   1aa8c:	00050493          	mv	s1,a0
   1aa90:	40050863          	beqz	a0,1aea0 <_realloc_r+0x4f4>
   1aa94:	ffc42783          	lw	a5,-4(s0)
   1aa98:	ff850713          	addi	a4,a0,-8
   1aa9c:	ffe7f793          	andi	a5,a5,-2
   1aaa0:	00fa87b3          	add	a5,s5,a5
   1aaa4:	24e78663          	beq	a5,a4,1acf0 <_realloc_r+0x344>
   1aaa8:	ffc98613          	addi	a2,s3,-4
   1aaac:	02400793          	li	a5,36
   1aab0:	2ec7e463          	bltu	a5,a2,1ad98 <_realloc_r+0x3ec>
   1aab4:	01300713          	li	a4,19
   1aab8:	20c76a63          	bltu	a4,a2,1accc <_realloc_r+0x320>
   1aabc:	00050793          	mv	a5,a0
   1aac0:	00040713          	mv	a4,s0
   1aac4:	00072683          	lw	a3,0(a4)
   1aac8:	00d7a023          	sw	a3,0(a5)
   1aacc:	00472683          	lw	a3,4(a4)
   1aad0:	00d7a223          	sw	a3,4(a5)
   1aad4:	00872703          	lw	a4,8(a4)
   1aad8:	00e7a423          	sw	a4,8(a5)
   1aadc:	00040593          	mv	a1,s0
   1aae0:	00090513          	mv	a0,s2
   1aae4:	f88f60ef          	jal	1126c <_free_r>
   1aae8:	00090513          	mv	a0,s2
   1aaec:	a50f70ef          	jal	11d3c <__malloc_unlock>
   1aaf0:	00812c03          	lw	s8,8(sp)
   1aaf4:	06c0006f          	j	1ab60 <_realloc_r+0x1b4>
   1aaf8:	01000a13          	li	s4,16
   1aafc:	f09a74e3          	bgeu	s4,s1,1aa04 <_realloc_r+0x58>
   1ab00:	00c00793          	li	a5,12
   1ab04:	00f92023          	sw	a5,0(s2)
   1ab08:	00000493          	li	s1,0
   1ab0c:	0540006f          	j	1ab60 <_realloc_r+0x1b4>
   1ab10:	00c6a783          	lw	a5,12(a3)
   1ab14:	0086a703          	lw	a4,8(a3)
   1ab18:	00812c03          	lw	s8,8(sp)
   1ab1c:	00060993          	mv	s3,a2
   1ab20:	00f72623          	sw	a5,12(a4)
   1ab24:	00e7a423          	sw	a4,8(a5)
   1ab28:	004aa783          	lw	a5,4(s5)
   1ab2c:	414986b3          	sub	a3,s3,s4
   1ab30:	00f00613          	li	a2,15
   1ab34:	0017f793          	andi	a5,a5,1
   1ab38:	013a8733          	add	a4,s5,s3
   1ab3c:	08d66263          	bltu	a2,a3,1abc0 <_realloc_r+0x214>
   1ab40:	0137e7b3          	or	a5,a5,s3
   1ab44:	00faa223          	sw	a5,4(s5)
   1ab48:	00472783          	lw	a5,4(a4)
   1ab4c:	0017e793          	ori	a5,a5,1
   1ab50:	00f72223          	sw	a5,4(a4)
   1ab54:	00090513          	mv	a0,s2
   1ab58:	9e4f70ef          	jal	11d3c <__malloc_unlock>
   1ab5c:	00040493          	mv	s1,s0
   1ab60:	02812403          	lw	s0,40(sp)
   1ab64:	02c12083          	lw	ra,44(sp)
   1ab68:	02012903          	lw	s2,32(sp)
   1ab6c:	01c12983          	lw	s3,28(sp)
   1ab70:	01812a03          	lw	s4,24(sp)
   1ab74:	01412a83          	lw	s5,20(sp)
   1ab78:	00048513          	mv	a0,s1
   1ab7c:	02412483          	lw	s1,36(sp)
   1ab80:	03010113          	addi	sp,sp,48
   1ab84:	00008067          	ret
   1ab88:	00177713          	andi	a4,a4,1
   1ab8c:	ee071ae3          	bnez	a4,1aa80 <_realloc_r+0xd4>
   1ab90:	01712623          	sw	s7,12(sp)
   1ab94:	ff842b83          	lw	s7,-8(s0)
   1ab98:	01612823          	sw	s6,16(sp)
   1ab9c:	417a8bb3          	sub	s7,s5,s7
   1aba0:	004ba703          	lw	a4,4(s7)
   1aba4:	ffc77713          	andi	a4,a4,-4
   1aba8:	ec9ff06f          	j	1aa70 <_realloc_r+0xc4>
   1abac:	02c12083          	lw	ra,44(sp)
   1abb0:	02412483          	lw	s1,36(sp)
   1abb4:	00060593          	mv	a1,a2
   1abb8:	03010113          	addi	sp,sp,48
   1abbc:	9b5f606f          	j	11570 <_malloc_r>
   1abc0:	0147e7b3          	or	a5,a5,s4
   1abc4:	00faa223          	sw	a5,4(s5)
   1abc8:	014a85b3          	add	a1,s5,s4
   1abcc:	0016e693          	ori	a3,a3,1
   1abd0:	00d5a223          	sw	a3,4(a1)
   1abd4:	00472783          	lw	a5,4(a4)
   1abd8:	00858593          	addi	a1,a1,8
   1abdc:	00090513          	mv	a0,s2
   1abe0:	0017e793          	ori	a5,a5,1
   1abe4:	00f72223          	sw	a5,4(a4)
   1abe8:	e84f60ef          	jal	1126c <_free_r>
   1abec:	f69ff06f          	j	1ab54 <_realloc_r+0x1a8>
   1abf0:	ffc7f793          	andi	a5,a5,-4
   1abf4:	013786b3          	add	a3,a5,s3
   1abf8:	010a0613          	addi	a2,s4,16
   1abfc:	26c6d063          	bge	a3,a2,1ae5c <_realloc_r+0x4b0>
   1ac00:	00177713          	andi	a4,a4,1
   1ac04:	e6071ee3          	bnez	a4,1aa80 <_realloc_r+0xd4>
   1ac08:	01712623          	sw	s7,12(sp)
   1ac0c:	ff842b83          	lw	s7,-8(s0)
   1ac10:	01612823          	sw	s6,16(sp)
   1ac14:	417a8bb3          	sub	s7,s5,s7
   1ac18:	004ba703          	lw	a4,4(s7)
   1ac1c:	ffc77713          	andi	a4,a4,-4
   1ac20:	00e787b3          	add	a5,a5,a4
   1ac24:	01378b33          	add	s6,a5,s3
   1ac28:	e4cb44e3          	blt	s6,a2,1aa70 <_realloc_r+0xc4>
   1ac2c:	00cba783          	lw	a5,12(s7)
   1ac30:	008ba703          	lw	a4,8(s7)
   1ac34:	ffc98613          	addi	a2,s3,-4
   1ac38:	02400693          	li	a3,36
   1ac3c:	00f72623          	sw	a5,12(a4)
   1ac40:	00e7a423          	sw	a4,8(a5)
   1ac44:	008b8493          	addi	s1,s7,8
   1ac48:	28c6e463          	bltu	a3,a2,1aed0 <_realloc_r+0x524>
   1ac4c:	01300713          	li	a4,19
   1ac50:	00048793          	mv	a5,s1
   1ac54:	02c77263          	bgeu	a4,a2,1ac78 <_realloc_r+0x2cc>
   1ac58:	00042703          	lw	a4,0(s0)
   1ac5c:	01b00793          	li	a5,27
   1ac60:	00eba423          	sw	a4,8(s7)
   1ac64:	00442703          	lw	a4,4(s0)
   1ac68:	00eba623          	sw	a4,12(s7)
   1ac6c:	26c7ea63          	bltu	a5,a2,1aee0 <_realloc_r+0x534>
   1ac70:	00840413          	addi	s0,s0,8
   1ac74:	010b8793          	addi	a5,s7,16
   1ac78:	00042703          	lw	a4,0(s0)
   1ac7c:	00e7a023          	sw	a4,0(a5)
   1ac80:	00442703          	lw	a4,4(s0)
   1ac84:	00e7a223          	sw	a4,4(a5)
   1ac88:	00842703          	lw	a4,8(s0)
   1ac8c:	00e7a423          	sw	a4,8(a5)
   1ac90:	014b8733          	add	a4,s7,s4
   1ac94:	414b07b3          	sub	a5,s6,s4
   1ac98:	00ec2423          	sw	a4,8(s8)
   1ac9c:	0017e793          	ori	a5,a5,1
   1aca0:	00f72223          	sw	a5,4(a4)
   1aca4:	004ba783          	lw	a5,4(s7)
   1aca8:	00090513          	mv	a0,s2
   1acac:	0017f793          	andi	a5,a5,1
   1acb0:	0147e7b3          	or	a5,a5,s4
   1acb4:	00fba223          	sw	a5,4(s7)
   1acb8:	884f70ef          	jal	11d3c <__malloc_unlock>
   1acbc:	01012b03          	lw	s6,16(sp)
   1acc0:	00c12b83          	lw	s7,12(sp)
   1acc4:	00812c03          	lw	s8,8(sp)
   1acc8:	e99ff06f          	j	1ab60 <_realloc_r+0x1b4>
   1accc:	00042683          	lw	a3,0(s0)
   1acd0:	01b00713          	li	a4,27
   1acd4:	00d52023          	sw	a3,0(a0)
   1acd8:	00442683          	lw	a3,4(s0)
   1acdc:	00d52223          	sw	a3,4(a0)
   1ace0:	14c76e63          	bltu	a4,a2,1ae3c <_realloc_r+0x490>
   1ace4:	00840713          	addi	a4,s0,8
   1ace8:	00850793          	addi	a5,a0,8
   1acec:	dd9ff06f          	j	1aac4 <_realloc_r+0x118>
   1acf0:	ffc52783          	lw	a5,-4(a0)
   1acf4:	00812c03          	lw	s8,8(sp)
   1acf8:	ffc7f793          	andi	a5,a5,-4
   1acfc:	00f989b3          	add	s3,s3,a5
   1ad00:	e29ff06f          	j	1ab28 <_realloc_r+0x17c>
   1ad04:	00cba783          	lw	a5,12(s7)
   1ad08:	008ba703          	lw	a4,8(s7)
   1ad0c:	ffc98613          	addi	a2,s3,-4
   1ad10:	02400693          	li	a3,36
   1ad14:	00f72623          	sw	a5,12(a4)
   1ad18:	00e7a423          	sw	a4,8(a5)
   1ad1c:	008b8493          	addi	s1,s7,8
   1ad20:	10c6e663          	bltu	a3,a2,1ae2c <_realloc_r+0x480>
   1ad24:	01300713          	li	a4,19
   1ad28:	00048793          	mv	a5,s1
   1ad2c:	02c77c63          	bgeu	a4,a2,1ad64 <_realloc_r+0x3b8>
   1ad30:	00042703          	lw	a4,0(s0)
   1ad34:	01b00793          	li	a5,27
   1ad38:	00eba423          	sw	a4,8(s7)
   1ad3c:	00442703          	lw	a4,4(s0)
   1ad40:	00eba623          	sw	a4,12(s7)
   1ad44:	14c7f863          	bgeu	a5,a2,1ae94 <_realloc_r+0x4e8>
   1ad48:	00842783          	lw	a5,8(s0)
   1ad4c:	00fba823          	sw	a5,16(s7)
   1ad50:	00c42783          	lw	a5,12(s0)
   1ad54:	00fbaa23          	sw	a5,20(s7)
   1ad58:	0ad60c63          	beq	a2,a3,1ae10 <_realloc_r+0x464>
   1ad5c:	01040413          	addi	s0,s0,16
   1ad60:	018b8793          	addi	a5,s7,24
   1ad64:	00042703          	lw	a4,0(s0)
   1ad68:	00e7a023          	sw	a4,0(a5)
   1ad6c:	00442703          	lw	a4,4(s0)
   1ad70:	00e7a223          	sw	a4,4(a5)
   1ad74:	00842703          	lw	a4,8(s0)
   1ad78:	00e7a423          	sw	a4,8(a5)
   1ad7c:	000b0993          	mv	s3,s6
   1ad80:	000b8a93          	mv	s5,s7
   1ad84:	01012b03          	lw	s6,16(sp)
   1ad88:	00c12b83          	lw	s7,12(sp)
   1ad8c:	00812c03          	lw	s8,8(sp)
   1ad90:	00048413          	mv	s0,s1
   1ad94:	d95ff06f          	j	1ab28 <_realloc_r+0x17c>
   1ad98:	00040593          	mv	a1,s0
   1ad9c:	e49fb0ef          	jal	16be4 <memmove>
   1ada0:	d3dff06f          	j	1aadc <_realloc_r+0x130>
   1ada4:	00c6a783          	lw	a5,12(a3)
   1ada8:	0086a703          	lw	a4,8(a3)
   1adac:	ffc98613          	addi	a2,s3,-4
   1adb0:	02400693          	li	a3,36
   1adb4:	00f72623          	sw	a5,12(a4)
   1adb8:	00e7a423          	sw	a4,8(a5)
   1adbc:	008ba703          	lw	a4,8(s7)
   1adc0:	00cba783          	lw	a5,12(s7)
   1adc4:	008b8493          	addi	s1,s7,8
   1adc8:	00f72623          	sw	a5,12(a4)
   1adcc:	00e7a423          	sw	a4,8(a5)
   1add0:	04c6ee63          	bltu	a3,a2,1ae2c <_realloc_r+0x480>
   1add4:	01300713          	li	a4,19
   1add8:	00048793          	mv	a5,s1
   1addc:	f8c774e3          	bgeu	a4,a2,1ad64 <_realloc_r+0x3b8>
   1ade0:	00042703          	lw	a4,0(s0)
   1ade4:	01b00793          	li	a5,27
   1ade8:	00eba423          	sw	a4,8(s7)
   1adec:	00442703          	lw	a4,4(s0)
   1adf0:	00eba623          	sw	a4,12(s7)
   1adf4:	0ac7f063          	bgeu	a5,a2,1ae94 <_realloc_r+0x4e8>
   1adf8:	00842703          	lw	a4,8(s0)
   1adfc:	02400793          	li	a5,36
   1ae00:	00eba823          	sw	a4,16(s7)
   1ae04:	00c42703          	lw	a4,12(s0)
   1ae08:	00ebaa23          	sw	a4,20(s7)
   1ae0c:	f4f618e3          	bne	a2,a5,1ad5c <_realloc_r+0x3b0>
   1ae10:	01042703          	lw	a4,16(s0)
   1ae14:	020b8793          	addi	a5,s7,32
   1ae18:	01840413          	addi	s0,s0,24
   1ae1c:	00ebac23          	sw	a4,24(s7)
   1ae20:	ffc42703          	lw	a4,-4(s0)
   1ae24:	00ebae23          	sw	a4,28(s7)
   1ae28:	f3dff06f          	j	1ad64 <_realloc_r+0x3b8>
   1ae2c:	00040593          	mv	a1,s0
   1ae30:	00048513          	mv	a0,s1
   1ae34:	db1fb0ef          	jal	16be4 <memmove>
   1ae38:	f45ff06f          	j	1ad7c <_realloc_r+0x3d0>
   1ae3c:	00842703          	lw	a4,8(s0)
   1ae40:	00e52423          	sw	a4,8(a0)
   1ae44:	00c42703          	lw	a4,12(s0)
   1ae48:	00e52623          	sw	a4,12(a0)
   1ae4c:	06f60463          	beq	a2,a5,1aeb4 <_realloc_r+0x508>
   1ae50:	01040713          	addi	a4,s0,16
   1ae54:	01050793          	addi	a5,a0,16
   1ae58:	c6dff06f          	j	1aac4 <_realloc_r+0x118>
   1ae5c:	014a8ab3          	add	s5,s5,s4
   1ae60:	414687b3          	sub	a5,a3,s4
   1ae64:	015c2423          	sw	s5,8(s8)
   1ae68:	0017e793          	ori	a5,a5,1
   1ae6c:	00faa223          	sw	a5,4(s5)
   1ae70:	ffc42783          	lw	a5,-4(s0)
   1ae74:	00090513          	mv	a0,s2
   1ae78:	00040493          	mv	s1,s0
   1ae7c:	0017f793          	andi	a5,a5,1
   1ae80:	0147e7b3          	or	a5,a5,s4
   1ae84:	fef42e23          	sw	a5,-4(s0)
   1ae88:	eb5f60ef          	jal	11d3c <__malloc_unlock>
   1ae8c:	00812c03          	lw	s8,8(sp)
   1ae90:	cd1ff06f          	j	1ab60 <_realloc_r+0x1b4>
   1ae94:	00840413          	addi	s0,s0,8
   1ae98:	010b8793          	addi	a5,s7,16
   1ae9c:	ec9ff06f          	j	1ad64 <_realloc_r+0x3b8>
   1aea0:	00090513          	mv	a0,s2
   1aea4:	e99f60ef          	jal	11d3c <__malloc_unlock>
   1aea8:	00000493          	li	s1,0
   1aeac:	00812c03          	lw	s8,8(sp)
   1aeb0:	cb1ff06f          	j	1ab60 <_realloc_r+0x1b4>
   1aeb4:	01042683          	lw	a3,16(s0)
   1aeb8:	01840713          	addi	a4,s0,24
   1aebc:	01850793          	addi	a5,a0,24
   1aec0:	00d52823          	sw	a3,16(a0)
   1aec4:	01442683          	lw	a3,20(s0)
   1aec8:	00d52a23          	sw	a3,20(a0)
   1aecc:	bf9ff06f          	j	1aac4 <_realloc_r+0x118>
   1aed0:	00040593          	mv	a1,s0
   1aed4:	00048513          	mv	a0,s1
   1aed8:	d0dfb0ef          	jal	16be4 <memmove>
   1aedc:	db5ff06f          	j	1ac90 <_realloc_r+0x2e4>
   1aee0:	00842783          	lw	a5,8(s0)
   1aee4:	00fba823          	sw	a5,16(s7)
   1aee8:	00c42783          	lw	a5,12(s0)
   1aeec:	00fbaa23          	sw	a5,20(s7)
   1aef0:	00d60863          	beq	a2,a3,1af00 <_realloc_r+0x554>
   1aef4:	01040413          	addi	s0,s0,16
   1aef8:	018b8793          	addi	a5,s7,24
   1aefc:	d7dff06f          	j	1ac78 <_realloc_r+0x2cc>
   1af00:	01042703          	lw	a4,16(s0)
   1af04:	020b8793          	addi	a5,s7,32
   1af08:	01840413          	addi	s0,s0,24
   1af0c:	00ebac23          	sw	a4,24(s7)
   1af10:	ffc42703          	lw	a4,-4(s0)
   1af14:	00ebae23          	sw	a4,28(s7)
   1af18:	d61ff06f          	j	1ac78 <_realloc_r+0x2cc>

0001af1c <_wctomb_r>:
   1af1c:	f981a783          	lw	a5,-104(gp) # 22818 <__global_locale+0xe0>
   1af20:	00078067          	jr	a5

0001af24 <__ascii_wctomb>:
   1af24:	02058463          	beqz	a1,1af4c <__ascii_wctomb+0x28>
   1af28:	0ff00793          	li	a5,255
   1af2c:	00c7e863          	bltu	a5,a2,1af3c <__ascii_wctomb+0x18>
   1af30:	00c58023          	sb	a2,0(a1)
   1af34:	00100513          	li	a0,1
   1af38:	00008067          	ret
   1af3c:	08a00793          	li	a5,138
   1af40:	00f52023          	sw	a5,0(a0)
   1af44:	fff00513          	li	a0,-1
   1af48:	00008067          	ret
   1af4c:	00000513          	li	a0,0
   1af50:	00008067          	ret

0001af54 <_wcrtomb_r>:
   1af54:	fe010113          	addi	sp,sp,-32
   1af58:	00812c23          	sw	s0,24(sp)
   1af5c:	00912a23          	sw	s1,20(sp)
   1af60:	00112e23          	sw	ra,28(sp)
   1af64:	00050413          	mv	s0,a0
   1af68:	00068493          	mv	s1,a3
   1af6c:	f981a783          	lw	a5,-104(gp) # 22818 <__global_locale+0xe0>
   1af70:	02058263          	beqz	a1,1af94 <_wcrtomb_r+0x40>
   1af74:	000780e7          	jalr	a5
   1af78:	fff00793          	li	a5,-1
   1af7c:	02f50663          	beq	a0,a5,1afa8 <_wcrtomb_r+0x54>
   1af80:	01c12083          	lw	ra,28(sp)
   1af84:	01812403          	lw	s0,24(sp)
   1af88:	01412483          	lw	s1,20(sp)
   1af8c:	02010113          	addi	sp,sp,32
   1af90:	00008067          	ret
   1af94:	00000613          	li	a2,0
   1af98:	00410593          	addi	a1,sp,4
   1af9c:	000780e7          	jalr	a5
   1afa0:	fff00793          	li	a5,-1
   1afa4:	fcf51ee3          	bne	a0,a5,1af80 <_wcrtomb_r+0x2c>
   1afa8:	0004a023          	sw	zero,0(s1)
   1afac:	08a00793          	li	a5,138
   1afb0:	01c12083          	lw	ra,28(sp)
   1afb4:	00f42023          	sw	a5,0(s0)
   1afb8:	01812403          	lw	s0,24(sp)
   1afbc:	01412483          	lw	s1,20(sp)
   1afc0:	02010113          	addi	sp,sp,32
   1afc4:	00008067          	ret

0001afc8 <wcrtomb>:
   1afc8:	fe010113          	addi	sp,sp,-32
   1afcc:	00812c23          	sw	s0,24(sp)
   1afd0:	00912a23          	sw	s1,20(sp)
   1afd4:	00112e23          	sw	ra,28(sp)
   1afd8:	00060413          	mv	s0,a2
   1afdc:	08c1a483          	lw	s1,140(gp) # 2290c <_impure_ptr>
   1afe0:	f981a783          	lw	a5,-104(gp) # 22818 <__global_locale+0xe0>
   1afe4:	02050a63          	beqz	a0,1b018 <wcrtomb+0x50>
   1afe8:	00058613          	mv	a2,a1
   1afec:	00040693          	mv	a3,s0
   1aff0:	00050593          	mv	a1,a0
   1aff4:	00048513          	mv	a0,s1
   1aff8:	000780e7          	jalr	a5
   1affc:	fff00793          	li	a5,-1
   1b000:	02f50a63          	beq	a0,a5,1b034 <wcrtomb+0x6c>
   1b004:	01c12083          	lw	ra,28(sp)
   1b008:	01812403          	lw	s0,24(sp)
   1b00c:	01412483          	lw	s1,20(sp)
   1b010:	02010113          	addi	sp,sp,32
   1b014:	00008067          	ret
   1b018:	00060693          	mv	a3,a2
   1b01c:	00410593          	addi	a1,sp,4
   1b020:	00000613          	li	a2,0
   1b024:	00048513          	mv	a0,s1
   1b028:	000780e7          	jalr	a5
   1b02c:	fff00793          	li	a5,-1
   1b030:	fcf51ae3          	bne	a0,a5,1b004 <wcrtomb+0x3c>
   1b034:	00042023          	sw	zero,0(s0)
   1b038:	01c12083          	lw	ra,28(sp)
   1b03c:	01812403          	lw	s0,24(sp)
   1b040:	08a00793          	li	a5,138
   1b044:	00f4a023          	sw	a5,0(s1)
   1b048:	01412483          	lw	s1,20(sp)
   1b04c:	02010113          	addi	sp,sp,32
   1b050:	00008067          	ret

0001b054 <__smakebuf_r>:
   1b054:	00c59783          	lh	a5,12(a1)
   1b058:	f8010113          	addi	sp,sp,-128
   1b05c:	06812c23          	sw	s0,120(sp)
   1b060:	06112e23          	sw	ra,124(sp)
   1b064:	0027f713          	andi	a4,a5,2
   1b068:	00058413          	mv	s0,a1
   1b06c:	02070463          	beqz	a4,1b094 <__smakebuf_r+0x40>
   1b070:	04358793          	addi	a5,a1,67
   1b074:	00f5a023          	sw	a5,0(a1)
   1b078:	00f5a823          	sw	a5,16(a1)
   1b07c:	00100793          	li	a5,1
   1b080:	00f5aa23          	sw	a5,20(a1)
   1b084:	07c12083          	lw	ra,124(sp)
   1b088:	07812403          	lw	s0,120(sp)
   1b08c:	08010113          	addi	sp,sp,128
   1b090:	00008067          	ret
   1b094:	00e59583          	lh	a1,14(a1)
   1b098:	06912a23          	sw	s1,116(sp)
   1b09c:	07212823          	sw	s2,112(sp)
   1b0a0:	07312623          	sw	s3,108(sp)
   1b0a4:	07412423          	sw	s4,104(sp)
   1b0a8:	00050493          	mv	s1,a0
   1b0ac:	0805c663          	bltz	a1,1b138 <__smakebuf_r+0xe4>
   1b0b0:	00810613          	addi	a2,sp,8
   1b0b4:	3b0000ef          	jal	1b464 <_fstat_r>
   1b0b8:	06054e63          	bltz	a0,1b134 <__smakebuf_r+0xe0>
   1b0bc:	00c12783          	lw	a5,12(sp)
   1b0c0:	0000f937          	lui	s2,0xf
   1b0c4:	000019b7          	lui	s3,0x1
   1b0c8:	00f97933          	and	s2,s2,a5
   1b0cc:	ffffe7b7          	lui	a5,0xffffe
   1b0d0:	00f90933          	add	s2,s2,a5
   1b0d4:	00193913          	seqz	s2,s2
   1b0d8:	40000a13          	li	s4,1024
   1b0dc:	80098993          	addi	s3,s3,-2048 # 800 <exit-0xf8b4>
   1b0e0:	000a0593          	mv	a1,s4
   1b0e4:	00048513          	mv	a0,s1
   1b0e8:	c88f60ef          	jal	11570 <_malloc_r>
   1b0ec:	00c41783          	lh	a5,12(s0)
   1b0f0:	06050863          	beqz	a0,1b160 <__smakebuf_r+0x10c>
   1b0f4:	0807e793          	ori	a5,a5,128
   1b0f8:	00a42023          	sw	a0,0(s0)
   1b0fc:	00a42823          	sw	a0,16(s0)
   1b100:	00f41623          	sh	a5,12(s0)
   1b104:	01442a23          	sw	s4,20(s0)
   1b108:	0a091063          	bnez	s2,1b1a8 <__smakebuf_r+0x154>
   1b10c:	0137e7b3          	or	a5,a5,s3
   1b110:	07c12083          	lw	ra,124(sp)
   1b114:	00f41623          	sh	a5,12(s0)
   1b118:	07812403          	lw	s0,120(sp)
   1b11c:	07412483          	lw	s1,116(sp)
   1b120:	07012903          	lw	s2,112(sp)
   1b124:	06c12983          	lw	s3,108(sp)
   1b128:	06812a03          	lw	s4,104(sp)
   1b12c:	08010113          	addi	sp,sp,128
   1b130:	00008067          	ret
   1b134:	00c41783          	lh	a5,12(s0)
   1b138:	0807f793          	andi	a5,a5,128
   1b13c:	00000913          	li	s2,0
   1b140:	04078e63          	beqz	a5,1b19c <__smakebuf_r+0x148>
   1b144:	04000a13          	li	s4,64
   1b148:	000a0593          	mv	a1,s4
   1b14c:	00048513          	mv	a0,s1
   1b150:	c20f60ef          	jal	11570 <_malloc_r>
   1b154:	00c41783          	lh	a5,12(s0)
   1b158:	00000993          	li	s3,0
   1b15c:	f8051ce3          	bnez	a0,1b0f4 <__smakebuf_r+0xa0>
   1b160:	2007f713          	andi	a4,a5,512
   1b164:	04071e63          	bnez	a4,1b1c0 <__smakebuf_r+0x16c>
   1b168:	ffc7f793          	andi	a5,a5,-4
   1b16c:	0027e793          	ori	a5,a5,2
   1b170:	04340713          	addi	a4,s0,67
   1b174:	00f41623          	sh	a5,12(s0)
   1b178:	00100793          	li	a5,1
   1b17c:	07412483          	lw	s1,116(sp)
   1b180:	07012903          	lw	s2,112(sp)
   1b184:	06c12983          	lw	s3,108(sp)
   1b188:	06812a03          	lw	s4,104(sp)
   1b18c:	00e42023          	sw	a4,0(s0)
   1b190:	00e42823          	sw	a4,16(s0)
   1b194:	00f42a23          	sw	a5,20(s0)
   1b198:	eedff06f          	j	1b084 <__smakebuf_r+0x30>
   1b19c:	40000a13          	li	s4,1024
   1b1a0:	00000993          	li	s3,0
   1b1a4:	f3dff06f          	j	1b0e0 <__smakebuf_r+0x8c>
   1b1a8:	00e41583          	lh	a1,14(s0)
   1b1ac:	00048513          	mv	a0,s1
   1b1b0:	30c000ef          	jal	1b4bc <_isatty_r>
   1b1b4:	02051063          	bnez	a0,1b1d4 <__smakebuf_r+0x180>
   1b1b8:	00c41783          	lh	a5,12(s0)
   1b1bc:	f51ff06f          	j	1b10c <__smakebuf_r+0xb8>
   1b1c0:	07412483          	lw	s1,116(sp)
   1b1c4:	07012903          	lw	s2,112(sp)
   1b1c8:	06c12983          	lw	s3,108(sp)
   1b1cc:	06812a03          	lw	s4,104(sp)
   1b1d0:	eb5ff06f          	j	1b084 <__smakebuf_r+0x30>
   1b1d4:	00c45783          	lhu	a5,12(s0)
   1b1d8:	ffc7f793          	andi	a5,a5,-4
   1b1dc:	0017e793          	ori	a5,a5,1
   1b1e0:	01079793          	slli	a5,a5,0x10
   1b1e4:	4107d793          	srai	a5,a5,0x10
   1b1e8:	f25ff06f          	j	1b10c <__smakebuf_r+0xb8>

0001b1ec <__swhatbuf_r>:
   1b1ec:	f9010113          	addi	sp,sp,-112
   1b1f0:	06812423          	sw	s0,104(sp)
   1b1f4:	00058413          	mv	s0,a1
   1b1f8:	00e59583          	lh	a1,14(a1)
   1b1fc:	06912223          	sw	s1,100(sp)
   1b200:	07212023          	sw	s2,96(sp)
   1b204:	06112623          	sw	ra,108(sp)
   1b208:	00060493          	mv	s1,a2
   1b20c:	00068913          	mv	s2,a3
   1b210:	0405ca63          	bltz	a1,1b264 <__swhatbuf_r+0x78>
   1b214:	00810613          	addi	a2,sp,8
   1b218:	24c000ef          	jal	1b464 <_fstat_r>
   1b21c:	04054463          	bltz	a0,1b264 <__swhatbuf_r+0x78>
   1b220:	00c12703          	lw	a4,12(sp)
   1b224:	0000f7b7          	lui	a5,0xf
   1b228:	06c12083          	lw	ra,108(sp)
   1b22c:	00e7f7b3          	and	a5,a5,a4
   1b230:	ffffe737          	lui	a4,0xffffe
   1b234:	00e787b3          	add	a5,a5,a4
   1b238:	06812403          	lw	s0,104(sp)
   1b23c:	0017b793          	seqz	a5,a5
   1b240:	00f92023          	sw	a5,0(s2) # f000 <exit-0x10b4>
   1b244:	40000713          	li	a4,1024
   1b248:	00e4a023          	sw	a4,0(s1)
   1b24c:	00001537          	lui	a0,0x1
   1b250:	06412483          	lw	s1,100(sp)
   1b254:	06012903          	lw	s2,96(sp)
   1b258:	80050513          	addi	a0,a0,-2048 # 800 <exit-0xf8b4>
   1b25c:	07010113          	addi	sp,sp,112
   1b260:	00008067          	ret
   1b264:	00c45783          	lhu	a5,12(s0)
   1b268:	0807f793          	andi	a5,a5,128
   1b26c:	02078863          	beqz	a5,1b29c <__swhatbuf_r+0xb0>
   1b270:	06c12083          	lw	ra,108(sp)
   1b274:	06812403          	lw	s0,104(sp)
   1b278:	00000793          	li	a5,0
   1b27c:	00f92023          	sw	a5,0(s2)
   1b280:	04000713          	li	a4,64
   1b284:	00e4a023          	sw	a4,0(s1)
   1b288:	06012903          	lw	s2,96(sp)
   1b28c:	06412483          	lw	s1,100(sp)
   1b290:	00000513          	li	a0,0
   1b294:	07010113          	addi	sp,sp,112
   1b298:	00008067          	ret
   1b29c:	06c12083          	lw	ra,108(sp)
   1b2a0:	06812403          	lw	s0,104(sp)
   1b2a4:	00f92023          	sw	a5,0(s2)
   1b2a8:	40000713          	li	a4,1024
   1b2ac:	00e4a023          	sw	a4,0(s1)
   1b2b0:	06012903          	lw	s2,96(sp)
   1b2b4:	06412483          	lw	s1,100(sp)
   1b2b8:	00000513          	li	a0,0
   1b2bc:	07010113          	addi	sp,sp,112
   1b2c0:	00008067          	ret

0001b2c4 <__swbuf_r>:
   1b2c4:	fe010113          	addi	sp,sp,-32
   1b2c8:	00812c23          	sw	s0,24(sp)
   1b2cc:	00912a23          	sw	s1,20(sp)
   1b2d0:	01212823          	sw	s2,16(sp)
   1b2d4:	00112e23          	sw	ra,28(sp)
   1b2d8:	00050913          	mv	s2,a0
   1b2dc:	00058493          	mv	s1,a1
   1b2e0:	00060413          	mv	s0,a2
   1b2e4:	00050663          	beqz	a0,1b2f0 <__swbuf_r+0x2c>
   1b2e8:	03452783          	lw	a5,52(a0)
   1b2ec:	16078063          	beqz	a5,1b44c <__swbuf_r+0x188>
   1b2f0:	01842783          	lw	a5,24(s0)
   1b2f4:	00c41703          	lh	a4,12(s0)
   1b2f8:	00f42423          	sw	a5,8(s0)
   1b2fc:	00877793          	andi	a5,a4,8
   1b300:	08078063          	beqz	a5,1b380 <__swbuf_r+0xbc>
   1b304:	01042783          	lw	a5,16(s0)
   1b308:	06078c63          	beqz	a5,1b380 <__swbuf_r+0xbc>
   1b30c:	01312623          	sw	s3,12(sp)
   1b310:	01271693          	slli	a3,a4,0x12
   1b314:	0ff4f993          	zext.b	s3,s1
   1b318:	0ff4f493          	zext.b	s1,s1
   1b31c:	0806d863          	bgez	a3,1b3ac <__swbuf_r+0xe8>
   1b320:	00042703          	lw	a4,0(s0)
   1b324:	01442683          	lw	a3,20(s0)
   1b328:	40f707b3          	sub	a5,a4,a5
   1b32c:	0ad7d863          	bge	a5,a3,1b3dc <__swbuf_r+0x118>
   1b330:	00842683          	lw	a3,8(s0)
   1b334:	00170613          	addi	a2,a4,1 # ffffe001 <__BSS_END__+0xfffdb2d1>
   1b338:	00c42023          	sw	a2,0(s0)
   1b33c:	fff68693          	addi	a3,a3,-1
   1b340:	00d42423          	sw	a3,8(s0)
   1b344:	01370023          	sb	s3,0(a4)
   1b348:	01442703          	lw	a4,20(s0)
   1b34c:	00178793          	addi	a5,a5,1 # f001 <exit-0x10b3>
   1b350:	0cf70263          	beq	a4,a5,1b414 <__swbuf_r+0x150>
   1b354:	00c45783          	lhu	a5,12(s0)
   1b358:	0017f793          	andi	a5,a5,1
   1b35c:	0c079a63          	bnez	a5,1b430 <__swbuf_r+0x16c>
   1b360:	00c12983          	lw	s3,12(sp)
   1b364:	01c12083          	lw	ra,28(sp)
   1b368:	01812403          	lw	s0,24(sp)
   1b36c:	01012903          	lw	s2,16(sp)
   1b370:	00048513          	mv	a0,s1
   1b374:	01412483          	lw	s1,20(sp)
   1b378:	02010113          	addi	sp,sp,32
   1b37c:	00008067          	ret
   1b380:	00040593          	mv	a1,s0
   1b384:	00090513          	mv	a0,s2
   1b388:	a4cfb0ef          	jal	165d4 <__swsetup_r>
   1b38c:	08051e63          	bnez	a0,1b428 <__swbuf_r+0x164>
   1b390:	00c41703          	lh	a4,12(s0)
   1b394:	01312623          	sw	s3,12(sp)
   1b398:	01042783          	lw	a5,16(s0)
   1b39c:	01271693          	slli	a3,a4,0x12
   1b3a0:	0ff4f993          	zext.b	s3,s1
   1b3a4:	0ff4f493          	zext.b	s1,s1
   1b3a8:	f606cce3          	bltz	a3,1b320 <__swbuf_r+0x5c>
   1b3ac:	06442683          	lw	a3,100(s0)
   1b3b0:	ffffe637          	lui	a2,0xffffe
   1b3b4:	000025b7          	lui	a1,0x2
   1b3b8:	00b76733          	or	a4,a4,a1
   1b3bc:	fff60613          	addi	a2,a2,-1 # ffffdfff <__BSS_END__+0xfffdb2cf>
   1b3c0:	00c6f6b3          	and	a3,a3,a2
   1b3c4:	00e41623          	sh	a4,12(s0)
   1b3c8:	00042703          	lw	a4,0(s0)
   1b3cc:	06d42223          	sw	a3,100(s0)
   1b3d0:	01442683          	lw	a3,20(s0)
   1b3d4:	40f707b3          	sub	a5,a4,a5
   1b3d8:	f4d7cce3          	blt	a5,a3,1b330 <__swbuf_r+0x6c>
   1b3dc:	00040593          	mv	a1,s0
   1b3e0:	00090513          	mv	a0,s2
   1b3e4:	c01fa0ef          	jal	15fe4 <_fflush_r>
   1b3e8:	02051e63          	bnez	a0,1b424 <__swbuf_r+0x160>
   1b3ec:	00042703          	lw	a4,0(s0)
   1b3f0:	00842683          	lw	a3,8(s0)
   1b3f4:	00100793          	li	a5,1
   1b3f8:	00170613          	addi	a2,a4,1
   1b3fc:	fff68693          	addi	a3,a3,-1
   1b400:	00c42023          	sw	a2,0(s0)
   1b404:	00d42423          	sw	a3,8(s0)
   1b408:	01370023          	sb	s3,0(a4)
   1b40c:	01442703          	lw	a4,20(s0)
   1b410:	f4f712e3          	bne	a4,a5,1b354 <__swbuf_r+0x90>
   1b414:	00040593          	mv	a1,s0
   1b418:	00090513          	mv	a0,s2
   1b41c:	bc9fa0ef          	jal	15fe4 <_fflush_r>
   1b420:	f40500e3          	beqz	a0,1b360 <__swbuf_r+0x9c>
   1b424:	00c12983          	lw	s3,12(sp)
   1b428:	fff00493          	li	s1,-1
   1b42c:	f39ff06f          	j	1b364 <__swbuf_r+0xa0>
   1b430:	00a00793          	li	a5,10
   1b434:	f2f496e3          	bne	s1,a5,1b360 <__swbuf_r+0x9c>
   1b438:	00040593          	mv	a1,s0
   1b43c:	00090513          	mv	a0,s2
   1b440:	ba5fa0ef          	jal	15fe4 <_fflush_r>
   1b444:	f0050ee3          	beqz	a0,1b360 <__swbuf_r+0x9c>
   1b448:	fddff06f          	j	1b424 <__swbuf_r+0x160>
   1b44c:	b9cf50ef          	jal	107e8 <__sinit>
   1b450:	ea1ff06f          	j	1b2f0 <__swbuf_r+0x2c>

0001b454 <__swbuf>:
   1b454:	00058613          	mv	a2,a1
   1b458:	00050593          	mv	a1,a0
   1b45c:	08c1a503          	lw	a0,140(gp) # 2290c <_impure_ptr>
   1b460:	e65ff06f          	j	1b2c4 <__swbuf_r>

0001b464 <_fstat_r>:
   1b464:	ff010113          	addi	sp,sp,-16
   1b468:	00058713          	mv	a4,a1
   1b46c:	00812423          	sw	s0,8(sp)
   1b470:	00060593          	mv	a1,a2
   1b474:	00050413          	mv	s0,a0
   1b478:	00070513          	mv	a0,a4
   1b47c:	0801ae23          	sw	zero,156(gp) # 2291c <errno>
   1b480:	00112623          	sw	ra,12(sp)
   1b484:	48c050ef          	jal	20910 <_fstat>
   1b488:	fff00793          	li	a5,-1
   1b48c:	00f50a63          	beq	a0,a5,1b4a0 <_fstat_r+0x3c>
   1b490:	00c12083          	lw	ra,12(sp)
   1b494:	00812403          	lw	s0,8(sp)
   1b498:	01010113          	addi	sp,sp,16
   1b49c:	00008067          	ret
   1b4a0:	09c1a783          	lw	a5,156(gp) # 2291c <errno>
   1b4a4:	fe0786e3          	beqz	a5,1b490 <_fstat_r+0x2c>
   1b4a8:	00c12083          	lw	ra,12(sp)
   1b4ac:	00f42023          	sw	a5,0(s0)
   1b4b0:	00812403          	lw	s0,8(sp)
   1b4b4:	01010113          	addi	sp,sp,16
   1b4b8:	00008067          	ret

0001b4bc <_isatty_r>:
   1b4bc:	ff010113          	addi	sp,sp,-16
   1b4c0:	00812423          	sw	s0,8(sp)
   1b4c4:	00050413          	mv	s0,a0
   1b4c8:	00058513          	mv	a0,a1
   1b4cc:	0801ae23          	sw	zero,156(gp) # 2291c <errno>
   1b4d0:	00112623          	sw	ra,12(sp)
   1b4d4:	45c050ef          	jal	20930 <_isatty>
   1b4d8:	fff00793          	li	a5,-1
   1b4dc:	00f50a63          	beq	a0,a5,1b4f0 <_isatty_r+0x34>
   1b4e0:	00c12083          	lw	ra,12(sp)
   1b4e4:	00812403          	lw	s0,8(sp)
   1b4e8:	01010113          	addi	sp,sp,16
   1b4ec:	00008067          	ret
   1b4f0:	09c1a783          	lw	a5,156(gp) # 2291c <errno>
   1b4f4:	fe0786e3          	beqz	a5,1b4e0 <_isatty_r+0x24>
   1b4f8:	00c12083          	lw	ra,12(sp)
   1b4fc:	00f42023          	sw	a5,0(s0)
   1b500:	00812403          	lw	s0,8(sp)
   1b504:	01010113          	addi	sp,sp,16
   1b508:	00008067          	ret

0001b50c <__assert_func>:
   1b50c:	ff010113          	addi	sp,sp,-16
   1b510:	00068793          	mv	a5,a3
   1b514:	08c1a703          	lw	a4,140(gp) # 2290c <_impure_ptr>
   1b518:	00060813          	mv	a6,a2
   1b51c:	00112623          	sw	ra,12(sp)
   1b520:	00c72883          	lw	a7,12(a4)
   1b524:	00078613          	mv	a2,a5
   1b528:	00050693          	mv	a3,a0
   1b52c:	00058713          	mv	a4,a1
   1b530:	00005797          	auipc	a5,0x5
   1b534:	71878793          	addi	a5,a5,1816 # 20c48 <_exit+0x298>
   1b538:	00080c63          	beqz	a6,1b550 <__assert_func+0x44>
   1b53c:	00005597          	auipc	a1,0x5
   1b540:	71c58593          	addi	a1,a1,1820 # 20c58 <_exit+0x2a8>
   1b544:	00088513          	mv	a0,a7
   1b548:	14c000ef          	jal	1b694 <fiprintf>
   1b54c:	198000ef          	jal	1b6e4 <abort>
   1b550:	00005797          	auipc	a5,0x5
   1b554:	70478793          	addi	a5,a5,1796 # 20c54 <_exit+0x2a4>
   1b558:	00078813          	mv	a6,a5
   1b55c:	fe1ff06f          	j	1b53c <__assert_func+0x30>

0001b560 <__assert>:
   1b560:	ff010113          	addi	sp,sp,-16
   1b564:	00060693          	mv	a3,a2
   1b568:	00000613          	li	a2,0
   1b56c:	00112623          	sw	ra,12(sp)
   1b570:	f9dff0ef          	jal	1b50c <__assert_func>

0001b574 <_calloc_r>:
   1b574:	02c5b7b3          	mulhu	a5,a1,a2
   1b578:	ff010113          	addi	sp,sp,-16
   1b57c:	00112623          	sw	ra,12(sp)
   1b580:	00812423          	sw	s0,8(sp)
   1b584:	02c585b3          	mul	a1,a1,a2
   1b588:	0a079063          	bnez	a5,1b628 <_calloc_r+0xb4>
   1b58c:	fe5f50ef          	jal	11570 <_malloc_r>
   1b590:	00050413          	mv	s0,a0
   1b594:	0a050063          	beqz	a0,1b634 <_calloc_r+0xc0>
   1b598:	ffc52603          	lw	a2,-4(a0)
   1b59c:	02400713          	li	a4,36
   1b5a0:	ffc67613          	andi	a2,a2,-4
   1b5a4:	ffc60613          	addi	a2,a2,-4
   1b5a8:	04c76863          	bltu	a4,a2,1b5f8 <_calloc_r+0x84>
   1b5ac:	01300693          	li	a3,19
   1b5b0:	00050793          	mv	a5,a0
   1b5b4:	02c6f263          	bgeu	a3,a2,1b5d8 <_calloc_r+0x64>
   1b5b8:	00052023          	sw	zero,0(a0)
   1b5bc:	00052223          	sw	zero,4(a0)
   1b5c0:	01b00793          	li	a5,27
   1b5c4:	04c7f863          	bgeu	a5,a2,1b614 <_calloc_r+0xa0>
   1b5c8:	00052423          	sw	zero,8(a0)
   1b5cc:	00052623          	sw	zero,12(a0)
   1b5d0:	01050793          	addi	a5,a0,16
   1b5d4:	06e60c63          	beq	a2,a4,1b64c <_calloc_r+0xd8>
   1b5d8:	0007a023          	sw	zero,0(a5)
   1b5dc:	0007a223          	sw	zero,4(a5)
   1b5e0:	0007a423          	sw	zero,8(a5)
   1b5e4:	00c12083          	lw	ra,12(sp)
   1b5e8:	00040513          	mv	a0,s0
   1b5ec:	00812403          	lw	s0,8(sp)
   1b5f0:	01010113          	addi	sp,sp,16
   1b5f4:	00008067          	ret
   1b5f8:	00000593          	li	a1,0
   1b5fc:	885f50ef          	jal	10e80 <memset>
   1b600:	00c12083          	lw	ra,12(sp)
   1b604:	00040513          	mv	a0,s0
   1b608:	00812403          	lw	s0,8(sp)
   1b60c:	01010113          	addi	sp,sp,16
   1b610:	00008067          	ret
   1b614:	00850793          	addi	a5,a0,8
   1b618:	0007a023          	sw	zero,0(a5)
   1b61c:	0007a223          	sw	zero,4(a5)
   1b620:	0007a423          	sw	zero,8(a5)
   1b624:	fc1ff06f          	j	1b5e4 <_calloc_r+0x70>
   1b628:	0b4000ef          	jal	1b6dc <__errno>
   1b62c:	00c00793          	li	a5,12
   1b630:	00f52023          	sw	a5,0(a0)
   1b634:	00000413          	li	s0,0
   1b638:	00c12083          	lw	ra,12(sp)
   1b63c:	00040513          	mv	a0,s0
   1b640:	00812403          	lw	s0,8(sp)
   1b644:	01010113          	addi	sp,sp,16
   1b648:	00008067          	ret
   1b64c:	00052823          	sw	zero,16(a0)
   1b650:	01850793          	addi	a5,a0,24
   1b654:	00052a23          	sw	zero,20(a0)
   1b658:	f81ff06f          	j	1b5d8 <_calloc_r+0x64>

0001b65c <_fiprintf_r>:
   1b65c:	fc010113          	addi	sp,sp,-64
   1b660:	02c10313          	addi	t1,sp,44
   1b664:	02d12623          	sw	a3,44(sp)
   1b668:	00030693          	mv	a3,t1
   1b66c:	00112e23          	sw	ra,28(sp)
   1b670:	02e12823          	sw	a4,48(sp)
   1b674:	02f12a23          	sw	a5,52(sp)
   1b678:	03012c23          	sw	a6,56(sp)
   1b67c:	03112e23          	sw	a7,60(sp)
   1b680:	00612623          	sw	t1,12(sp)
   1b684:	9c0f90ef          	jal	14844 <_vfiprintf_r>
   1b688:	01c12083          	lw	ra,28(sp)
   1b68c:	04010113          	addi	sp,sp,64
   1b690:	00008067          	ret

0001b694 <fiprintf>:
   1b694:	fc010113          	addi	sp,sp,-64
   1b698:	02810313          	addi	t1,sp,40
   1b69c:	02c12423          	sw	a2,40(sp)
   1b6a0:	02d12623          	sw	a3,44(sp)
   1b6a4:	00058613          	mv	a2,a1
   1b6a8:	00030693          	mv	a3,t1
   1b6ac:	00050593          	mv	a1,a0
   1b6b0:	08c1a503          	lw	a0,140(gp) # 2290c <_impure_ptr>
   1b6b4:	00112e23          	sw	ra,28(sp)
   1b6b8:	02e12823          	sw	a4,48(sp)
   1b6bc:	02f12a23          	sw	a5,52(sp)
   1b6c0:	03012c23          	sw	a6,56(sp)
   1b6c4:	03112e23          	sw	a7,60(sp)
   1b6c8:	00612623          	sw	t1,12(sp)
   1b6cc:	978f90ef          	jal	14844 <_vfiprintf_r>
   1b6d0:	01c12083          	lw	ra,28(sp)
   1b6d4:	04010113          	addi	sp,sp,64
   1b6d8:	00008067          	ret

0001b6dc <__errno>:
   1b6dc:	08c1a503          	lw	a0,140(gp) # 2290c <_impure_ptr>
   1b6e0:	00008067          	ret

0001b6e4 <abort>:
   1b6e4:	ff010113          	addi	sp,sp,-16
   1b6e8:	00600513          	li	a0,6
   1b6ec:	00112623          	sw	ra,12(sp)
   1b6f0:	2a8000ef          	jal	1b998 <raise>
   1b6f4:	00100513          	li	a0,1
   1b6f8:	2b8050ef          	jal	209b0 <_exit>

0001b6fc <_init_signal_r>:
   1b6fc:	11852783          	lw	a5,280(a0)
   1b700:	00078663          	beqz	a5,1b70c <_init_signal_r+0x10>
   1b704:	00000513          	li	a0,0
   1b708:	00008067          	ret
   1b70c:	ff010113          	addi	sp,sp,-16
   1b710:	08000593          	li	a1,128
   1b714:	00812423          	sw	s0,8(sp)
   1b718:	00112623          	sw	ra,12(sp)
   1b71c:	00050413          	mv	s0,a0
   1b720:	e51f50ef          	jal	11570 <_malloc_r>
   1b724:	10a42c23          	sw	a0,280(s0)
   1b728:	02050463          	beqz	a0,1b750 <_init_signal_r+0x54>
   1b72c:	08050793          	addi	a5,a0,128
   1b730:	00052023          	sw	zero,0(a0)
   1b734:	00450513          	addi	a0,a0,4
   1b738:	fef51ce3          	bne	a0,a5,1b730 <_init_signal_r+0x34>
   1b73c:	00000513          	li	a0,0
   1b740:	00c12083          	lw	ra,12(sp)
   1b744:	00812403          	lw	s0,8(sp)
   1b748:	01010113          	addi	sp,sp,16
   1b74c:	00008067          	ret
   1b750:	fff00513          	li	a0,-1
   1b754:	fedff06f          	j	1b740 <_init_signal_r+0x44>

0001b758 <_signal_r>:
   1b758:	fe010113          	addi	sp,sp,-32
   1b75c:	00912a23          	sw	s1,20(sp)
   1b760:	00112e23          	sw	ra,28(sp)
   1b764:	01f00793          	li	a5,31
   1b768:	00050493          	mv	s1,a0
   1b76c:	02b7ec63          	bltu	a5,a1,1b7a4 <_signal_r+0x4c>
   1b770:	11852783          	lw	a5,280(a0)
   1b774:	00812c23          	sw	s0,24(sp)
   1b778:	00058413          	mv	s0,a1
   1b77c:	02078c63          	beqz	a5,1b7b4 <_signal_r+0x5c>
   1b780:	00241413          	slli	s0,s0,0x2
   1b784:	008787b3          	add	a5,a5,s0
   1b788:	01812403          	lw	s0,24(sp)
   1b78c:	0007a503          	lw	a0,0(a5)
   1b790:	00c7a023          	sw	a2,0(a5)
   1b794:	01c12083          	lw	ra,28(sp)
   1b798:	01412483          	lw	s1,20(sp)
   1b79c:	02010113          	addi	sp,sp,32
   1b7a0:	00008067          	ret
   1b7a4:	01600793          	li	a5,22
   1b7a8:	00f52023          	sw	a5,0(a0)
   1b7ac:	fff00513          	li	a0,-1
   1b7b0:	fe5ff06f          	j	1b794 <_signal_r+0x3c>
   1b7b4:	08000593          	li	a1,128
   1b7b8:	00c12623          	sw	a2,12(sp)
   1b7bc:	db5f50ef          	jal	11570 <_malloc_r>
   1b7c0:	10a4ac23          	sw	a0,280(s1)
   1b7c4:	00c12603          	lw	a2,12(sp)
   1b7c8:	00050793          	mv	a5,a0
   1b7cc:	00050713          	mv	a4,a0
   1b7d0:	08050693          	addi	a3,a0,128
   1b7d4:	00050a63          	beqz	a0,1b7e8 <_signal_r+0x90>
   1b7d8:	00072023          	sw	zero,0(a4)
   1b7dc:	00470713          	addi	a4,a4,4
   1b7e0:	fed71ce3          	bne	a4,a3,1b7d8 <_signal_r+0x80>
   1b7e4:	f9dff06f          	j	1b780 <_signal_r+0x28>
   1b7e8:	01812403          	lw	s0,24(sp)
   1b7ec:	fff00513          	li	a0,-1
   1b7f0:	fa5ff06f          	j	1b794 <_signal_r+0x3c>

0001b7f4 <_raise_r>:
   1b7f4:	ff010113          	addi	sp,sp,-16
   1b7f8:	00912223          	sw	s1,4(sp)
   1b7fc:	00112623          	sw	ra,12(sp)
   1b800:	01f00793          	li	a5,31
   1b804:	00050493          	mv	s1,a0
   1b808:	0ab7e063          	bltu	a5,a1,1b8a8 <_raise_r+0xb4>
   1b80c:	11852783          	lw	a5,280(a0)
   1b810:	00812423          	sw	s0,8(sp)
   1b814:	00058413          	mv	s0,a1
   1b818:	04078463          	beqz	a5,1b860 <_raise_r+0x6c>
   1b81c:	00259713          	slli	a4,a1,0x2
   1b820:	00e787b3          	add	a5,a5,a4
   1b824:	0007a703          	lw	a4,0(a5)
   1b828:	02070c63          	beqz	a4,1b860 <_raise_r+0x6c>
   1b82c:	00100693          	li	a3,1
   1b830:	00d70c63          	beq	a4,a3,1b848 <_raise_r+0x54>
   1b834:	fff00693          	li	a3,-1
   1b838:	04d70863          	beq	a4,a3,1b888 <_raise_r+0x94>
   1b83c:	0007a023          	sw	zero,0(a5)
   1b840:	00058513          	mv	a0,a1
   1b844:	000700e7          	jalr	a4
   1b848:	00812403          	lw	s0,8(sp)
   1b84c:	00000513          	li	a0,0
   1b850:	00c12083          	lw	ra,12(sp)
   1b854:	00412483          	lw	s1,4(sp)
   1b858:	01010113          	addi	sp,sp,16
   1b85c:	00008067          	ret
   1b860:	00048513          	mv	a0,s1
   1b864:	430000ef          	jal	1bc94 <_getpid_r>
   1b868:	00040613          	mv	a2,s0
   1b86c:	00812403          	lw	s0,8(sp)
   1b870:	00c12083          	lw	ra,12(sp)
   1b874:	00050593          	mv	a1,a0
   1b878:	00048513          	mv	a0,s1
   1b87c:	00412483          	lw	s1,4(sp)
   1b880:	01010113          	addi	sp,sp,16
   1b884:	3b80006f          	j	1bc3c <_kill_r>
   1b888:	00812403          	lw	s0,8(sp)
   1b88c:	00c12083          	lw	ra,12(sp)
   1b890:	01600793          	li	a5,22
   1b894:	00f52023          	sw	a5,0(a0)
   1b898:	00412483          	lw	s1,4(sp)
   1b89c:	00100513          	li	a0,1
   1b8a0:	01010113          	addi	sp,sp,16
   1b8a4:	00008067          	ret
   1b8a8:	01600793          	li	a5,22
   1b8ac:	00f52023          	sw	a5,0(a0)
   1b8b0:	fff00513          	li	a0,-1
   1b8b4:	f9dff06f          	j	1b850 <_raise_r+0x5c>

0001b8b8 <__sigtramp_r>:
   1b8b8:	01f00793          	li	a5,31
   1b8bc:	0cb7ea63          	bltu	a5,a1,1b990 <__sigtramp_r+0xd8>
   1b8c0:	11852783          	lw	a5,280(a0)
   1b8c4:	ff010113          	addi	sp,sp,-16
   1b8c8:	00812423          	sw	s0,8(sp)
   1b8cc:	00912223          	sw	s1,4(sp)
   1b8d0:	00112623          	sw	ra,12(sp)
   1b8d4:	00058413          	mv	s0,a1
   1b8d8:	00050493          	mv	s1,a0
   1b8dc:	08078063          	beqz	a5,1b95c <__sigtramp_r+0xa4>
   1b8e0:	00241713          	slli	a4,s0,0x2
   1b8e4:	00e787b3          	add	a5,a5,a4
   1b8e8:	0007a703          	lw	a4,0(a5)
   1b8ec:	02070c63          	beqz	a4,1b924 <__sigtramp_r+0x6c>
   1b8f0:	fff00693          	li	a3,-1
   1b8f4:	06d70063          	beq	a4,a3,1b954 <__sigtramp_r+0x9c>
   1b8f8:	00100693          	li	a3,1
   1b8fc:	04d70063          	beq	a4,a3,1b93c <__sigtramp_r+0x84>
   1b900:	00040513          	mv	a0,s0
   1b904:	0007a023          	sw	zero,0(a5)
   1b908:	000700e7          	jalr	a4
   1b90c:	00000513          	li	a0,0
   1b910:	00c12083          	lw	ra,12(sp)
   1b914:	00812403          	lw	s0,8(sp)
   1b918:	00412483          	lw	s1,4(sp)
   1b91c:	01010113          	addi	sp,sp,16
   1b920:	00008067          	ret
   1b924:	00c12083          	lw	ra,12(sp)
   1b928:	00812403          	lw	s0,8(sp)
   1b92c:	00412483          	lw	s1,4(sp)
   1b930:	00100513          	li	a0,1
   1b934:	01010113          	addi	sp,sp,16
   1b938:	00008067          	ret
   1b93c:	00c12083          	lw	ra,12(sp)
   1b940:	00812403          	lw	s0,8(sp)
   1b944:	00412483          	lw	s1,4(sp)
   1b948:	00300513          	li	a0,3
   1b94c:	01010113          	addi	sp,sp,16
   1b950:	00008067          	ret
   1b954:	00200513          	li	a0,2
   1b958:	fb9ff06f          	j	1b910 <__sigtramp_r+0x58>
   1b95c:	08000593          	li	a1,128
   1b960:	c11f50ef          	jal	11570 <_malloc_r>
   1b964:	10a4ac23          	sw	a0,280(s1)
   1b968:	00050793          	mv	a5,a0
   1b96c:	00050e63          	beqz	a0,1b988 <__sigtramp_r+0xd0>
   1b970:	00050713          	mv	a4,a0
   1b974:	08050693          	addi	a3,a0,128
   1b978:	00072023          	sw	zero,0(a4)
   1b97c:	00470713          	addi	a4,a4,4
   1b980:	fed71ce3          	bne	a4,a3,1b978 <__sigtramp_r+0xc0>
   1b984:	f5dff06f          	j	1b8e0 <__sigtramp_r+0x28>
   1b988:	fff00513          	li	a0,-1
   1b98c:	f85ff06f          	j	1b910 <__sigtramp_r+0x58>
   1b990:	fff00513          	li	a0,-1
   1b994:	00008067          	ret

0001b998 <raise>:
   1b998:	ff010113          	addi	sp,sp,-16
   1b99c:	00912223          	sw	s1,4(sp)
   1b9a0:	00112623          	sw	ra,12(sp)
   1b9a4:	01f00793          	li	a5,31
   1b9a8:	08c1a483          	lw	s1,140(gp) # 2290c <_impure_ptr>
   1b9ac:	08a7ee63          	bltu	a5,a0,1ba48 <raise+0xb0>
   1b9b0:	1184a783          	lw	a5,280(s1)
   1b9b4:	00812423          	sw	s0,8(sp)
   1b9b8:	00050413          	mv	s0,a0
   1b9bc:	04078263          	beqz	a5,1ba00 <raise+0x68>
   1b9c0:	00251713          	slli	a4,a0,0x2
   1b9c4:	00e787b3          	add	a5,a5,a4
   1b9c8:	0007a703          	lw	a4,0(a5)
   1b9cc:	02070a63          	beqz	a4,1ba00 <raise+0x68>
   1b9d0:	00100693          	li	a3,1
   1b9d4:	00d70a63          	beq	a4,a3,1b9e8 <raise+0x50>
   1b9d8:	fff00693          	li	a3,-1
   1b9dc:	04d70663          	beq	a4,a3,1ba28 <raise+0x90>
   1b9e0:	0007a023          	sw	zero,0(a5)
   1b9e4:	000700e7          	jalr	a4
   1b9e8:	00812403          	lw	s0,8(sp)
   1b9ec:	00000513          	li	a0,0
   1b9f0:	00c12083          	lw	ra,12(sp)
   1b9f4:	00412483          	lw	s1,4(sp)
   1b9f8:	01010113          	addi	sp,sp,16
   1b9fc:	00008067          	ret
   1ba00:	00048513          	mv	a0,s1
   1ba04:	290000ef          	jal	1bc94 <_getpid_r>
   1ba08:	00040613          	mv	a2,s0
   1ba0c:	00812403          	lw	s0,8(sp)
   1ba10:	00c12083          	lw	ra,12(sp)
   1ba14:	00050593          	mv	a1,a0
   1ba18:	00048513          	mv	a0,s1
   1ba1c:	00412483          	lw	s1,4(sp)
   1ba20:	01010113          	addi	sp,sp,16
   1ba24:	2180006f          	j	1bc3c <_kill_r>
   1ba28:	00812403          	lw	s0,8(sp)
   1ba2c:	00c12083          	lw	ra,12(sp)
   1ba30:	01600793          	li	a5,22
   1ba34:	00f4a023          	sw	a5,0(s1)
   1ba38:	00100513          	li	a0,1
   1ba3c:	00412483          	lw	s1,4(sp)
   1ba40:	01010113          	addi	sp,sp,16
   1ba44:	00008067          	ret
   1ba48:	01600793          	li	a5,22
   1ba4c:	00f4a023          	sw	a5,0(s1)
   1ba50:	fff00513          	li	a0,-1
   1ba54:	f9dff06f          	j	1b9f0 <raise+0x58>

0001ba58 <signal>:
   1ba58:	ff010113          	addi	sp,sp,-16
   1ba5c:	01212023          	sw	s2,0(sp)
   1ba60:	00112623          	sw	ra,12(sp)
   1ba64:	01f00793          	li	a5,31
   1ba68:	08c1a903          	lw	s2,140(gp) # 2290c <_impure_ptr>
   1ba6c:	04a7e263          	bltu	a5,a0,1bab0 <signal+0x58>
   1ba70:	00812423          	sw	s0,8(sp)
   1ba74:	00050413          	mv	s0,a0
   1ba78:	11892503          	lw	a0,280(s2)
   1ba7c:	00912223          	sw	s1,4(sp)
   1ba80:	00058493          	mv	s1,a1
   1ba84:	02050e63          	beqz	a0,1bac0 <signal+0x68>
   1ba88:	00241413          	slli	s0,s0,0x2
   1ba8c:	008507b3          	add	a5,a0,s0
   1ba90:	0007a503          	lw	a0,0(a5)
   1ba94:	00812403          	lw	s0,8(sp)
   1ba98:	0097a023          	sw	s1,0(a5)
   1ba9c:	00412483          	lw	s1,4(sp)
   1baa0:	00c12083          	lw	ra,12(sp)
   1baa4:	00012903          	lw	s2,0(sp)
   1baa8:	01010113          	addi	sp,sp,16
   1baac:	00008067          	ret
   1bab0:	01600793          	li	a5,22
   1bab4:	00f92023          	sw	a5,0(s2)
   1bab8:	fff00513          	li	a0,-1
   1babc:	fe5ff06f          	j	1baa0 <signal+0x48>
   1bac0:	08000593          	li	a1,128
   1bac4:	00090513          	mv	a0,s2
   1bac8:	aa9f50ef          	jal	11570 <_malloc_r>
   1bacc:	10a92c23          	sw	a0,280(s2)
   1bad0:	00050793          	mv	a5,a0
   1bad4:	08050713          	addi	a4,a0,128
   1bad8:	00050a63          	beqz	a0,1baec <signal+0x94>
   1badc:	0007a023          	sw	zero,0(a5)
   1bae0:	00478793          	addi	a5,a5,4
   1bae4:	fef71ce3          	bne	a4,a5,1badc <signal+0x84>
   1bae8:	fa1ff06f          	j	1ba88 <signal+0x30>
   1baec:	00812403          	lw	s0,8(sp)
   1baf0:	00412483          	lw	s1,4(sp)
   1baf4:	fff00513          	li	a0,-1
   1baf8:	fa9ff06f          	j	1baa0 <signal+0x48>

0001bafc <_init_signal>:
   1bafc:	ff010113          	addi	sp,sp,-16
   1bb00:	00812423          	sw	s0,8(sp)
   1bb04:	08c1a403          	lw	s0,140(gp) # 2290c <_impure_ptr>
   1bb08:	11842783          	lw	a5,280(s0)
   1bb0c:	00112623          	sw	ra,12(sp)
   1bb10:	00078c63          	beqz	a5,1bb28 <_init_signal+0x2c>
   1bb14:	00000513          	li	a0,0
   1bb18:	00c12083          	lw	ra,12(sp)
   1bb1c:	00812403          	lw	s0,8(sp)
   1bb20:	01010113          	addi	sp,sp,16
   1bb24:	00008067          	ret
   1bb28:	08000593          	li	a1,128
   1bb2c:	00040513          	mv	a0,s0
   1bb30:	a41f50ef          	jal	11570 <_malloc_r>
   1bb34:	10a42c23          	sw	a0,280(s0)
   1bb38:	00050c63          	beqz	a0,1bb50 <_init_signal+0x54>
   1bb3c:	08050793          	addi	a5,a0,128
   1bb40:	00052023          	sw	zero,0(a0)
   1bb44:	00450513          	addi	a0,a0,4
   1bb48:	fef51ce3          	bne	a0,a5,1bb40 <_init_signal+0x44>
   1bb4c:	fc9ff06f          	j	1bb14 <_init_signal+0x18>
   1bb50:	fff00513          	li	a0,-1
   1bb54:	fc5ff06f          	j	1bb18 <_init_signal+0x1c>

0001bb58 <__sigtramp>:
   1bb58:	ff010113          	addi	sp,sp,-16
   1bb5c:	00912223          	sw	s1,4(sp)
   1bb60:	00112623          	sw	ra,12(sp)
   1bb64:	01f00793          	li	a5,31
   1bb68:	08c1a483          	lw	s1,140(gp) # 2290c <_impure_ptr>
   1bb6c:	0ca7e463          	bltu	a5,a0,1bc34 <__sigtramp+0xdc>
   1bb70:	1184a783          	lw	a5,280(s1)
   1bb74:	00812423          	sw	s0,8(sp)
   1bb78:	00050413          	mv	s0,a0
   1bb7c:	08078263          	beqz	a5,1bc00 <__sigtramp+0xa8>
   1bb80:	00241713          	slli	a4,s0,0x2
   1bb84:	00e787b3          	add	a5,a5,a4
   1bb88:	0007a703          	lw	a4,0(a5)
   1bb8c:	02070c63          	beqz	a4,1bbc4 <__sigtramp+0x6c>
   1bb90:	fff00693          	li	a3,-1
   1bb94:	06d70063          	beq	a4,a3,1bbf4 <__sigtramp+0x9c>
   1bb98:	00100693          	li	a3,1
   1bb9c:	04d70063          	beq	a4,a3,1bbdc <__sigtramp+0x84>
   1bba0:	00040513          	mv	a0,s0
   1bba4:	0007a023          	sw	zero,0(a5)
   1bba8:	000700e7          	jalr	a4
   1bbac:	00812403          	lw	s0,8(sp)
   1bbb0:	00000513          	li	a0,0
   1bbb4:	00c12083          	lw	ra,12(sp)
   1bbb8:	00412483          	lw	s1,4(sp)
   1bbbc:	01010113          	addi	sp,sp,16
   1bbc0:	00008067          	ret
   1bbc4:	00812403          	lw	s0,8(sp)
   1bbc8:	00c12083          	lw	ra,12(sp)
   1bbcc:	00412483          	lw	s1,4(sp)
   1bbd0:	00100513          	li	a0,1
   1bbd4:	01010113          	addi	sp,sp,16
   1bbd8:	00008067          	ret
   1bbdc:	00812403          	lw	s0,8(sp)
   1bbe0:	00c12083          	lw	ra,12(sp)
   1bbe4:	00412483          	lw	s1,4(sp)
   1bbe8:	00300513          	li	a0,3
   1bbec:	01010113          	addi	sp,sp,16
   1bbf0:	00008067          	ret
   1bbf4:	00812403          	lw	s0,8(sp)
   1bbf8:	00200513          	li	a0,2
   1bbfc:	fb9ff06f          	j	1bbb4 <__sigtramp+0x5c>
   1bc00:	08000593          	li	a1,128
   1bc04:	00048513          	mv	a0,s1
   1bc08:	969f50ef          	jal	11570 <_malloc_r>
   1bc0c:	10a4ac23          	sw	a0,280(s1)
   1bc10:	00050793          	mv	a5,a0
   1bc14:	00050e63          	beqz	a0,1bc30 <__sigtramp+0xd8>
   1bc18:	00050713          	mv	a4,a0
   1bc1c:	08050693          	addi	a3,a0,128
   1bc20:	00072023          	sw	zero,0(a4)
   1bc24:	00470713          	addi	a4,a4,4
   1bc28:	fee69ce3          	bne	a3,a4,1bc20 <__sigtramp+0xc8>
   1bc2c:	f55ff06f          	j	1bb80 <__sigtramp+0x28>
   1bc30:	00812403          	lw	s0,8(sp)
   1bc34:	fff00513          	li	a0,-1
   1bc38:	f7dff06f          	j	1bbb4 <__sigtramp+0x5c>

0001bc3c <_kill_r>:
   1bc3c:	ff010113          	addi	sp,sp,-16
   1bc40:	00058713          	mv	a4,a1
   1bc44:	00812423          	sw	s0,8(sp)
   1bc48:	00060593          	mv	a1,a2
   1bc4c:	00050413          	mv	s0,a0
   1bc50:	00070513          	mv	a0,a4
   1bc54:	0801ae23          	sw	zero,156(gp) # 2291c <errno>
   1bc58:	00112623          	sw	ra,12(sp)
   1bc5c:	4e5040ef          	jal	20940 <_kill>
   1bc60:	fff00793          	li	a5,-1
   1bc64:	00f50a63          	beq	a0,a5,1bc78 <_kill_r+0x3c>
   1bc68:	00c12083          	lw	ra,12(sp)
   1bc6c:	00812403          	lw	s0,8(sp)
   1bc70:	01010113          	addi	sp,sp,16
   1bc74:	00008067          	ret
   1bc78:	09c1a783          	lw	a5,156(gp) # 2291c <errno>
   1bc7c:	fe0786e3          	beqz	a5,1bc68 <_kill_r+0x2c>
   1bc80:	00c12083          	lw	ra,12(sp)
   1bc84:	00f42023          	sw	a5,0(s0)
   1bc88:	00812403          	lw	s0,8(sp)
   1bc8c:	01010113          	addi	sp,sp,16
   1bc90:	00008067          	ret

0001bc94 <_getpid_r>:
   1bc94:	48d0406f          	j	20920 <_getpid>

0001bc98 <__adddf3>:
   1bc98:	00100837          	lui	a6,0x100
   1bc9c:	fff80813          	addi	a6,a6,-1 # fffff <__BSS_END__+0xdd2cf>
   1bca0:	fe010113          	addi	sp,sp,-32
   1bca4:	00b878b3          	and	a7,a6,a1
   1bca8:	0145d713          	srli	a4,a1,0x14
   1bcac:	01d55793          	srli	a5,a0,0x1d
   1bcb0:	00d87833          	and	a6,a6,a3
   1bcb4:	00912a23          	sw	s1,20(sp)
   1bcb8:	7ff77493          	andi	s1,a4,2047
   1bcbc:	00389713          	slli	a4,a7,0x3
   1bcc0:	0146d893          	srli	a7,a3,0x14
   1bcc4:	00381813          	slli	a6,a6,0x3
   1bcc8:	01212823          	sw	s2,16(sp)
   1bccc:	00e7e7b3          	or	a5,a5,a4
   1bcd0:	7ff8f893          	andi	a7,a7,2047
   1bcd4:	01d65713          	srli	a4,a2,0x1d
   1bcd8:	00112e23          	sw	ra,28(sp)
   1bcdc:	00812c23          	sw	s0,24(sp)
   1bce0:	01312623          	sw	s3,12(sp)
   1bce4:	01f5d913          	srli	s2,a1,0x1f
   1bce8:	01f6d693          	srli	a3,a3,0x1f
   1bcec:	01076733          	or	a4,a4,a6
   1bcf0:	00351513          	slli	a0,a0,0x3
   1bcf4:	00361613          	slli	a2,a2,0x3
   1bcf8:	41148833          	sub	a6,s1,a7
   1bcfc:	2ad91a63          	bne	s2,a3,1bfb0 <__adddf3+0x318>
   1bd00:	11005c63          	blez	a6,1be18 <__adddf3+0x180>
   1bd04:	04089063          	bnez	a7,1bd44 <__adddf3+0xac>
   1bd08:	00c766b3          	or	a3,a4,a2
   1bd0c:	66068063          	beqz	a3,1c36c <__adddf3+0x6d4>
   1bd10:	fff80593          	addi	a1,a6,-1
   1bd14:	02059063          	bnez	a1,1bd34 <__adddf3+0x9c>
   1bd18:	00c50633          	add	a2,a0,a2
   1bd1c:	00a636b3          	sltu	a3,a2,a0
   1bd20:	00e78733          	add	a4,a5,a4
   1bd24:	00060513          	mv	a0,a2
   1bd28:	00d707b3          	add	a5,a4,a3
   1bd2c:	00100493          	li	s1,1
   1bd30:	06c0006f          	j	1bd9c <__adddf3+0x104>
   1bd34:	7ff00693          	li	a3,2047
   1bd38:	02d81063          	bne	a6,a3,1bd58 <__adddf3+0xc0>
   1bd3c:	7ff00493          	li	s1,2047
   1bd40:	1f80006f          	j	1bf38 <__adddf3+0x2a0>
   1bd44:	7ff00693          	li	a3,2047
   1bd48:	1ed48863          	beq	s1,a3,1bf38 <__adddf3+0x2a0>
   1bd4c:	008006b7          	lui	a3,0x800
   1bd50:	00d76733          	or	a4,a4,a3
   1bd54:	00080593          	mv	a1,a6
   1bd58:	03800693          	li	a3,56
   1bd5c:	0ab6c863          	blt	a3,a1,1be0c <__adddf3+0x174>
   1bd60:	01f00693          	li	a3,31
   1bd64:	06b6ca63          	blt	a3,a1,1bdd8 <__adddf3+0x140>
   1bd68:	02000813          	li	a6,32
   1bd6c:	40b80833          	sub	a6,a6,a1
   1bd70:	010716b3          	sll	a3,a4,a6
   1bd74:	00b658b3          	srl	a7,a2,a1
   1bd78:	01061833          	sll	a6,a2,a6
   1bd7c:	0116e6b3          	or	a3,a3,a7
   1bd80:	01003833          	snez	a6,a6
   1bd84:	0106e6b3          	or	a3,a3,a6
   1bd88:	00b755b3          	srl	a1,a4,a1
   1bd8c:	00a68533          	add	a0,a3,a0
   1bd90:	00f585b3          	add	a1,a1,a5
   1bd94:	00d536b3          	sltu	a3,a0,a3
   1bd98:	00d587b3          	add	a5,a1,a3
   1bd9c:	00879713          	slli	a4,a5,0x8
   1bda0:	18075c63          	bgez	a4,1bf38 <__adddf3+0x2a0>
   1bda4:	00148493          	addi	s1,s1,1
   1bda8:	7ff00713          	li	a4,2047
   1bdac:	5ae48a63          	beq	s1,a4,1c360 <__adddf3+0x6c8>
   1bdb0:	ff800737          	lui	a4,0xff800
   1bdb4:	fff70713          	addi	a4,a4,-1 # ff7fffff <__BSS_END__+0xff7dd2cf>
   1bdb8:	00e7f733          	and	a4,a5,a4
   1bdbc:	00155793          	srli	a5,a0,0x1
   1bdc0:	00157513          	andi	a0,a0,1
   1bdc4:	00a7e7b3          	or	a5,a5,a0
   1bdc8:	01f71513          	slli	a0,a4,0x1f
   1bdcc:	00f56533          	or	a0,a0,a5
   1bdd0:	00175793          	srli	a5,a4,0x1
   1bdd4:	1640006f          	j	1bf38 <__adddf3+0x2a0>
   1bdd8:	fe058693          	addi	a3,a1,-32
   1bddc:	02000893          	li	a7,32
   1bde0:	00d756b3          	srl	a3,a4,a3
   1bde4:	00000813          	li	a6,0
   1bde8:	01158863          	beq	a1,a7,1bdf8 <__adddf3+0x160>
   1bdec:	04000813          	li	a6,64
   1bdf0:	40b80833          	sub	a6,a6,a1
   1bdf4:	01071833          	sll	a6,a4,a6
   1bdf8:	00c86833          	or	a6,a6,a2
   1bdfc:	01003833          	snez	a6,a6
   1be00:	0106e6b3          	or	a3,a3,a6
   1be04:	00000593          	li	a1,0
   1be08:	f85ff06f          	j	1bd8c <__adddf3+0xf4>
   1be0c:	00c766b3          	or	a3,a4,a2
   1be10:	00d036b3          	snez	a3,a3
   1be14:	ff1ff06f          	j	1be04 <__adddf3+0x16c>
   1be18:	0c080a63          	beqz	a6,1beec <__adddf3+0x254>
   1be1c:	409886b3          	sub	a3,a7,s1
   1be20:	02049463          	bnez	s1,1be48 <__adddf3+0x1b0>
   1be24:	00a7e5b3          	or	a1,a5,a0
   1be28:	50058e63          	beqz	a1,1c344 <__adddf3+0x6ac>
   1be2c:	fff68593          	addi	a1,a3,-1 # 7fffff <__BSS_END__+0x7dd2cf>
   1be30:	ee0584e3          	beqz	a1,1bd18 <__adddf3+0x80>
   1be34:	7ff00813          	li	a6,2047
   1be38:	03069263          	bne	a3,a6,1be5c <__adddf3+0x1c4>
   1be3c:	00070793          	mv	a5,a4
   1be40:	00060513          	mv	a0,a2
   1be44:	ef9ff06f          	j	1bd3c <__adddf3+0xa4>
   1be48:	7ff00593          	li	a1,2047
   1be4c:	feb888e3          	beq	a7,a1,1be3c <__adddf3+0x1a4>
   1be50:	008005b7          	lui	a1,0x800
   1be54:	00b7e7b3          	or	a5,a5,a1
   1be58:	00068593          	mv	a1,a3
   1be5c:	03800693          	li	a3,56
   1be60:	08b6c063          	blt	a3,a1,1bee0 <__adddf3+0x248>
   1be64:	01f00693          	li	a3,31
   1be68:	04b6c263          	blt	a3,a1,1beac <__adddf3+0x214>
   1be6c:	02000813          	li	a6,32
   1be70:	40b80833          	sub	a6,a6,a1
   1be74:	010796b3          	sll	a3,a5,a6
   1be78:	00b55333          	srl	t1,a0,a1
   1be7c:	01051833          	sll	a6,a0,a6
   1be80:	0066e6b3          	or	a3,a3,t1
   1be84:	01003833          	snez	a6,a6
   1be88:	0106e6b3          	or	a3,a3,a6
   1be8c:	00b7d5b3          	srl	a1,a5,a1
   1be90:	00c68633          	add	a2,a3,a2
   1be94:	00e585b3          	add	a1,a1,a4
   1be98:	00d636b3          	sltu	a3,a2,a3
   1be9c:	00060513          	mv	a0,a2
   1bea0:	00d587b3          	add	a5,a1,a3
   1bea4:	00088493          	mv	s1,a7
   1bea8:	ef5ff06f          	j	1bd9c <__adddf3+0x104>
   1beac:	fe058693          	addi	a3,a1,-32 # 7fffe0 <__BSS_END__+0x7dd2b0>
   1beb0:	02000313          	li	t1,32
   1beb4:	00d7d6b3          	srl	a3,a5,a3
   1beb8:	00000813          	li	a6,0
   1bebc:	00658863          	beq	a1,t1,1becc <__adddf3+0x234>
   1bec0:	04000813          	li	a6,64
   1bec4:	40b80833          	sub	a6,a6,a1
   1bec8:	01079833          	sll	a6,a5,a6
   1becc:	00a86833          	or	a6,a6,a0
   1bed0:	01003833          	snez	a6,a6
   1bed4:	0106e6b3          	or	a3,a3,a6
   1bed8:	00000593          	li	a1,0
   1bedc:	fb5ff06f          	j	1be90 <__adddf3+0x1f8>
   1bee0:	00a7e6b3          	or	a3,a5,a0
   1bee4:	00d036b3          	snez	a3,a3
   1bee8:	ff1ff06f          	j	1bed8 <__adddf3+0x240>
   1beec:	00148693          	addi	a3,s1,1
   1bef0:	7fe6f593          	andi	a1,a3,2046
   1bef4:	08059663          	bnez	a1,1bf80 <__adddf3+0x2e8>
   1bef8:	00a7e6b3          	or	a3,a5,a0
   1befc:	06049263          	bnez	s1,1bf60 <__adddf3+0x2c8>
   1bf00:	44068863          	beqz	a3,1c350 <__adddf3+0x6b8>
   1bf04:	00c766b3          	or	a3,a4,a2
   1bf08:	02068863          	beqz	a3,1bf38 <__adddf3+0x2a0>
   1bf0c:	00c50633          	add	a2,a0,a2
   1bf10:	00a636b3          	sltu	a3,a2,a0
   1bf14:	00e78733          	add	a4,a5,a4
   1bf18:	00d707b3          	add	a5,a4,a3
   1bf1c:	00879713          	slli	a4,a5,0x8
   1bf20:	00060513          	mv	a0,a2
   1bf24:	00075a63          	bgez	a4,1bf38 <__adddf3+0x2a0>
   1bf28:	ff800737          	lui	a4,0xff800
   1bf2c:	fff70713          	addi	a4,a4,-1 # ff7fffff <__BSS_END__+0xff7dd2cf>
   1bf30:	00e7f7b3          	and	a5,a5,a4
   1bf34:	00100493          	li	s1,1
   1bf38:	00757713          	andi	a4,a0,7
   1bf3c:	44070863          	beqz	a4,1c38c <__adddf3+0x6f4>
   1bf40:	00f57713          	andi	a4,a0,15
   1bf44:	00400693          	li	a3,4
   1bf48:	44d70263          	beq	a4,a3,1c38c <__adddf3+0x6f4>
   1bf4c:	00450713          	addi	a4,a0,4
   1bf50:	00a736b3          	sltu	a3,a4,a0
   1bf54:	00d787b3          	add	a5,a5,a3
   1bf58:	00070513          	mv	a0,a4
   1bf5c:	4300006f          	j	1c38c <__adddf3+0x6f4>
   1bf60:	ec068ee3          	beqz	a3,1be3c <__adddf3+0x1a4>
   1bf64:	00c76633          	or	a2,a4,a2
   1bf68:	dc060ae3          	beqz	a2,1bd3c <__adddf3+0xa4>
   1bf6c:	00000913          	li	s2,0
   1bf70:	004007b7          	lui	a5,0x400
   1bf74:	00000513          	li	a0,0
   1bf78:	7ff00493          	li	s1,2047
   1bf7c:	4100006f          	j	1c38c <__adddf3+0x6f4>
   1bf80:	7ff00593          	li	a1,2047
   1bf84:	3cb68c63          	beq	a3,a1,1c35c <__adddf3+0x6c4>
   1bf88:	00c50633          	add	a2,a0,a2
   1bf8c:	00a63533          	sltu	a0,a2,a0
   1bf90:	00e78733          	add	a4,a5,a4
   1bf94:	00a70733          	add	a4,a4,a0
   1bf98:	01f71513          	slli	a0,a4,0x1f
   1bf9c:	00165613          	srli	a2,a2,0x1
   1bfa0:	00c56533          	or	a0,a0,a2
   1bfa4:	00175793          	srli	a5,a4,0x1
   1bfa8:	00068493          	mv	s1,a3
   1bfac:	f8dff06f          	j	1bf38 <__adddf3+0x2a0>
   1bfb0:	0f005c63          	blez	a6,1c0a8 <__adddf3+0x410>
   1bfb4:	08089e63          	bnez	a7,1c050 <__adddf3+0x3b8>
   1bfb8:	00c766b3          	or	a3,a4,a2
   1bfbc:	3a068863          	beqz	a3,1c36c <__adddf3+0x6d4>
   1bfc0:	fff80693          	addi	a3,a6,-1
   1bfc4:	02069063          	bnez	a3,1bfe4 <__adddf3+0x34c>
   1bfc8:	40c50633          	sub	a2,a0,a2
   1bfcc:	00c536b3          	sltu	a3,a0,a2
   1bfd0:	40e78733          	sub	a4,a5,a4
   1bfd4:	00060513          	mv	a0,a2
   1bfd8:	40d707b3          	sub	a5,a4,a3
   1bfdc:	00100493          	li	s1,1
   1bfe0:	0540006f          	j	1c034 <__adddf3+0x39c>
   1bfe4:	7ff00593          	li	a1,2047
   1bfe8:	d4b80ae3          	beq	a6,a1,1bd3c <__adddf3+0xa4>
   1bfec:	03800593          	li	a1,56
   1bff0:	0ad5c663          	blt	a1,a3,1c09c <__adddf3+0x404>
   1bff4:	01f00593          	li	a1,31
   1bff8:	06d5c863          	blt	a1,a3,1c068 <__adddf3+0x3d0>
   1bffc:	02000813          	li	a6,32
   1c000:	40d80833          	sub	a6,a6,a3
   1c004:	00d658b3          	srl	a7,a2,a3
   1c008:	010715b3          	sll	a1,a4,a6
   1c00c:	01061833          	sll	a6,a2,a6
   1c010:	0115e5b3          	or	a1,a1,a7
   1c014:	01003833          	snez	a6,a6
   1c018:	0105e633          	or	a2,a1,a6
   1c01c:	00d756b3          	srl	a3,a4,a3
   1c020:	40c50633          	sub	a2,a0,a2
   1c024:	00c53733          	sltu	a4,a0,a2
   1c028:	40d786b3          	sub	a3,a5,a3
   1c02c:	00060513          	mv	a0,a2
   1c030:	40e687b3          	sub	a5,a3,a4
   1c034:	00879713          	slli	a4,a5,0x8
   1c038:	f00750e3          	bgez	a4,1bf38 <__adddf3+0x2a0>
   1c03c:	00800437          	lui	s0,0x800
   1c040:	fff40413          	addi	s0,s0,-1 # 7fffff <__BSS_END__+0x7dd2cf>
   1c044:	0087f433          	and	s0,a5,s0
   1c048:	00050993          	mv	s3,a0
   1c04c:	2100006f          	j	1c25c <__adddf3+0x5c4>
   1c050:	7ff00693          	li	a3,2047
   1c054:	eed482e3          	beq	s1,a3,1bf38 <__adddf3+0x2a0>
   1c058:	008006b7          	lui	a3,0x800
   1c05c:	00d76733          	or	a4,a4,a3
   1c060:	00080693          	mv	a3,a6
   1c064:	f89ff06f          	j	1bfec <__adddf3+0x354>
   1c068:	fe068593          	addi	a1,a3,-32 # 7fffe0 <__BSS_END__+0x7dd2b0>
   1c06c:	02000893          	li	a7,32
   1c070:	00b755b3          	srl	a1,a4,a1
   1c074:	00000813          	li	a6,0
   1c078:	01168863          	beq	a3,a7,1c088 <__adddf3+0x3f0>
   1c07c:	04000813          	li	a6,64
   1c080:	40d80833          	sub	a6,a6,a3
   1c084:	01071833          	sll	a6,a4,a6
   1c088:	00c86833          	or	a6,a6,a2
   1c08c:	01003833          	snez	a6,a6
   1c090:	0105e633          	or	a2,a1,a6
   1c094:	00000693          	li	a3,0
   1c098:	f89ff06f          	j	1c020 <__adddf3+0x388>
   1c09c:	00c76633          	or	a2,a4,a2
   1c0a0:	00c03633          	snez	a2,a2
   1c0a4:	ff1ff06f          	j	1c094 <__adddf3+0x3fc>
   1c0a8:	0e080863          	beqz	a6,1c198 <__adddf3+0x500>
   1c0ac:	40988833          	sub	a6,a7,s1
   1c0b0:	04049263          	bnez	s1,1c0f4 <__adddf3+0x45c>
   1c0b4:	00a7e5b3          	or	a1,a5,a0
   1c0b8:	2a058e63          	beqz	a1,1c374 <__adddf3+0x6dc>
   1c0bc:	fff80593          	addi	a1,a6,-1
   1c0c0:	00059e63          	bnez	a1,1c0dc <__adddf3+0x444>
   1c0c4:	40a60533          	sub	a0,a2,a0
   1c0c8:	40f70733          	sub	a4,a4,a5
   1c0cc:	00a63633          	sltu	a2,a2,a0
   1c0d0:	40c707b3          	sub	a5,a4,a2
   1c0d4:	00068913          	mv	s2,a3
   1c0d8:	f05ff06f          	j	1bfdc <__adddf3+0x344>
   1c0dc:	7ff00313          	li	t1,2047
   1c0e0:	02681463          	bne	a6,t1,1c108 <__adddf3+0x470>
   1c0e4:	00070793          	mv	a5,a4
   1c0e8:	00060513          	mv	a0,a2
   1c0ec:	7ff00493          	li	s1,2047
   1c0f0:	0d00006f          	j	1c1c0 <__adddf3+0x528>
   1c0f4:	7ff00593          	li	a1,2047
   1c0f8:	feb886e3          	beq	a7,a1,1c0e4 <__adddf3+0x44c>
   1c0fc:	008005b7          	lui	a1,0x800
   1c100:	00b7e7b3          	or	a5,a5,a1
   1c104:	00080593          	mv	a1,a6
   1c108:	03800813          	li	a6,56
   1c10c:	08b84063          	blt	a6,a1,1c18c <__adddf3+0x4f4>
   1c110:	01f00813          	li	a6,31
   1c114:	04b84263          	blt	a6,a1,1c158 <__adddf3+0x4c0>
   1c118:	02000313          	li	t1,32
   1c11c:	40b30333          	sub	t1,t1,a1
   1c120:	00b55e33          	srl	t3,a0,a1
   1c124:	00679833          	sll	a6,a5,t1
   1c128:	00651333          	sll	t1,a0,t1
   1c12c:	01c86833          	or	a6,a6,t3
   1c130:	00603333          	snez	t1,t1
   1c134:	00686533          	or	a0,a6,t1
   1c138:	00b7d5b3          	srl	a1,a5,a1
   1c13c:	40a60533          	sub	a0,a2,a0
   1c140:	40b705b3          	sub	a1,a4,a1
   1c144:	00a63633          	sltu	a2,a2,a0
   1c148:	40c587b3          	sub	a5,a1,a2
   1c14c:	00088493          	mv	s1,a7
   1c150:	00068913          	mv	s2,a3
   1c154:	ee1ff06f          	j	1c034 <__adddf3+0x39c>
   1c158:	fe058813          	addi	a6,a1,-32 # 7fffe0 <__BSS_END__+0x7dd2b0>
   1c15c:	02000e13          	li	t3,32
   1c160:	0107d833          	srl	a6,a5,a6
   1c164:	00000313          	li	t1,0
   1c168:	01c58863          	beq	a1,t3,1c178 <__adddf3+0x4e0>
   1c16c:	04000313          	li	t1,64
   1c170:	40b30333          	sub	t1,t1,a1
   1c174:	00679333          	sll	t1,a5,t1
   1c178:	00a36333          	or	t1,t1,a0
   1c17c:	00603333          	snez	t1,t1
   1c180:	00686533          	or	a0,a6,t1
   1c184:	00000593          	li	a1,0
   1c188:	fb5ff06f          	j	1c13c <__adddf3+0x4a4>
   1c18c:	00a7e533          	or	a0,a5,a0
   1c190:	00a03533          	snez	a0,a0
   1c194:	ff1ff06f          	j	1c184 <__adddf3+0x4ec>
   1c198:	00148593          	addi	a1,s1,1
   1c19c:	7fe5f593          	andi	a1,a1,2046
   1c1a0:	08059663          	bnez	a1,1c22c <__adddf3+0x594>
   1c1a4:	00a7e833          	or	a6,a5,a0
   1c1a8:	00c765b3          	or	a1,a4,a2
   1c1ac:	06049063          	bnez	s1,1c20c <__adddf3+0x574>
   1c1b0:	00081c63          	bnez	a6,1c1c8 <__adddf3+0x530>
   1c1b4:	10058e63          	beqz	a1,1c2d0 <__adddf3+0x638>
   1c1b8:	00070793          	mv	a5,a4
   1c1bc:	00060513          	mv	a0,a2
   1c1c0:	00068913          	mv	s2,a3
   1c1c4:	d75ff06f          	j	1bf38 <__adddf3+0x2a0>
   1c1c8:	d60588e3          	beqz	a1,1bf38 <__adddf3+0x2a0>
   1c1cc:	40c50833          	sub	a6,a0,a2
   1c1d0:	010538b3          	sltu	a7,a0,a6
   1c1d4:	40e785b3          	sub	a1,a5,a4
   1c1d8:	411585b3          	sub	a1,a1,a7
   1c1dc:	00859893          	slli	a7,a1,0x8
   1c1e0:	0008dc63          	bgez	a7,1c1f8 <__adddf3+0x560>
   1c1e4:	40a60533          	sub	a0,a2,a0
   1c1e8:	40f70733          	sub	a4,a4,a5
   1c1ec:	00a63633          	sltu	a2,a2,a0
   1c1f0:	40c707b3          	sub	a5,a4,a2
   1c1f4:	fcdff06f          	j	1c1c0 <__adddf3+0x528>
   1c1f8:	00b86533          	or	a0,a6,a1
   1c1fc:	18050463          	beqz	a0,1c384 <__adddf3+0x6ec>
   1c200:	00058793          	mv	a5,a1
   1c204:	00080513          	mv	a0,a6
   1c208:	d31ff06f          	j	1bf38 <__adddf3+0x2a0>
   1c20c:	00081c63          	bnez	a6,1c224 <__adddf3+0x58c>
   1c210:	d4058ee3          	beqz	a1,1bf6c <__adddf3+0x2d4>
   1c214:	00070793          	mv	a5,a4
   1c218:	00060513          	mv	a0,a2
   1c21c:	00068913          	mv	s2,a3
   1c220:	b1dff06f          	j	1bd3c <__adddf3+0xa4>
   1c224:	b0058ce3          	beqz	a1,1bd3c <__adddf3+0xa4>
   1c228:	d45ff06f          	j	1bf6c <__adddf3+0x2d4>
   1c22c:	40c505b3          	sub	a1,a0,a2
   1c230:	00b53833          	sltu	a6,a0,a1
   1c234:	40e78433          	sub	s0,a5,a4
   1c238:	41040433          	sub	s0,s0,a6
   1c23c:	00841813          	slli	a6,s0,0x8
   1c240:	00058993          	mv	s3,a1
   1c244:	08085063          	bgez	a6,1c2c4 <__adddf3+0x62c>
   1c248:	40a609b3          	sub	s3,a2,a0
   1c24c:	40f70433          	sub	s0,a4,a5
   1c250:	01363633          	sltu	a2,a2,s3
   1c254:	40c40433          	sub	s0,s0,a2
   1c258:	00068913          	mv	s2,a3
   1c25c:	06040e63          	beqz	s0,1c2d8 <__adddf3+0x640>
   1c260:	00040513          	mv	a0,s0
   1c264:	650040ef          	jal	208b4 <__clzsi2>
   1c268:	ff850693          	addi	a3,a0,-8
   1c26c:	02000793          	li	a5,32
   1c270:	40d787b3          	sub	a5,a5,a3
   1c274:	00d41433          	sll	s0,s0,a3
   1c278:	00f9d7b3          	srl	a5,s3,a5
   1c27c:	0087e7b3          	or	a5,a5,s0
   1c280:	00d99433          	sll	s0,s3,a3
   1c284:	0a96c463          	blt	a3,s1,1c32c <__adddf3+0x694>
   1c288:	409686b3          	sub	a3,a3,s1
   1c28c:	00168613          	addi	a2,a3,1
   1c290:	01f00713          	li	a4,31
   1c294:	06c74263          	blt	a4,a2,1c2f8 <__adddf3+0x660>
   1c298:	02000713          	li	a4,32
   1c29c:	40c70733          	sub	a4,a4,a2
   1c2a0:	00e79533          	sll	a0,a5,a4
   1c2a4:	00c456b3          	srl	a3,s0,a2
   1c2a8:	00e41733          	sll	a4,s0,a4
   1c2ac:	00d56533          	or	a0,a0,a3
   1c2b0:	00e03733          	snez	a4,a4
   1c2b4:	00e56533          	or	a0,a0,a4
   1c2b8:	00c7d7b3          	srl	a5,a5,a2
   1c2bc:	00000493          	li	s1,0
   1c2c0:	c79ff06f          	j	1bf38 <__adddf3+0x2a0>
   1c2c4:	0085e5b3          	or	a1,a1,s0
   1c2c8:	f8059ae3          	bnez	a1,1c25c <__adddf3+0x5c4>
   1c2cc:	00000493          	li	s1,0
   1c2d0:	00000913          	li	s2,0
   1c2d4:	08c0006f          	j	1c360 <__adddf3+0x6c8>
   1c2d8:	00098513          	mv	a0,s3
   1c2dc:	5d8040ef          	jal	208b4 <__clzsi2>
   1c2e0:	01850693          	addi	a3,a0,24
   1c2e4:	01f00793          	li	a5,31
   1c2e8:	f8d7d2e3          	bge	a5,a3,1c26c <__adddf3+0x5d4>
   1c2ec:	ff850793          	addi	a5,a0,-8
   1c2f0:	00f997b3          	sll	a5,s3,a5
   1c2f4:	f91ff06f          	j	1c284 <__adddf3+0x5ec>
   1c2f8:	fe168693          	addi	a3,a3,-31
   1c2fc:	00d7d533          	srl	a0,a5,a3
   1c300:	02000693          	li	a3,32
   1c304:	00000713          	li	a4,0
   1c308:	00d60863          	beq	a2,a3,1c318 <__adddf3+0x680>
   1c30c:	04000713          	li	a4,64
   1c310:	40c70733          	sub	a4,a4,a2
   1c314:	00e79733          	sll	a4,a5,a4
   1c318:	00e46733          	or	a4,s0,a4
   1c31c:	00e03733          	snez	a4,a4
   1c320:	00e56533          	or	a0,a0,a4
   1c324:	00000793          	li	a5,0
   1c328:	f95ff06f          	j	1c2bc <__adddf3+0x624>
   1c32c:	ff800737          	lui	a4,0xff800
   1c330:	fff70713          	addi	a4,a4,-1 # ff7fffff <__BSS_END__+0xff7dd2cf>
   1c334:	40d484b3          	sub	s1,s1,a3
   1c338:	00e7f7b3          	and	a5,a5,a4
   1c33c:	00040513          	mv	a0,s0
   1c340:	bf9ff06f          	j	1bf38 <__adddf3+0x2a0>
   1c344:	00070793          	mv	a5,a4
   1c348:	00060513          	mv	a0,a2
   1c34c:	c5dff06f          	j	1bfa8 <__adddf3+0x310>
   1c350:	00070793          	mv	a5,a4
   1c354:	00060513          	mv	a0,a2
   1c358:	be1ff06f          	j	1bf38 <__adddf3+0x2a0>
   1c35c:	7ff00493          	li	s1,2047
   1c360:	00000793          	li	a5,0
   1c364:	00000513          	li	a0,0
   1c368:	0240006f          	j	1c38c <__adddf3+0x6f4>
   1c36c:	00080493          	mv	s1,a6
   1c370:	bc9ff06f          	j	1bf38 <__adddf3+0x2a0>
   1c374:	00070793          	mv	a5,a4
   1c378:	00060513          	mv	a0,a2
   1c37c:	00080493          	mv	s1,a6
   1c380:	e41ff06f          	j	1c1c0 <__adddf3+0x528>
   1c384:	00000793          	li	a5,0
   1c388:	00000913          	li	s2,0
   1c38c:	00879713          	slli	a4,a5,0x8
   1c390:	00075e63          	bgez	a4,1c3ac <__adddf3+0x714>
   1c394:	00148493          	addi	s1,s1,1
   1c398:	7ff00713          	li	a4,2047
   1c39c:	08e48263          	beq	s1,a4,1c420 <__adddf3+0x788>
   1c3a0:	ff800737          	lui	a4,0xff800
   1c3a4:	fff70713          	addi	a4,a4,-1 # ff7fffff <__BSS_END__+0xff7dd2cf>
   1c3a8:	00e7f7b3          	and	a5,a5,a4
   1c3ac:	01d79693          	slli	a3,a5,0x1d
   1c3b0:	00355513          	srli	a0,a0,0x3
   1c3b4:	7ff00713          	li	a4,2047
   1c3b8:	00a6e6b3          	or	a3,a3,a0
   1c3bc:	0037d793          	srli	a5,a5,0x3
   1c3c0:	00e49e63          	bne	s1,a4,1c3dc <__adddf3+0x744>
   1c3c4:	00f6e6b3          	or	a3,a3,a5
   1c3c8:	00000793          	li	a5,0
   1c3cc:	00068863          	beqz	a3,1c3dc <__adddf3+0x744>
   1c3d0:	000807b7          	lui	a5,0x80
   1c3d4:	00000693          	li	a3,0
   1c3d8:	00000913          	li	s2,0
   1c3dc:	01449713          	slli	a4,s1,0x14
   1c3e0:	7ff00637          	lui	a2,0x7ff00
   1c3e4:	00c79793          	slli	a5,a5,0xc
   1c3e8:	00c77733          	and	a4,a4,a2
   1c3ec:	01c12083          	lw	ra,28(sp)
   1c3f0:	01812403          	lw	s0,24(sp)
   1c3f4:	00c7d793          	srli	a5,a5,0xc
   1c3f8:	00f767b3          	or	a5,a4,a5
   1c3fc:	01f91713          	slli	a4,s2,0x1f
   1c400:	00e7e633          	or	a2,a5,a4
   1c404:	01412483          	lw	s1,20(sp)
   1c408:	01012903          	lw	s2,16(sp)
   1c40c:	00c12983          	lw	s3,12(sp)
   1c410:	00068513          	mv	a0,a3
   1c414:	00060593          	mv	a1,a2
   1c418:	02010113          	addi	sp,sp,32
   1c41c:	00008067          	ret
   1c420:	00000793          	li	a5,0
   1c424:	00000513          	li	a0,0
   1c428:	f85ff06f          	j	1c3ac <__adddf3+0x714>

0001c42c <__divdf3>:
   1c42c:	fd010113          	addi	sp,sp,-48
   1c430:	0145d813          	srli	a6,a1,0x14
   1c434:	02912223          	sw	s1,36(sp)
   1c438:	03212023          	sw	s2,32(sp)
   1c43c:	01312e23          	sw	s3,28(sp)
   1c440:	01612823          	sw	s6,16(sp)
   1c444:	01712623          	sw	s7,12(sp)
   1c448:	00c59493          	slli	s1,a1,0xc
   1c44c:	02112623          	sw	ra,44(sp)
   1c450:	02812423          	sw	s0,40(sp)
   1c454:	01412c23          	sw	s4,24(sp)
   1c458:	01512a23          	sw	s5,20(sp)
   1c45c:	7ff87813          	andi	a6,a6,2047
   1c460:	00050b13          	mv	s6,a0
   1c464:	00060b93          	mv	s7,a2
   1c468:	00068913          	mv	s2,a3
   1c46c:	00c4d493          	srli	s1,s1,0xc
   1c470:	01f5d993          	srli	s3,a1,0x1f
   1c474:	0a080263          	beqz	a6,1c518 <__divdf3+0xec>
   1c478:	7ff00793          	li	a5,2047
   1c47c:	10f80263          	beq	a6,a5,1c580 <__divdf3+0x154>
   1c480:	01d55a13          	srli	s4,a0,0x1d
   1c484:	00349493          	slli	s1,s1,0x3
   1c488:	009a6a33          	or	s4,s4,s1
   1c48c:	008007b7          	lui	a5,0x800
   1c490:	00fa6a33          	or	s4,s4,a5
   1c494:	00351413          	slli	s0,a0,0x3
   1c498:	c0180a93          	addi	s5,a6,-1023
   1c49c:	00000b13          	li	s6,0
   1c4a0:	01495713          	srli	a4,s2,0x14
   1c4a4:	00c91493          	slli	s1,s2,0xc
   1c4a8:	7ff77713          	andi	a4,a4,2047
   1c4ac:	00c4d493          	srli	s1,s1,0xc
   1c4b0:	01f95913          	srli	s2,s2,0x1f
   1c4b4:	10070463          	beqz	a4,1c5bc <__divdf3+0x190>
   1c4b8:	7ff00793          	li	a5,2047
   1c4bc:	16f70863          	beq	a4,a5,1c62c <__divdf3+0x200>
   1c4c0:	00349493          	slli	s1,s1,0x3
   1c4c4:	01dbd793          	srli	a5,s7,0x1d
   1c4c8:	0097e7b3          	or	a5,a5,s1
   1c4cc:	008004b7          	lui	s1,0x800
   1c4d0:	0097e4b3          	or	s1,a5,s1
   1c4d4:	003b9f13          	slli	t5,s7,0x3
   1c4d8:	c0170713          	addi	a4,a4,-1023
   1c4dc:	00000793          	li	a5,0
   1c4e0:	40ea8833          	sub	a6,s5,a4
   1c4e4:	002b1713          	slli	a4,s6,0x2
   1c4e8:	00f76733          	or	a4,a4,a5
   1c4ec:	fff70713          	addi	a4,a4,-1
   1c4f0:	00e00693          	li	a3,14
   1c4f4:	0129c633          	xor	a2,s3,s2
   1c4f8:	16e6e663          	bltu	a3,a4,1c664 <__divdf3+0x238>
   1c4fc:	00005697          	auipc	a3,0x5
   1c500:	de068693          	addi	a3,a3,-544 # 212dc <_ctype_+0x104>
   1c504:	00271713          	slli	a4,a4,0x2
   1c508:	00d70733          	add	a4,a4,a3
   1c50c:	00072703          	lw	a4,0(a4)
   1c510:	00d70733          	add	a4,a4,a3
   1c514:	00070067          	jr	a4
   1c518:	00a4ea33          	or	s4,s1,a0
   1c51c:	060a0e63          	beqz	s4,1c598 <__divdf3+0x16c>
   1c520:	02048e63          	beqz	s1,1c55c <__divdf3+0x130>
   1c524:	00048513          	mv	a0,s1
   1c528:	38c040ef          	jal	208b4 <__clzsi2>
   1c52c:	ff550793          	addi	a5,a0,-11
   1c530:	01d00a13          	li	s4,29
   1c534:	ff850713          	addi	a4,a0,-8
   1c538:	40fa0a33          	sub	s4,s4,a5
   1c53c:	00e494b3          	sll	s1,s1,a4
   1c540:	014b5a33          	srl	s4,s6,s4
   1c544:	009a6a33          	or	s4,s4,s1
   1c548:	00eb14b3          	sll	s1,s6,a4
   1c54c:	c0d00813          	li	a6,-1011
   1c550:	40a80ab3          	sub	s5,a6,a0
   1c554:	00048413          	mv	s0,s1
   1c558:	f45ff06f          	j	1c49c <__divdf3+0x70>
   1c55c:	358040ef          	jal	208b4 <__clzsi2>
   1c560:	00050a13          	mv	s4,a0
   1c564:	015a0793          	addi	a5,s4,21
   1c568:	01c00713          	li	a4,28
   1c56c:	02050513          	addi	a0,a0,32
   1c570:	fcf750e3          	bge	a4,a5,1c530 <__divdf3+0x104>
   1c574:	ff8a0a13          	addi	s4,s4,-8
   1c578:	014b1a33          	sll	s4,s6,s4
   1c57c:	fd1ff06f          	j	1c54c <__divdf3+0x120>
   1c580:	00a4ea33          	or	s4,s1,a0
   1c584:	020a1263          	bnez	s4,1c5a8 <__divdf3+0x17c>
   1c588:	00000413          	li	s0,0
   1c58c:	7ff00a93          	li	s5,2047
   1c590:	00200b13          	li	s6,2
   1c594:	f0dff06f          	j	1c4a0 <__divdf3+0x74>
   1c598:	00000413          	li	s0,0
   1c59c:	00000a93          	li	s5,0
   1c5a0:	00100b13          	li	s6,1
   1c5a4:	efdff06f          	j	1c4a0 <__divdf3+0x74>
   1c5a8:	00050413          	mv	s0,a0
   1c5ac:	00048a13          	mv	s4,s1
   1c5b0:	7ff00a93          	li	s5,2047
   1c5b4:	00300b13          	li	s6,3
   1c5b8:	ee9ff06f          	j	1c4a0 <__divdf3+0x74>
   1c5bc:	0174ef33          	or	t5,s1,s7
   1c5c0:	080f0263          	beqz	t5,1c644 <__divdf3+0x218>
   1c5c4:	04048063          	beqz	s1,1c604 <__divdf3+0x1d8>
   1c5c8:	00048513          	mv	a0,s1
   1c5cc:	2e8040ef          	jal	208b4 <__clzsi2>
   1c5d0:	ff550713          	addi	a4,a0,-11
   1c5d4:	01d00793          	li	a5,29
   1c5d8:	ff850693          	addi	a3,a0,-8
   1c5dc:	40e787b3          	sub	a5,a5,a4
   1c5e0:	00d494b3          	sll	s1,s1,a3
   1c5e4:	00fbd7b3          	srl	a5,s7,a5
   1c5e8:	0097e7b3          	or	a5,a5,s1
   1c5ec:	00db94b3          	sll	s1,s7,a3
   1c5f0:	c0d00713          	li	a4,-1011
   1c5f4:	00048f13          	mv	t5,s1
   1c5f8:	40a70733          	sub	a4,a4,a0
   1c5fc:	00078493          	mv	s1,a5
   1c600:	eddff06f          	j	1c4dc <__divdf3+0xb0>
   1c604:	000b8513          	mv	a0,s7
   1c608:	2ac040ef          	jal	208b4 <__clzsi2>
   1c60c:	00050793          	mv	a5,a0
   1c610:	01578713          	addi	a4,a5,21 # 800015 <__BSS_END__+0x7dd2e5>
   1c614:	01c00693          	li	a3,28
   1c618:	02050513          	addi	a0,a0,32
   1c61c:	fae6dce3          	bge	a3,a4,1c5d4 <__divdf3+0x1a8>
   1c620:	ff878793          	addi	a5,a5,-8
   1c624:	00fb97b3          	sll	a5,s7,a5
   1c628:	fc9ff06f          	j	1c5f0 <__divdf3+0x1c4>
   1c62c:	0174ef33          	or	t5,s1,s7
   1c630:	020f1263          	bnez	t5,1c654 <__divdf3+0x228>
   1c634:	00000493          	li	s1,0
   1c638:	7ff00713          	li	a4,2047
   1c63c:	00200793          	li	a5,2
   1c640:	ea1ff06f          	j	1c4e0 <__divdf3+0xb4>
   1c644:	00000493          	li	s1,0
   1c648:	00000713          	li	a4,0
   1c64c:	00100793          	li	a5,1
   1c650:	e91ff06f          	j	1c4e0 <__divdf3+0xb4>
   1c654:	000b8f13          	mv	t5,s7
   1c658:	7ff00713          	li	a4,2047
   1c65c:	00300793          	li	a5,3
   1c660:	e81ff06f          	j	1c4e0 <__divdf3+0xb4>
   1c664:	0144e663          	bltu	s1,s4,1c670 <__divdf3+0x244>
   1c668:	349a1c63          	bne	s4,s1,1c9c0 <__divdf3+0x594>
   1c66c:	35e46a63          	bltu	s0,t5,1c9c0 <__divdf3+0x594>
   1c670:	01fa1693          	slli	a3,s4,0x1f
   1c674:	00145793          	srli	a5,s0,0x1
   1c678:	01f41713          	slli	a4,s0,0x1f
   1c67c:	001a5a13          	srli	s4,s4,0x1
   1c680:	00f6e433          	or	s0,a3,a5
   1c684:	00849893          	slli	a7,s1,0x8
   1c688:	018f5593          	srli	a1,t5,0x18
   1c68c:	0115e5b3          	or	a1,a1,a7
   1c690:	0108d893          	srli	a7,a7,0x10
   1c694:	031a5eb3          	divu	t4,s4,a7
   1c698:	01059313          	slli	t1,a1,0x10
   1c69c:	01035313          	srli	t1,t1,0x10
   1c6a0:	01045793          	srli	a5,s0,0x10
   1c6a4:	008f1513          	slli	a0,t5,0x8
   1c6a8:	031a7a33          	remu	s4,s4,a7
   1c6ac:	000e8693          	mv	a3,t4
   1c6b0:	03d30e33          	mul	t3,t1,t4
   1c6b4:	010a1a13          	slli	s4,s4,0x10
   1c6b8:	0147e7b3          	or	a5,a5,s4
   1c6bc:	01c7fe63          	bgeu	a5,t3,1c6d8 <__divdf3+0x2ac>
   1c6c0:	00f587b3          	add	a5,a1,a5
   1c6c4:	fffe8693          	addi	a3,t4,-1
   1c6c8:	00b7e863          	bltu	a5,a1,1c6d8 <__divdf3+0x2ac>
   1c6cc:	01c7f663          	bgeu	a5,t3,1c6d8 <__divdf3+0x2ac>
   1c6d0:	ffee8693          	addi	a3,t4,-2
   1c6d4:	00b787b3          	add	a5,a5,a1
   1c6d8:	41c787b3          	sub	a5,a5,t3
   1c6dc:	0317df33          	divu	t5,a5,a7
   1c6e0:	01041413          	slli	s0,s0,0x10
   1c6e4:	01045413          	srli	s0,s0,0x10
   1c6e8:	0317f7b3          	remu	a5,a5,a7
   1c6ec:	000f0e13          	mv	t3,t5
   1c6f0:	03e30eb3          	mul	t4,t1,t5
   1c6f4:	01079793          	slli	a5,a5,0x10
   1c6f8:	00f467b3          	or	a5,s0,a5
   1c6fc:	01d7fe63          	bgeu	a5,t4,1c718 <__divdf3+0x2ec>
   1c700:	00f587b3          	add	a5,a1,a5
   1c704:	ffff0e13          	addi	t3,t5,-1
   1c708:	00b7e863          	bltu	a5,a1,1c718 <__divdf3+0x2ec>
   1c70c:	01d7f663          	bgeu	a5,t4,1c718 <__divdf3+0x2ec>
   1c710:	ffef0e13          	addi	t3,t5,-2
   1c714:	00b787b3          	add	a5,a5,a1
   1c718:	01069693          	slli	a3,a3,0x10
   1c71c:	00010437          	lui	s0,0x10
   1c720:	01c6e2b3          	or	t0,a3,t3
   1c724:	fff40e13          	addi	t3,s0,-1 # ffff <exit-0xb5>
   1c728:	01c2f6b3          	and	a3,t0,t3
   1c72c:	0102df93          	srli	t6,t0,0x10
   1c730:	01c57e33          	and	t3,a0,t3
   1c734:	41d787b3          	sub	a5,a5,t4
   1c738:	01055e93          	srli	t4,a0,0x10
   1c73c:	02de03b3          	mul	t2,t3,a3
   1c740:	03cf84b3          	mul	s1,t6,t3
   1c744:	02de86b3          	mul	a3,t4,a3
   1c748:	00968f33          	add	t5,a3,s1
   1c74c:	0103d693          	srli	a3,t2,0x10
   1c750:	01e686b3          	add	a3,a3,t5
   1c754:	03df8fb3          	mul	t6,t6,t4
   1c758:	0096f463          	bgeu	a3,s1,1c760 <__divdf3+0x334>
   1c75c:	008f8fb3          	add	t6,t6,s0
   1c760:	0106df13          	srli	t5,a3,0x10
   1c764:	01ff0f33          	add	t5,t5,t6
   1c768:	00010fb7          	lui	t6,0x10
   1c76c:	ffff8f93          	addi	t6,t6,-1 # ffff <exit-0xb5>
   1c770:	01f6f6b3          	and	a3,a3,t6
   1c774:	01069693          	slli	a3,a3,0x10
   1c778:	01f3f3b3          	and	t2,t2,t6
   1c77c:	007686b3          	add	a3,a3,t2
   1c780:	01e7e863          	bltu	a5,t5,1c790 <__divdf3+0x364>
   1c784:	00028493          	mv	s1,t0
   1c788:	05e79863          	bne	a5,t5,1c7d8 <__divdf3+0x3ac>
   1c78c:	04d77663          	bgeu	a4,a3,1c7d8 <__divdf3+0x3ac>
   1c790:	00a70fb3          	add	t6,a4,a0
   1c794:	00efb3b3          	sltu	t2,t6,a4
   1c798:	00b38433          	add	s0,t2,a1
   1c79c:	008787b3          	add	a5,a5,s0
   1c7a0:	fff28493          	addi	s1,t0,-1
   1c7a4:	000f8713          	mv	a4,t6
   1c7a8:	00f5e663          	bltu	a1,a5,1c7b4 <__divdf3+0x388>
   1c7ac:	02f59663          	bne	a1,a5,1c7d8 <__divdf3+0x3ac>
   1c7b0:	02039463          	bnez	t2,1c7d8 <__divdf3+0x3ac>
   1c7b4:	01e7e663          	bltu	a5,t5,1c7c0 <__divdf3+0x394>
   1c7b8:	02ff1063          	bne	t5,a5,1c7d8 <__divdf3+0x3ac>
   1c7bc:	00dffe63          	bgeu	t6,a3,1c7d8 <__divdf3+0x3ac>
   1c7c0:	01f50fb3          	add	t6,a0,t6
   1c7c4:	000f8713          	mv	a4,t6
   1c7c8:	00afbfb3          	sltu	t6,t6,a0
   1c7cc:	00bf8fb3          	add	t6,t6,a1
   1c7d0:	ffe28493          	addi	s1,t0,-2
   1c7d4:	01f787b3          	add	a5,a5,t6
   1c7d8:	40d706b3          	sub	a3,a4,a3
   1c7dc:	41e787b3          	sub	a5,a5,t5
   1c7e0:	00d73733          	sltu	a4,a4,a3
   1c7e4:	40e787b3          	sub	a5,a5,a4
   1c7e8:	fff00f13          	li	t5,-1
   1c7ec:	12f58663          	beq	a1,a5,1c918 <__divdf3+0x4ec>
   1c7f0:	0317dfb3          	divu	t6,a5,a7
   1c7f4:	0106d713          	srli	a4,a3,0x10
   1c7f8:	0317f7b3          	remu	a5,a5,a7
   1c7fc:	03f30f33          	mul	t5,t1,t6
   1c800:	01079793          	slli	a5,a5,0x10
   1c804:	00f767b3          	or	a5,a4,a5
   1c808:	000f8713          	mv	a4,t6
   1c80c:	01e7fe63          	bgeu	a5,t5,1c828 <__divdf3+0x3fc>
   1c810:	00f587b3          	add	a5,a1,a5
   1c814:	ffff8713          	addi	a4,t6,-1
   1c818:	00b7e863          	bltu	a5,a1,1c828 <__divdf3+0x3fc>
   1c81c:	01e7f663          	bgeu	a5,t5,1c828 <__divdf3+0x3fc>
   1c820:	ffef8713          	addi	a4,t6,-2
   1c824:	00b787b3          	add	a5,a5,a1
   1c828:	41e787b3          	sub	a5,a5,t5
   1c82c:	0317df33          	divu	t5,a5,a7
   1c830:	01069693          	slli	a3,a3,0x10
   1c834:	0106d693          	srli	a3,a3,0x10
   1c838:	0317f7b3          	remu	a5,a5,a7
   1c83c:	000f0893          	mv	a7,t5
   1c840:	03e30333          	mul	t1,t1,t5
   1c844:	01079793          	slli	a5,a5,0x10
   1c848:	00f6e7b3          	or	a5,a3,a5
   1c84c:	0067fe63          	bgeu	a5,t1,1c868 <__divdf3+0x43c>
   1c850:	00f587b3          	add	a5,a1,a5
   1c854:	ffff0893          	addi	a7,t5,-1
   1c858:	00b7e863          	bltu	a5,a1,1c868 <__divdf3+0x43c>
   1c85c:	0067f663          	bgeu	a5,t1,1c868 <__divdf3+0x43c>
   1c860:	ffef0893          	addi	a7,t5,-2
   1c864:	00b787b3          	add	a5,a5,a1
   1c868:	01071693          	slli	a3,a4,0x10
   1c86c:	0116e6b3          	or	a3,a3,a7
   1c870:	01069713          	slli	a4,a3,0x10
   1c874:	01075713          	srli	a4,a4,0x10
   1c878:	406787b3          	sub	a5,a5,t1
   1c87c:	0106d313          	srli	t1,a3,0x10
   1c880:	03c70f33          	mul	t5,a4,t3
   1c884:	03c30e33          	mul	t3,t1,t3
   1c888:	026e8333          	mul	t1,t4,t1
   1c88c:	02ee8eb3          	mul	t4,t4,a4
   1c890:	010f5713          	srli	a4,t5,0x10
   1c894:	01ce8eb3          	add	t4,t4,t3
   1c898:	01d70733          	add	a4,a4,t4
   1c89c:	01c77663          	bgeu	a4,t3,1c8a8 <__divdf3+0x47c>
   1c8a0:	000108b7          	lui	a7,0x10
   1c8a4:	01130333          	add	t1,t1,a7
   1c8a8:	01075893          	srli	a7,a4,0x10
   1c8ac:	006888b3          	add	a7,a7,t1
   1c8b0:	00010337          	lui	t1,0x10
   1c8b4:	fff30313          	addi	t1,t1,-1 # ffff <exit-0xb5>
   1c8b8:	00677733          	and	a4,a4,t1
   1c8bc:	01071713          	slli	a4,a4,0x10
   1c8c0:	006f7f33          	and	t5,t5,t1
   1c8c4:	01e70733          	add	a4,a4,t5
   1c8c8:	0117e863          	bltu	a5,a7,1c8d8 <__divdf3+0x4ac>
   1c8cc:	23179c63          	bne	a5,a7,1cb04 <__divdf3+0x6d8>
   1c8d0:	00068f13          	mv	t5,a3
   1c8d4:	04070263          	beqz	a4,1c918 <__divdf3+0x4ec>
   1c8d8:	00f587b3          	add	a5,a1,a5
   1c8dc:	fff68f13          	addi	t5,a3,-1
   1c8e0:	00078313          	mv	t1,a5
   1c8e4:	02b7e463          	bltu	a5,a1,1c90c <__divdf3+0x4e0>
   1c8e8:	0117e663          	bltu	a5,a7,1c8f4 <__divdf3+0x4c8>
   1c8ec:	21179a63          	bne	a5,a7,1cb00 <__divdf3+0x6d4>
   1c8f0:	02e57063          	bgeu	a0,a4,1c910 <__divdf3+0x4e4>
   1c8f4:	ffe68f13          	addi	t5,a3,-2
   1c8f8:	00151693          	slli	a3,a0,0x1
   1c8fc:	00a6b333          	sltu	t1,a3,a0
   1c900:	00b30333          	add	t1,t1,a1
   1c904:	00678333          	add	t1,a5,t1
   1c908:	00068513          	mv	a0,a3
   1c90c:	01131463          	bne	t1,a7,1c914 <__divdf3+0x4e8>
   1c910:	00a70463          	beq	a4,a0,1c918 <__divdf3+0x4ec>
   1c914:	001f6f13          	ori	t5,t5,1
   1c918:	3ff80713          	addi	a4,a6,1023
   1c91c:	10e05263          	blez	a4,1ca20 <__divdf3+0x5f4>
   1c920:	007f7793          	andi	a5,t5,7
   1c924:	02078063          	beqz	a5,1c944 <__divdf3+0x518>
   1c928:	00ff7793          	andi	a5,t5,15
   1c92c:	00400693          	li	a3,4
   1c930:	00d78a63          	beq	a5,a3,1c944 <__divdf3+0x518>
   1c934:	004f0793          	addi	a5,t5,4
   1c938:	01e7b6b3          	sltu	a3,a5,t5
   1c93c:	00d484b3          	add	s1,s1,a3
   1c940:	00078f13          	mv	t5,a5
   1c944:	00749793          	slli	a5,s1,0x7
   1c948:	0007da63          	bgez	a5,1c95c <__divdf3+0x530>
   1c94c:	ff0007b7          	lui	a5,0xff000
   1c950:	fff78793          	addi	a5,a5,-1 # feffffff <__BSS_END__+0xfefdd2cf>
   1c954:	00f4f4b3          	and	s1,s1,a5
   1c958:	40080713          	addi	a4,a6,1024
   1c95c:	7fe00793          	li	a5,2046
   1c960:	08e7ca63          	blt	a5,a4,1c9f4 <__divdf3+0x5c8>
   1c964:	003f5f13          	srli	t5,t5,0x3
   1c968:	01d49793          	slli	a5,s1,0x1d
   1c96c:	01e7ef33          	or	t5,a5,t5
   1c970:	0034d513          	srli	a0,s1,0x3
   1c974:	00c51513          	slli	a0,a0,0xc
   1c978:	02c12083          	lw	ra,44(sp)
   1c97c:	02812403          	lw	s0,40(sp)
   1c980:	00c55513          	srli	a0,a0,0xc
   1c984:	01471713          	slli	a4,a4,0x14
   1c988:	00a76733          	or	a4,a4,a0
   1c98c:	01f61613          	slli	a2,a2,0x1f
   1c990:	00c767b3          	or	a5,a4,a2
   1c994:	02412483          	lw	s1,36(sp)
   1c998:	02012903          	lw	s2,32(sp)
   1c99c:	01c12983          	lw	s3,28(sp)
   1c9a0:	01812a03          	lw	s4,24(sp)
   1c9a4:	01412a83          	lw	s5,20(sp)
   1c9a8:	01012b03          	lw	s6,16(sp)
   1c9ac:	00c12b83          	lw	s7,12(sp)
   1c9b0:	000f0513          	mv	a0,t5
   1c9b4:	00078593          	mv	a1,a5
   1c9b8:	03010113          	addi	sp,sp,48
   1c9bc:	00008067          	ret
   1c9c0:	fff80813          	addi	a6,a6,-1
   1c9c4:	00000713          	li	a4,0
   1c9c8:	cbdff06f          	j	1c684 <__divdf3+0x258>
   1c9cc:	00098613          	mv	a2,s3
   1c9d0:	000a0493          	mv	s1,s4
   1c9d4:	00040f13          	mv	t5,s0
   1c9d8:	000b0793          	mv	a5,s6
   1c9dc:	00300713          	li	a4,3
   1c9e0:	0ee78863          	beq	a5,a4,1cad0 <__divdf3+0x6a4>
   1c9e4:	00100713          	li	a4,1
   1c9e8:	0ee78e63          	beq	a5,a4,1cae4 <__divdf3+0x6b8>
   1c9ec:	00200713          	li	a4,2
   1c9f0:	f2e794e3          	bne	a5,a4,1c918 <__divdf3+0x4ec>
   1c9f4:	00000513          	li	a0,0
   1c9f8:	00000f13          	li	t5,0
   1c9fc:	7ff00713          	li	a4,2047
   1ca00:	f75ff06f          	j	1c974 <__divdf3+0x548>
   1ca04:	00090613          	mv	a2,s2
   1ca08:	fd5ff06f          	j	1c9dc <__divdf3+0x5b0>
   1ca0c:	000804b7          	lui	s1,0x80
   1ca10:	00000f13          	li	t5,0
   1ca14:	00000613          	li	a2,0
   1ca18:	00300793          	li	a5,3
   1ca1c:	fc1ff06f          	j	1c9dc <__divdf3+0x5b0>
   1ca20:	00100513          	li	a0,1
   1ca24:	40e50533          	sub	a0,a0,a4
   1ca28:	03800793          	li	a5,56
   1ca2c:	0aa7cc63          	blt	a5,a0,1cae4 <__divdf3+0x6b8>
   1ca30:	01f00793          	li	a5,31
   1ca34:	06a7c463          	blt	a5,a0,1ca9c <__divdf3+0x670>
   1ca38:	41e80813          	addi	a6,a6,1054
   1ca3c:	010497b3          	sll	a5,s1,a6
   1ca40:	00af5733          	srl	a4,t5,a0
   1ca44:	010f1833          	sll	a6,t5,a6
   1ca48:	00e7e7b3          	or	a5,a5,a4
   1ca4c:	01003833          	snez	a6,a6
   1ca50:	0107e7b3          	or	a5,a5,a6
   1ca54:	00a4d533          	srl	a0,s1,a0
   1ca58:	0077f713          	andi	a4,a5,7
   1ca5c:	02070063          	beqz	a4,1ca7c <__divdf3+0x650>
   1ca60:	00f7f713          	andi	a4,a5,15
   1ca64:	00400693          	li	a3,4
   1ca68:	00d70a63          	beq	a4,a3,1ca7c <__divdf3+0x650>
   1ca6c:	00478713          	addi	a4,a5,4
   1ca70:	00f736b3          	sltu	a3,a4,a5
   1ca74:	00d50533          	add	a0,a0,a3
   1ca78:	00070793          	mv	a5,a4
   1ca7c:	00851713          	slli	a4,a0,0x8
   1ca80:	06074863          	bltz	a4,1caf0 <__divdf3+0x6c4>
   1ca84:	01d51f13          	slli	t5,a0,0x1d
   1ca88:	0037d793          	srli	a5,a5,0x3
   1ca8c:	00ff6f33          	or	t5,t5,a5
   1ca90:	00355513          	srli	a0,a0,0x3
   1ca94:	00000713          	li	a4,0
   1ca98:	eddff06f          	j	1c974 <__divdf3+0x548>
   1ca9c:	fe100793          	li	a5,-31
   1caa0:	40e787b3          	sub	a5,a5,a4
   1caa4:	02000693          	li	a3,32
   1caa8:	00f4d7b3          	srl	a5,s1,a5
   1caac:	00000713          	li	a4,0
   1cab0:	00d50663          	beq	a0,a3,1cabc <__divdf3+0x690>
   1cab4:	43e80713          	addi	a4,a6,1086
   1cab8:	00e49733          	sll	a4,s1,a4
   1cabc:	01e76733          	or	a4,a4,t5
   1cac0:	00e03733          	snez	a4,a4
   1cac4:	00e7e7b3          	or	a5,a5,a4
   1cac8:	00000513          	li	a0,0
   1cacc:	f8dff06f          	j	1ca58 <__divdf3+0x62c>
   1cad0:	00080537          	lui	a0,0x80
   1cad4:	00000f13          	li	t5,0
   1cad8:	7ff00713          	li	a4,2047
   1cadc:	00000613          	li	a2,0
   1cae0:	e95ff06f          	j	1c974 <__divdf3+0x548>
   1cae4:	00000513          	li	a0,0
   1cae8:	00000f13          	li	t5,0
   1caec:	fa9ff06f          	j	1ca94 <__divdf3+0x668>
   1caf0:	00000513          	li	a0,0
   1caf4:	00000f13          	li	t5,0
   1caf8:	00100713          	li	a4,1
   1cafc:	e79ff06f          	j	1c974 <__divdf3+0x548>
   1cb00:	000f0693          	mv	a3,t5
   1cb04:	00068f13          	mv	t5,a3
   1cb08:	e0dff06f          	j	1c914 <__divdf3+0x4e8>

0001cb0c <__eqdf2>:
   1cb0c:	0145d713          	srli	a4,a1,0x14
   1cb10:	001007b7          	lui	a5,0x100
   1cb14:	fff78793          	addi	a5,a5,-1 # fffff <__BSS_END__+0xdd2cf>
   1cb18:	0146d813          	srli	a6,a3,0x14
   1cb1c:	00050313          	mv	t1,a0
   1cb20:	00050e93          	mv	t4,a0
   1cb24:	7ff77713          	andi	a4,a4,2047
   1cb28:	7ff00513          	li	a0,2047
   1cb2c:	00b7f8b3          	and	a7,a5,a1
   1cb30:	00060f13          	mv	t5,a2
   1cb34:	00d7f7b3          	and	a5,a5,a3
   1cb38:	01f5d593          	srli	a1,a1,0x1f
   1cb3c:	7ff87813          	andi	a6,a6,2047
   1cb40:	01f6d693          	srli	a3,a3,0x1f
   1cb44:	00a71c63          	bne	a4,a0,1cb5c <__eqdf2+0x50>
   1cb48:	0068ee33          	or	t3,a7,t1
   1cb4c:	00100513          	li	a0,1
   1cb50:	000e1463          	bnez	t3,1cb58 <__eqdf2+0x4c>
   1cb54:	00e80663          	beq	a6,a4,1cb60 <__eqdf2+0x54>
   1cb58:	00008067          	ret
   1cb5c:	00a81863          	bne	a6,a0,1cb6c <__eqdf2+0x60>
   1cb60:	00c7e633          	or	a2,a5,a2
   1cb64:	00100513          	li	a0,1
   1cb68:	fe0618e3          	bnez	a2,1cb58 <__eqdf2+0x4c>
   1cb6c:	00100513          	li	a0,1
   1cb70:	ff0714e3          	bne	a4,a6,1cb58 <__eqdf2+0x4c>
   1cb74:	fef892e3          	bne	a7,a5,1cb58 <__eqdf2+0x4c>
   1cb78:	ffee90e3          	bne	t4,t5,1cb58 <__eqdf2+0x4c>
   1cb7c:	00d58a63          	beq	a1,a3,1cb90 <__eqdf2+0x84>
   1cb80:	fc071ce3          	bnez	a4,1cb58 <__eqdf2+0x4c>
   1cb84:	0068e8b3          	or	a7,a7,t1
   1cb88:	01103533          	snez	a0,a7
   1cb8c:	00008067          	ret
   1cb90:	00000513          	li	a0,0
   1cb94:	00008067          	ret

0001cb98 <__gedf2>:
   1cb98:	0146d793          	srli	a5,a3,0x14
   1cb9c:	0145d893          	srli	a7,a1,0x14
   1cba0:	00100737          	lui	a4,0x100
   1cba4:	fff70713          	addi	a4,a4,-1 # fffff <__BSS_END__+0xdd2cf>
   1cba8:	00050813          	mv	a6,a0
   1cbac:	00050e13          	mv	t3,a0
   1cbb0:	7ff8f893          	andi	a7,a7,2047
   1cbb4:	7ff7f513          	andi	a0,a5,2047
   1cbb8:	7ff00793          	li	a5,2047
   1cbbc:	00b77333          	and	t1,a4,a1
   1cbc0:	00060e93          	mv	t4,a2
   1cbc4:	00d77733          	and	a4,a4,a3
   1cbc8:	01f5d593          	srli	a1,a1,0x1f
   1cbcc:	01f6d693          	srli	a3,a3,0x1f
   1cbd0:	00f89663          	bne	a7,a5,1cbdc <__gedf2+0x44>
   1cbd4:	010367b3          	or	a5,t1,a6
   1cbd8:	06079c63          	bnez	a5,1cc50 <__gedf2+0xb8>
   1cbdc:	7ff00793          	li	a5,2047
   1cbe0:	00f51663          	bne	a0,a5,1cbec <__gedf2+0x54>
   1cbe4:	00c767b3          	or	a5,a4,a2
   1cbe8:	06079463          	bnez	a5,1cc50 <__gedf2+0xb8>
   1cbec:	00000793          	li	a5,0
   1cbf0:	00089663          	bnez	a7,1cbfc <__gedf2+0x64>
   1cbf4:	01036833          	or	a6,t1,a6
   1cbf8:	00183793          	seqz	a5,a6
   1cbfc:	04051e63          	bnez	a0,1cc58 <__gedf2+0xc0>
   1cc00:	00c76633          	or	a2,a4,a2
   1cc04:	00078c63          	beqz	a5,1cc1c <__gedf2+0x84>
   1cc08:	02060063          	beqz	a2,1cc28 <__gedf2+0x90>
   1cc0c:	00100513          	li	a0,1
   1cc10:	00069c63          	bnez	a3,1cc28 <__gedf2+0x90>
   1cc14:	fff00513          	li	a0,-1
   1cc18:	00008067          	ret
   1cc1c:	04061063          	bnez	a2,1cc5c <__gedf2+0xc4>
   1cc20:	fff00513          	li	a0,-1
   1cc24:	04058463          	beqz	a1,1cc6c <__gedf2+0xd4>
   1cc28:	00008067          	ret
   1cc2c:	fea8c0e3          	blt	a7,a0,1cc0c <__gedf2+0x74>
   1cc30:	fe6768e3          	bltu	a4,t1,1cc20 <__gedf2+0x88>
   1cc34:	00e31863          	bne	t1,a4,1cc44 <__gedf2+0xac>
   1cc38:	ffcee4e3          	bltu	t4,t3,1cc20 <__gedf2+0x88>
   1cc3c:	00000513          	li	a0,0
   1cc40:	ffde74e3          	bgeu	t3,t4,1cc28 <__gedf2+0x90>
   1cc44:	00100513          	li	a0,1
   1cc48:	fe0590e3          	bnez	a1,1cc28 <__gedf2+0x90>
   1cc4c:	fc9ff06f          	j	1cc14 <__gedf2+0x7c>
   1cc50:	ffe00513          	li	a0,-2
   1cc54:	00008067          	ret
   1cc58:	fa079ae3          	bnez	a5,1cc0c <__gedf2+0x74>
   1cc5c:	fcb692e3          	bne	a3,a1,1cc20 <__gedf2+0x88>
   1cc60:	fd1556e3          	bge	a0,a7,1cc2c <__gedf2+0x94>
   1cc64:	fff00513          	li	a0,-1
   1cc68:	fc0690e3          	bnez	a3,1cc28 <__gedf2+0x90>
   1cc6c:	00100513          	li	a0,1
   1cc70:	00008067          	ret

0001cc74 <__ledf2>:
   1cc74:	0146d793          	srli	a5,a3,0x14
   1cc78:	0145d893          	srli	a7,a1,0x14
   1cc7c:	00100737          	lui	a4,0x100
   1cc80:	fff70713          	addi	a4,a4,-1 # fffff <__BSS_END__+0xdd2cf>
   1cc84:	00050813          	mv	a6,a0
   1cc88:	00050e13          	mv	t3,a0
   1cc8c:	7ff8f893          	andi	a7,a7,2047
   1cc90:	7ff7f513          	andi	a0,a5,2047
   1cc94:	7ff00793          	li	a5,2047
   1cc98:	00b77333          	and	t1,a4,a1
   1cc9c:	00060e93          	mv	t4,a2
   1cca0:	00d77733          	and	a4,a4,a3
   1cca4:	01f5d593          	srli	a1,a1,0x1f
   1cca8:	01f6d693          	srli	a3,a3,0x1f
   1ccac:	00f89663          	bne	a7,a5,1ccb8 <__ledf2+0x44>
   1ccb0:	010367b3          	or	a5,t1,a6
   1ccb4:	06079c63          	bnez	a5,1cd2c <__ledf2+0xb8>
   1ccb8:	7ff00793          	li	a5,2047
   1ccbc:	00f51663          	bne	a0,a5,1ccc8 <__ledf2+0x54>
   1ccc0:	00c767b3          	or	a5,a4,a2
   1ccc4:	06079463          	bnez	a5,1cd2c <__ledf2+0xb8>
   1ccc8:	00000793          	li	a5,0
   1cccc:	00089663          	bnez	a7,1ccd8 <__ledf2+0x64>
   1ccd0:	01036833          	or	a6,t1,a6
   1ccd4:	00183793          	seqz	a5,a6
   1ccd8:	04051e63          	bnez	a0,1cd34 <__ledf2+0xc0>
   1ccdc:	00c76633          	or	a2,a4,a2
   1cce0:	00078c63          	beqz	a5,1ccf8 <__ledf2+0x84>
   1cce4:	02060063          	beqz	a2,1cd04 <__ledf2+0x90>
   1cce8:	00100513          	li	a0,1
   1ccec:	00069c63          	bnez	a3,1cd04 <__ledf2+0x90>
   1ccf0:	fff00513          	li	a0,-1
   1ccf4:	00008067          	ret
   1ccf8:	04061063          	bnez	a2,1cd38 <__ledf2+0xc4>
   1ccfc:	fff00513          	li	a0,-1
   1cd00:	04058463          	beqz	a1,1cd48 <__ledf2+0xd4>
   1cd04:	00008067          	ret
   1cd08:	fea8c0e3          	blt	a7,a0,1cce8 <__ledf2+0x74>
   1cd0c:	fe6768e3          	bltu	a4,t1,1ccfc <__ledf2+0x88>
   1cd10:	00e31863          	bne	t1,a4,1cd20 <__ledf2+0xac>
   1cd14:	ffcee4e3          	bltu	t4,t3,1ccfc <__ledf2+0x88>
   1cd18:	00000513          	li	a0,0
   1cd1c:	ffde74e3          	bgeu	t3,t4,1cd04 <__ledf2+0x90>
   1cd20:	00100513          	li	a0,1
   1cd24:	fe0590e3          	bnez	a1,1cd04 <__ledf2+0x90>
   1cd28:	fc9ff06f          	j	1ccf0 <__ledf2+0x7c>
   1cd2c:	00200513          	li	a0,2
   1cd30:	00008067          	ret
   1cd34:	fa079ae3          	bnez	a5,1cce8 <__ledf2+0x74>
   1cd38:	fcb692e3          	bne	a3,a1,1ccfc <__ledf2+0x88>
   1cd3c:	fd1556e3          	bge	a0,a7,1cd08 <__ledf2+0x94>
   1cd40:	fff00513          	li	a0,-1
   1cd44:	fc0690e3          	bnez	a3,1cd04 <__ledf2+0x90>
   1cd48:	00100513          	li	a0,1
   1cd4c:	00008067          	ret

0001cd50 <__muldf3>:
   1cd50:	fd010113          	addi	sp,sp,-48
   1cd54:	01512a23          	sw	s5,20(sp)
   1cd58:	0145da93          	srli	s5,a1,0x14
   1cd5c:	02812423          	sw	s0,40(sp)
   1cd60:	02912223          	sw	s1,36(sp)
   1cd64:	01312e23          	sw	s3,28(sp)
   1cd68:	01412c23          	sw	s4,24(sp)
   1cd6c:	01612823          	sw	s6,16(sp)
   1cd70:	00c59493          	slli	s1,a1,0xc
   1cd74:	02112623          	sw	ra,44(sp)
   1cd78:	03212023          	sw	s2,32(sp)
   1cd7c:	01712623          	sw	s7,12(sp)
   1cd80:	7ffafa93          	andi	s5,s5,2047
   1cd84:	00050413          	mv	s0,a0
   1cd88:	00060b13          	mv	s6,a2
   1cd8c:	00068993          	mv	s3,a3
   1cd90:	00c4d493          	srli	s1,s1,0xc
   1cd94:	01f5da13          	srli	s4,a1,0x1f
   1cd98:	240a8e63          	beqz	s5,1cff4 <__muldf3+0x2a4>
   1cd9c:	7ff00793          	li	a5,2047
   1cda0:	2cfa8063          	beq	s5,a5,1d060 <__muldf3+0x310>
   1cda4:	00349493          	slli	s1,s1,0x3
   1cda8:	01d55793          	srli	a5,a0,0x1d
   1cdac:	0097e7b3          	or	a5,a5,s1
   1cdb0:	008004b7          	lui	s1,0x800
   1cdb4:	0097e4b3          	or	s1,a5,s1
   1cdb8:	00351913          	slli	s2,a0,0x3
   1cdbc:	c01a8a93          	addi	s5,s5,-1023
   1cdc0:	00000b93          	li	s7,0
   1cdc4:	0149d713          	srli	a4,s3,0x14
   1cdc8:	00c99413          	slli	s0,s3,0xc
   1cdcc:	7ff77713          	andi	a4,a4,2047
   1cdd0:	00c45413          	srli	s0,s0,0xc
   1cdd4:	01f9d993          	srli	s3,s3,0x1f
   1cdd8:	2c070063          	beqz	a4,1d098 <__muldf3+0x348>
   1cddc:	7ff00793          	li	a5,2047
   1cde0:	32f70463          	beq	a4,a5,1d108 <__muldf3+0x3b8>
   1cde4:	00341413          	slli	s0,s0,0x3
   1cde8:	01db5793          	srli	a5,s6,0x1d
   1cdec:	0087e7b3          	or	a5,a5,s0
   1cdf0:	00800437          	lui	s0,0x800
   1cdf4:	0087e433          	or	s0,a5,s0
   1cdf8:	c0170693          	addi	a3,a4,-1023
   1cdfc:	003b1793          	slli	a5,s6,0x3
   1ce00:	00000713          	li	a4,0
   1ce04:	00da8ab3          	add	s5,s5,a3
   1ce08:	002b9693          	slli	a3,s7,0x2
   1ce0c:	00e6e6b3          	or	a3,a3,a4
   1ce10:	00a00613          	li	a2,10
   1ce14:	001a8513          	addi	a0,s5,1
   1ce18:	40d64663          	blt	a2,a3,1d224 <__muldf3+0x4d4>
   1ce1c:	00200613          	li	a2,2
   1ce20:	013a45b3          	xor	a1,s4,s3
   1ce24:	30d64e63          	blt	a2,a3,1d140 <__muldf3+0x3f0>
   1ce28:	fff68693          	addi	a3,a3,-1
   1ce2c:	00100613          	li	a2,1
   1ce30:	32d67a63          	bgeu	a2,a3,1d164 <__muldf3+0x414>
   1ce34:	00010337          	lui	t1,0x10
   1ce38:	fff30e13          	addi	t3,t1,-1 # ffff <exit-0xb5>
   1ce3c:	01095713          	srli	a4,s2,0x10
   1ce40:	0107d893          	srli	a7,a5,0x10
   1ce44:	01c97933          	and	s2,s2,t3
   1ce48:	01c7ff33          	and	t5,a5,t3
   1ce4c:	03e907b3          	mul	a5,s2,t5
   1ce50:	03e70eb3          	mul	t4,a4,t5
   1ce54:	0107d813          	srli	a6,a5,0x10
   1ce58:	03288633          	mul	a2,a7,s2
   1ce5c:	01d60633          	add	a2,a2,t4
   1ce60:	00c80833          	add	a6,a6,a2
   1ce64:	031706b3          	mul	a3,a4,a7
   1ce68:	01d87463          	bgeu	a6,t4,1ce70 <__muldf3+0x120>
   1ce6c:	006686b3          	add	a3,a3,t1
   1ce70:	01085293          	srli	t0,a6,0x10
   1ce74:	01c87833          	and	a6,a6,t3
   1ce78:	01c7f7b3          	and	a5,a5,t3
   1ce7c:	01045613          	srli	a2,s0,0x10
   1ce80:	01c47e33          	and	t3,s0,t3
   1ce84:	01081813          	slli	a6,a6,0x10
   1ce88:	00f80833          	add	a6,a6,a5
   1ce8c:	03c90eb3          	mul	t4,s2,t3
   1ce90:	03c707b3          	mul	a5,a4,t3
   1ce94:	03260933          	mul	s2,a2,s2
   1ce98:	02c70333          	mul	t1,a4,a2
   1ce9c:	00f90933          	add	s2,s2,a5
   1cea0:	010ed713          	srli	a4,t4,0x10
   1cea4:	01270733          	add	a4,a4,s2
   1cea8:	00f77663          	bgeu	a4,a5,1ceb4 <__muldf3+0x164>
   1ceac:	000107b7          	lui	a5,0x10
   1ceb0:	00f30333          	add	t1,t1,a5
   1ceb4:	00010437          	lui	s0,0x10
   1ceb8:	01075793          	srli	a5,a4,0x10
   1cebc:	fff40f93          	addi	t6,s0,-1 # ffff <exit-0xb5>
   1cec0:	00678333          	add	t1,a5,t1
   1cec4:	01f777b3          	and	a5,a4,t6
   1cec8:	01fefeb3          	and	t4,t4,t6
   1cecc:	01079793          	slli	a5,a5,0x10
   1ced0:	01f4ffb3          	and	t6,s1,t6
   1ced4:	01d787b3          	add	a5,a5,t4
   1ced8:	0104de93          	srli	t4,s1,0x10
   1cedc:	03ff03b3          	mul	t2,t5,t6
   1cee0:	00f282b3          	add	t0,t0,a5
   1cee4:	03ee8f33          	mul	t5,t4,t5
   1cee8:	0103d713          	srli	a4,t2,0x10
   1ceec:	03d884b3          	mul	s1,a7,t4
   1cef0:	03f888b3          	mul	a7,a7,t6
   1cef4:	01e888b3          	add	a7,a7,t5
   1cef8:	01170733          	add	a4,a4,a7
   1cefc:	01e77463          	bgeu	a4,t5,1cf04 <__muldf3+0x1b4>
   1cf00:	008484b3          	add	s1,s1,s0
   1cf04:	01075f13          	srli	t5,a4,0x10
   1cf08:	009f0f33          	add	t5,t5,s1
   1cf0c:	000104b7          	lui	s1,0x10
   1cf10:	fff48413          	addi	s0,s1,-1 # ffff <exit-0xb5>
   1cf14:	00877733          	and	a4,a4,s0
   1cf18:	0083f3b3          	and	t2,t2,s0
   1cf1c:	01071713          	slli	a4,a4,0x10
   1cf20:	007708b3          	add	a7,a4,t2
   1cf24:	03fe03b3          	mul	t2,t3,t6
   1cf28:	03ce8e33          	mul	t3,t4,t3
   1cf2c:	03d60eb3          	mul	t4,a2,t4
   1cf30:	03f60633          	mul	a2,a2,t6
   1cf34:	0103df93          	srli	t6,t2,0x10
   1cf38:	01c60633          	add	a2,a2,t3
   1cf3c:	00cf8fb3          	add	t6,t6,a2
   1cf40:	01cff463          	bgeu	t6,t3,1cf48 <__muldf3+0x1f8>
   1cf44:	009e8eb3          	add	t4,t4,s1
   1cf48:	008ff733          	and	a4,t6,s0
   1cf4c:	0083f3b3          	and	t2,t2,s0
   1cf50:	01071713          	slli	a4,a4,0x10
   1cf54:	005686b3          	add	a3,a3,t0
   1cf58:	00770733          	add	a4,a4,t2
   1cf5c:	00670333          	add	t1,a4,t1
   1cf60:	00f6b7b3          	sltu	a5,a3,a5
   1cf64:	00f307b3          	add	a5,t1,a5
   1cf68:	00e33633          	sltu	a2,t1,a4
   1cf6c:	011688b3          	add	a7,a3,a7
   1cf70:	0067b333          	sltu	t1,a5,t1
   1cf74:	00666633          	or	a2,a2,t1
   1cf78:	00d8b6b3          	sltu	a3,a7,a3
   1cf7c:	01e78333          	add	t1,a5,t5
   1cf80:	00d306b3          	add	a3,t1,a3
   1cf84:	00f33733          	sltu	a4,t1,a5
   1cf88:	010fdf93          	srli	t6,t6,0x10
   1cf8c:	0066b333          	sltu	t1,a3,t1
   1cf90:	00989793          	slli	a5,a7,0x9
   1cf94:	01f60633          	add	a2,a2,t6
   1cf98:	00676733          	or	a4,a4,t1
   1cf9c:	00c70733          	add	a4,a4,a2
   1cfa0:	0107e7b3          	or	a5,a5,a6
   1cfa4:	01d70733          	add	a4,a4,t4
   1cfa8:	00f037b3          	snez	a5,a5
   1cfac:	0178d893          	srli	a7,a7,0x17
   1cfb0:	00971713          	slli	a4,a4,0x9
   1cfb4:	0176d413          	srli	s0,a3,0x17
   1cfb8:	0117e7b3          	or	a5,a5,a7
   1cfbc:	00969693          	slli	a3,a3,0x9
   1cfc0:	00d7e7b3          	or	a5,a5,a3
   1cfc4:	00771693          	slli	a3,a4,0x7
   1cfc8:	00876433          	or	s0,a4,s0
   1cfcc:	0206d063          	bgez	a3,1cfec <__muldf3+0x29c>
   1cfd0:	0017d713          	srli	a4,a5,0x1
   1cfd4:	0017f793          	andi	a5,a5,1
   1cfd8:	00f76733          	or	a4,a4,a5
   1cfdc:	01f41793          	slli	a5,s0,0x1f
   1cfe0:	00f767b3          	or	a5,a4,a5
   1cfe4:	00145413          	srli	s0,s0,0x1
   1cfe8:	00050a93          	mv	s5,a0
   1cfec:	000a8513          	mv	a0,s5
   1cff0:	18c0006f          	j	1d17c <__muldf3+0x42c>
   1cff4:	00a4e933          	or	s2,s1,a0
   1cff8:	08090063          	beqz	s2,1d078 <__muldf3+0x328>
   1cffc:	04048063          	beqz	s1,1d03c <__muldf3+0x2ec>
   1d000:	00048513          	mv	a0,s1
   1d004:	0b1030ef          	jal	208b4 <__clzsi2>
   1d008:	ff550713          	addi	a4,a0,-11 # 7fff5 <__BSS_END__+0x5d2c5>
   1d00c:	01d00793          	li	a5,29
   1d010:	ff850693          	addi	a3,a0,-8
   1d014:	40e787b3          	sub	a5,a5,a4
   1d018:	00d494b3          	sll	s1,s1,a3
   1d01c:	00f457b3          	srl	a5,s0,a5
   1d020:	0097e7b3          	or	a5,a5,s1
   1d024:	00d414b3          	sll	s1,s0,a3
   1d028:	c0d00a93          	li	s5,-1011
   1d02c:	00048913          	mv	s2,s1
   1d030:	40aa8ab3          	sub	s5,s5,a0
   1d034:	00078493          	mv	s1,a5
   1d038:	d89ff06f          	j	1cdc0 <__muldf3+0x70>
   1d03c:	079030ef          	jal	208b4 <__clzsi2>
   1d040:	00050793          	mv	a5,a0
   1d044:	01578713          	addi	a4,a5,21 # 10015 <exit-0x9f>
   1d048:	01c00693          	li	a3,28
   1d04c:	02050513          	addi	a0,a0,32
   1d050:	fae6dee3          	bge	a3,a4,1d00c <__muldf3+0x2bc>
   1d054:	ff878793          	addi	a5,a5,-8
   1d058:	00f417b3          	sll	a5,s0,a5
   1d05c:	fcdff06f          	j	1d028 <__muldf3+0x2d8>
   1d060:	00a4e933          	or	s2,s1,a0
   1d064:	02091263          	bnez	s2,1d088 <__muldf3+0x338>
   1d068:	00000493          	li	s1,0
   1d06c:	7ff00a93          	li	s5,2047
   1d070:	00200b93          	li	s7,2
   1d074:	d51ff06f          	j	1cdc4 <__muldf3+0x74>
   1d078:	00000493          	li	s1,0
   1d07c:	00000a93          	li	s5,0
   1d080:	00100b93          	li	s7,1
   1d084:	d41ff06f          	j	1cdc4 <__muldf3+0x74>
   1d088:	00050913          	mv	s2,a0
   1d08c:	7ff00a93          	li	s5,2047
   1d090:	00300b93          	li	s7,3
   1d094:	d31ff06f          	j	1cdc4 <__muldf3+0x74>
   1d098:	016467b3          	or	a5,s0,s6
   1d09c:	08078263          	beqz	a5,1d120 <__muldf3+0x3d0>
   1d0a0:	04040063          	beqz	s0,1d0e0 <__muldf3+0x390>
   1d0a4:	00040513          	mv	a0,s0
   1d0a8:	00d030ef          	jal	208b4 <__clzsi2>
   1d0ac:	ff550693          	addi	a3,a0,-11
   1d0b0:	01d00713          	li	a4,29
   1d0b4:	ff850793          	addi	a5,a0,-8
   1d0b8:	40d70733          	sub	a4,a4,a3
   1d0bc:	00f41433          	sll	s0,s0,a5
   1d0c0:	00eb5733          	srl	a4,s6,a4
   1d0c4:	00876733          	or	a4,a4,s0
   1d0c8:	00fb1433          	sll	s0,s6,a5
   1d0cc:	c0d00693          	li	a3,-1011
   1d0d0:	00040793          	mv	a5,s0
   1d0d4:	40a686b3          	sub	a3,a3,a0
   1d0d8:	00070413          	mv	s0,a4
   1d0dc:	d25ff06f          	j	1ce00 <__muldf3+0xb0>
   1d0e0:	000b0513          	mv	a0,s6
   1d0e4:	7d0030ef          	jal	208b4 <__clzsi2>
   1d0e8:	00050793          	mv	a5,a0
   1d0ec:	01578693          	addi	a3,a5,21
   1d0f0:	01c00713          	li	a4,28
   1d0f4:	02050513          	addi	a0,a0,32
   1d0f8:	fad75ce3          	bge	a4,a3,1d0b0 <__muldf3+0x360>
   1d0fc:	ff878793          	addi	a5,a5,-8
   1d100:	00fb1733          	sll	a4,s6,a5
   1d104:	fc9ff06f          	j	1d0cc <__muldf3+0x37c>
   1d108:	016467b3          	or	a5,s0,s6
   1d10c:	02079263          	bnez	a5,1d130 <__muldf3+0x3e0>
   1d110:	00000413          	li	s0,0
   1d114:	7ff00693          	li	a3,2047
   1d118:	00200713          	li	a4,2
   1d11c:	ce9ff06f          	j	1ce04 <__muldf3+0xb4>
   1d120:	00000413          	li	s0,0
   1d124:	00000693          	li	a3,0
   1d128:	00100713          	li	a4,1
   1d12c:	cd9ff06f          	j	1ce04 <__muldf3+0xb4>
   1d130:	000b0793          	mv	a5,s6
   1d134:	7ff00693          	li	a3,2047
   1d138:	00300713          	li	a4,3
   1d13c:	cc9ff06f          	j	1ce04 <__muldf3+0xb4>
   1d140:	00100613          	li	a2,1
   1d144:	00d61633          	sll	a2,a2,a3
   1d148:	53067693          	andi	a3,a2,1328
   1d14c:	0e069663          	bnez	a3,1d238 <__muldf3+0x4e8>
   1d150:	24067813          	andi	a6,a2,576
   1d154:	1a081a63          	bnez	a6,1d308 <__muldf3+0x5b8>
   1d158:	08867613          	andi	a2,a2,136
   1d15c:	cc060ce3          	beqz	a2,1ce34 <__muldf3+0xe4>
   1d160:	00098593          	mv	a1,s3
   1d164:	00200693          	li	a3,2
   1d168:	18d70863          	beq	a4,a3,1d2f8 <__muldf3+0x5a8>
   1d16c:	00300693          	li	a3,3
   1d170:	1ad70463          	beq	a4,a3,1d318 <__muldf3+0x5c8>
   1d174:	00100693          	li	a3,1
   1d178:	1ad70663          	beq	a4,a3,1d324 <__muldf3+0x5d4>
   1d17c:	3ff50613          	addi	a2,a0,1023
   1d180:	0cc05463          	blez	a2,1d248 <__muldf3+0x4f8>
   1d184:	0077f713          	andi	a4,a5,7
   1d188:	02070063          	beqz	a4,1d1a8 <__muldf3+0x458>
   1d18c:	00f7f713          	andi	a4,a5,15
   1d190:	00400693          	li	a3,4
   1d194:	00d70a63          	beq	a4,a3,1d1a8 <__muldf3+0x458>
   1d198:	00478713          	addi	a4,a5,4
   1d19c:	00f736b3          	sltu	a3,a4,a5
   1d1a0:	00d40433          	add	s0,s0,a3
   1d1a4:	00070793          	mv	a5,a4
   1d1a8:	00741713          	slli	a4,s0,0x7
   1d1ac:	00075a63          	bgez	a4,1d1c0 <__muldf3+0x470>
   1d1b0:	ff000737          	lui	a4,0xff000
   1d1b4:	fff70713          	addi	a4,a4,-1 # feffffff <__BSS_END__+0xfefdd2cf>
   1d1b8:	00e47433          	and	s0,s0,a4
   1d1bc:	40050613          	addi	a2,a0,1024
   1d1c0:	7fe00713          	li	a4,2046
   1d1c4:	12c74a63          	blt	a4,a2,1d2f8 <__muldf3+0x5a8>
   1d1c8:	0037d793          	srli	a5,a5,0x3
   1d1cc:	01d41693          	slli	a3,s0,0x1d
   1d1d0:	00f6e6b3          	or	a3,a3,a5
   1d1d4:	00345713          	srli	a4,s0,0x3
   1d1d8:	00c71713          	slli	a4,a4,0xc
   1d1dc:	02c12083          	lw	ra,44(sp)
   1d1e0:	02812403          	lw	s0,40(sp)
   1d1e4:	01461613          	slli	a2,a2,0x14
   1d1e8:	00c75713          	srli	a4,a4,0xc
   1d1ec:	01f59593          	slli	a1,a1,0x1f
   1d1f0:	00e66633          	or	a2,a2,a4
   1d1f4:	00b667b3          	or	a5,a2,a1
   1d1f8:	02412483          	lw	s1,36(sp)
   1d1fc:	02012903          	lw	s2,32(sp)
   1d200:	01c12983          	lw	s3,28(sp)
   1d204:	01812a03          	lw	s4,24(sp)
   1d208:	01412a83          	lw	s5,20(sp)
   1d20c:	01012b03          	lw	s6,16(sp)
   1d210:	00c12b83          	lw	s7,12(sp)
   1d214:	00068513          	mv	a0,a3
   1d218:	00078593          	mv	a1,a5
   1d21c:	03010113          	addi	sp,sp,48
   1d220:	00008067          	ret
   1d224:	00f00613          	li	a2,15
   1d228:	0ec68863          	beq	a3,a2,1d318 <__muldf3+0x5c8>
   1d22c:	00b00613          	li	a2,11
   1d230:	000a0593          	mv	a1,s4
   1d234:	f2c686e3          	beq	a3,a2,1d160 <__muldf3+0x410>
   1d238:	00048413          	mv	s0,s1
   1d23c:	00090793          	mv	a5,s2
   1d240:	000b8713          	mv	a4,s7
   1d244:	f21ff06f          	j	1d164 <__muldf3+0x414>
   1d248:	00100713          	li	a4,1
   1d24c:	40c70733          	sub	a4,a4,a2
   1d250:	03800693          	li	a3,56
   1d254:	0ce6c863          	blt	a3,a4,1d324 <__muldf3+0x5d4>
   1d258:	01f00693          	li	a3,31
   1d25c:	06e6c463          	blt	a3,a4,1d2c4 <__muldf3+0x574>
   1d260:	41e50513          	addi	a0,a0,1054
   1d264:	00e7d633          	srl	a2,a5,a4
   1d268:	00a416b3          	sll	a3,s0,a0
   1d26c:	00a79533          	sll	a0,a5,a0
   1d270:	00c6e6b3          	or	a3,a3,a2
   1d274:	00a03533          	snez	a0,a0
   1d278:	00a6e7b3          	or	a5,a3,a0
   1d27c:	00e45733          	srl	a4,s0,a4
   1d280:	0077f693          	andi	a3,a5,7
   1d284:	02068063          	beqz	a3,1d2a4 <__muldf3+0x554>
   1d288:	00f7f693          	andi	a3,a5,15
   1d28c:	00400613          	li	a2,4
   1d290:	00c68a63          	beq	a3,a2,1d2a4 <__muldf3+0x554>
   1d294:	00478693          	addi	a3,a5,4
   1d298:	00f6b633          	sltu	a2,a3,a5
   1d29c:	00c70733          	add	a4,a4,a2
   1d2a0:	00068793          	mv	a5,a3
   1d2a4:	00871693          	slli	a3,a4,0x8
   1d2a8:	0806c463          	bltz	a3,1d330 <__muldf3+0x5e0>
   1d2ac:	01d71693          	slli	a3,a4,0x1d
   1d2b0:	0037d793          	srli	a5,a5,0x3
   1d2b4:	00f6e6b3          	or	a3,a3,a5
   1d2b8:	00375713          	srli	a4,a4,0x3
   1d2bc:	00000613          	li	a2,0
   1d2c0:	f19ff06f          	j	1d1d8 <__muldf3+0x488>
   1d2c4:	fe100693          	li	a3,-31
   1d2c8:	40c686b3          	sub	a3,a3,a2
   1d2cc:	02000813          	li	a6,32
   1d2d0:	00d456b3          	srl	a3,s0,a3
   1d2d4:	00000613          	li	a2,0
   1d2d8:	01070663          	beq	a4,a6,1d2e4 <__muldf3+0x594>
   1d2dc:	43e50613          	addi	a2,a0,1086
   1d2e0:	00c41633          	sll	a2,s0,a2
   1d2e4:	00f66633          	or	a2,a2,a5
   1d2e8:	00c03633          	snez	a2,a2
   1d2ec:	00c6e7b3          	or	a5,a3,a2
   1d2f0:	00000713          	li	a4,0
   1d2f4:	f8dff06f          	j	1d280 <__muldf3+0x530>
   1d2f8:	00000713          	li	a4,0
   1d2fc:	00000693          	li	a3,0
   1d300:	7ff00613          	li	a2,2047
   1d304:	ed5ff06f          	j	1d1d8 <__muldf3+0x488>
   1d308:	00080737          	lui	a4,0x80
   1d30c:	7ff00613          	li	a2,2047
   1d310:	00000593          	li	a1,0
   1d314:	ec5ff06f          	j	1d1d8 <__muldf3+0x488>
   1d318:	00080737          	lui	a4,0x80
   1d31c:	00000693          	li	a3,0
   1d320:	fedff06f          	j	1d30c <__muldf3+0x5bc>
   1d324:	00000713          	li	a4,0
   1d328:	00000693          	li	a3,0
   1d32c:	f91ff06f          	j	1d2bc <__muldf3+0x56c>
   1d330:	00000713          	li	a4,0
   1d334:	00000693          	li	a3,0
   1d338:	00100613          	li	a2,1
   1d33c:	e9dff06f          	j	1d1d8 <__muldf3+0x488>

0001d340 <__subdf3>:
   1d340:	00100837          	lui	a6,0x100
   1d344:	fff80813          	addi	a6,a6,-1 # fffff <__BSS_END__+0xdd2cf>
   1d348:	fe010113          	addi	sp,sp,-32
   1d34c:	00b878b3          	and	a7,a6,a1
   1d350:	0145d713          	srli	a4,a1,0x14
   1d354:	01d55793          	srli	a5,a0,0x1d
   1d358:	00d87833          	and	a6,a6,a3
   1d35c:	01212823          	sw	s2,16(sp)
   1d360:	7ff77913          	andi	s2,a4,2047
   1d364:	00389713          	slli	a4,a7,0x3
   1d368:	0146d893          	srli	a7,a3,0x14
   1d36c:	00912a23          	sw	s1,20(sp)
   1d370:	00e7e7b3          	or	a5,a5,a4
   1d374:	01f5d493          	srli	s1,a1,0x1f
   1d378:	01d65713          	srli	a4,a2,0x1d
   1d37c:	00381813          	slli	a6,a6,0x3
   1d380:	00112e23          	sw	ra,28(sp)
   1d384:	00812c23          	sw	s0,24(sp)
   1d388:	01312623          	sw	s3,12(sp)
   1d38c:	7ff8f893          	andi	a7,a7,2047
   1d390:	7ff00593          	li	a1,2047
   1d394:	00351513          	slli	a0,a0,0x3
   1d398:	01f6d693          	srli	a3,a3,0x1f
   1d39c:	01076733          	or	a4,a4,a6
   1d3a0:	00361613          	slli	a2,a2,0x3
   1d3a4:	00b89663          	bne	a7,a1,1d3b0 <__subdf3+0x70>
   1d3a8:	00c765b3          	or	a1,a4,a2
   1d3ac:	00059463          	bnez	a1,1d3b4 <__subdf3+0x74>
   1d3b0:	0016c693          	xori	a3,a3,1
   1d3b4:	41190833          	sub	a6,s2,a7
   1d3b8:	2a969a63          	bne	a3,s1,1d66c <__subdf3+0x32c>
   1d3bc:	11005c63          	blez	a6,1d4d4 <__subdf3+0x194>
   1d3c0:	04089063          	bnez	a7,1d400 <__subdf3+0xc0>
   1d3c4:	00c766b3          	or	a3,a4,a2
   1d3c8:	66068063          	beqz	a3,1da28 <__subdf3+0x6e8>
   1d3cc:	fff80593          	addi	a1,a6,-1
   1d3d0:	02059063          	bnez	a1,1d3f0 <__subdf3+0xb0>
   1d3d4:	00c50633          	add	a2,a0,a2
   1d3d8:	00a636b3          	sltu	a3,a2,a0
   1d3dc:	00e78733          	add	a4,a5,a4
   1d3e0:	00060513          	mv	a0,a2
   1d3e4:	00d707b3          	add	a5,a4,a3
   1d3e8:	00100913          	li	s2,1
   1d3ec:	06c0006f          	j	1d458 <__subdf3+0x118>
   1d3f0:	7ff00693          	li	a3,2047
   1d3f4:	02d81063          	bne	a6,a3,1d414 <__subdf3+0xd4>
   1d3f8:	7ff00913          	li	s2,2047
   1d3fc:	1f80006f          	j	1d5f4 <__subdf3+0x2b4>
   1d400:	7ff00693          	li	a3,2047
   1d404:	1ed90863          	beq	s2,a3,1d5f4 <__subdf3+0x2b4>
   1d408:	008006b7          	lui	a3,0x800
   1d40c:	00d76733          	or	a4,a4,a3
   1d410:	00080593          	mv	a1,a6
   1d414:	03800693          	li	a3,56
   1d418:	0ab6c863          	blt	a3,a1,1d4c8 <__subdf3+0x188>
   1d41c:	01f00693          	li	a3,31
   1d420:	06b6ca63          	blt	a3,a1,1d494 <__subdf3+0x154>
   1d424:	02000813          	li	a6,32
   1d428:	40b80833          	sub	a6,a6,a1
   1d42c:	010716b3          	sll	a3,a4,a6
   1d430:	00b658b3          	srl	a7,a2,a1
   1d434:	01061833          	sll	a6,a2,a6
   1d438:	0116e6b3          	or	a3,a3,a7
   1d43c:	01003833          	snez	a6,a6
   1d440:	0106e6b3          	or	a3,a3,a6
   1d444:	00b755b3          	srl	a1,a4,a1
   1d448:	00a68533          	add	a0,a3,a0
   1d44c:	00f585b3          	add	a1,a1,a5
   1d450:	00d536b3          	sltu	a3,a0,a3
   1d454:	00d587b3          	add	a5,a1,a3
   1d458:	00879713          	slli	a4,a5,0x8
   1d45c:	18075c63          	bgez	a4,1d5f4 <__subdf3+0x2b4>
   1d460:	00190913          	addi	s2,s2,1
   1d464:	7ff00713          	li	a4,2047
   1d468:	5ae90a63          	beq	s2,a4,1da1c <__subdf3+0x6dc>
   1d46c:	ff800737          	lui	a4,0xff800
   1d470:	fff70713          	addi	a4,a4,-1 # ff7fffff <__BSS_END__+0xff7dd2cf>
   1d474:	00e7f733          	and	a4,a5,a4
   1d478:	00155793          	srli	a5,a0,0x1
   1d47c:	00157513          	andi	a0,a0,1
   1d480:	00a7e7b3          	or	a5,a5,a0
   1d484:	01f71513          	slli	a0,a4,0x1f
   1d488:	00f56533          	or	a0,a0,a5
   1d48c:	00175793          	srli	a5,a4,0x1
   1d490:	1640006f          	j	1d5f4 <__subdf3+0x2b4>
   1d494:	fe058693          	addi	a3,a1,-32
   1d498:	02000893          	li	a7,32
   1d49c:	00d756b3          	srl	a3,a4,a3
   1d4a0:	00000813          	li	a6,0
   1d4a4:	01158863          	beq	a1,a7,1d4b4 <__subdf3+0x174>
   1d4a8:	04000813          	li	a6,64
   1d4ac:	40b80833          	sub	a6,a6,a1
   1d4b0:	01071833          	sll	a6,a4,a6
   1d4b4:	00c86833          	or	a6,a6,a2
   1d4b8:	01003833          	snez	a6,a6
   1d4bc:	0106e6b3          	or	a3,a3,a6
   1d4c0:	00000593          	li	a1,0
   1d4c4:	f85ff06f          	j	1d448 <__subdf3+0x108>
   1d4c8:	00c766b3          	or	a3,a4,a2
   1d4cc:	00d036b3          	snez	a3,a3
   1d4d0:	ff1ff06f          	j	1d4c0 <__subdf3+0x180>
   1d4d4:	0c080a63          	beqz	a6,1d5a8 <__subdf3+0x268>
   1d4d8:	412886b3          	sub	a3,a7,s2
   1d4dc:	02091463          	bnez	s2,1d504 <__subdf3+0x1c4>
   1d4e0:	00a7e5b3          	or	a1,a5,a0
   1d4e4:	50058e63          	beqz	a1,1da00 <__subdf3+0x6c0>
   1d4e8:	fff68593          	addi	a1,a3,-1 # 7fffff <__BSS_END__+0x7dd2cf>
   1d4ec:	ee0584e3          	beqz	a1,1d3d4 <__subdf3+0x94>
   1d4f0:	7ff00813          	li	a6,2047
   1d4f4:	03069263          	bne	a3,a6,1d518 <__subdf3+0x1d8>
   1d4f8:	00070793          	mv	a5,a4
   1d4fc:	00060513          	mv	a0,a2
   1d500:	ef9ff06f          	j	1d3f8 <__subdf3+0xb8>
   1d504:	7ff00593          	li	a1,2047
   1d508:	feb888e3          	beq	a7,a1,1d4f8 <__subdf3+0x1b8>
   1d50c:	008005b7          	lui	a1,0x800
   1d510:	00b7e7b3          	or	a5,a5,a1
   1d514:	00068593          	mv	a1,a3
   1d518:	03800693          	li	a3,56
   1d51c:	08b6c063          	blt	a3,a1,1d59c <__subdf3+0x25c>
   1d520:	01f00693          	li	a3,31
   1d524:	04b6c263          	blt	a3,a1,1d568 <__subdf3+0x228>
   1d528:	02000813          	li	a6,32
   1d52c:	40b80833          	sub	a6,a6,a1
   1d530:	010796b3          	sll	a3,a5,a6
   1d534:	00b55333          	srl	t1,a0,a1
   1d538:	01051833          	sll	a6,a0,a6
   1d53c:	0066e6b3          	or	a3,a3,t1
   1d540:	01003833          	snez	a6,a6
   1d544:	0106e6b3          	or	a3,a3,a6
   1d548:	00b7d5b3          	srl	a1,a5,a1
   1d54c:	00c68633          	add	a2,a3,a2
   1d550:	00e585b3          	add	a1,a1,a4
   1d554:	00d636b3          	sltu	a3,a2,a3
   1d558:	00060513          	mv	a0,a2
   1d55c:	00d587b3          	add	a5,a1,a3
   1d560:	00088913          	mv	s2,a7
   1d564:	ef5ff06f          	j	1d458 <__subdf3+0x118>
   1d568:	fe058693          	addi	a3,a1,-32 # 7fffe0 <__BSS_END__+0x7dd2b0>
   1d56c:	02000313          	li	t1,32
   1d570:	00d7d6b3          	srl	a3,a5,a3
   1d574:	00000813          	li	a6,0
   1d578:	00658863          	beq	a1,t1,1d588 <__subdf3+0x248>
   1d57c:	04000813          	li	a6,64
   1d580:	40b80833          	sub	a6,a6,a1
   1d584:	01079833          	sll	a6,a5,a6
   1d588:	00a86833          	or	a6,a6,a0
   1d58c:	01003833          	snez	a6,a6
   1d590:	0106e6b3          	or	a3,a3,a6
   1d594:	00000593          	li	a1,0
   1d598:	fb5ff06f          	j	1d54c <__subdf3+0x20c>
   1d59c:	00a7e6b3          	or	a3,a5,a0
   1d5a0:	00d036b3          	snez	a3,a3
   1d5a4:	ff1ff06f          	j	1d594 <__subdf3+0x254>
   1d5a8:	00190693          	addi	a3,s2,1
   1d5ac:	7fe6f593          	andi	a1,a3,2046
   1d5b0:	08059663          	bnez	a1,1d63c <__subdf3+0x2fc>
   1d5b4:	00a7e6b3          	or	a3,a5,a0
   1d5b8:	06091263          	bnez	s2,1d61c <__subdf3+0x2dc>
   1d5bc:	44068863          	beqz	a3,1da0c <__subdf3+0x6cc>
   1d5c0:	00c766b3          	or	a3,a4,a2
   1d5c4:	02068863          	beqz	a3,1d5f4 <__subdf3+0x2b4>
   1d5c8:	00c50633          	add	a2,a0,a2
   1d5cc:	00a636b3          	sltu	a3,a2,a0
   1d5d0:	00e78733          	add	a4,a5,a4
   1d5d4:	00d707b3          	add	a5,a4,a3
   1d5d8:	00879713          	slli	a4,a5,0x8
   1d5dc:	00060513          	mv	a0,a2
   1d5e0:	00075a63          	bgez	a4,1d5f4 <__subdf3+0x2b4>
   1d5e4:	ff800737          	lui	a4,0xff800
   1d5e8:	fff70713          	addi	a4,a4,-1 # ff7fffff <__BSS_END__+0xff7dd2cf>
   1d5ec:	00e7f7b3          	and	a5,a5,a4
   1d5f0:	00100913          	li	s2,1
   1d5f4:	00757713          	andi	a4,a0,7
   1d5f8:	44070863          	beqz	a4,1da48 <__subdf3+0x708>
   1d5fc:	00f57713          	andi	a4,a0,15
   1d600:	00400693          	li	a3,4
   1d604:	44d70263          	beq	a4,a3,1da48 <__subdf3+0x708>
   1d608:	00450713          	addi	a4,a0,4
   1d60c:	00a736b3          	sltu	a3,a4,a0
   1d610:	00d787b3          	add	a5,a5,a3
   1d614:	00070513          	mv	a0,a4
   1d618:	4300006f          	j	1da48 <__subdf3+0x708>
   1d61c:	ec068ee3          	beqz	a3,1d4f8 <__subdf3+0x1b8>
   1d620:	00c76633          	or	a2,a4,a2
   1d624:	dc060ae3          	beqz	a2,1d3f8 <__subdf3+0xb8>
   1d628:	00000493          	li	s1,0
   1d62c:	004007b7          	lui	a5,0x400
   1d630:	00000513          	li	a0,0
   1d634:	7ff00913          	li	s2,2047
   1d638:	4100006f          	j	1da48 <__subdf3+0x708>
   1d63c:	7ff00593          	li	a1,2047
   1d640:	3cb68c63          	beq	a3,a1,1da18 <__subdf3+0x6d8>
   1d644:	00c50633          	add	a2,a0,a2
   1d648:	00a63533          	sltu	a0,a2,a0
   1d64c:	00e78733          	add	a4,a5,a4
   1d650:	00a70733          	add	a4,a4,a0
   1d654:	01f71513          	slli	a0,a4,0x1f
   1d658:	00165613          	srli	a2,a2,0x1
   1d65c:	00c56533          	or	a0,a0,a2
   1d660:	00175793          	srli	a5,a4,0x1
   1d664:	00068913          	mv	s2,a3
   1d668:	f8dff06f          	j	1d5f4 <__subdf3+0x2b4>
   1d66c:	0f005c63          	blez	a6,1d764 <__subdf3+0x424>
   1d670:	08089e63          	bnez	a7,1d70c <__subdf3+0x3cc>
   1d674:	00c766b3          	or	a3,a4,a2
   1d678:	3a068863          	beqz	a3,1da28 <__subdf3+0x6e8>
   1d67c:	fff80693          	addi	a3,a6,-1
   1d680:	02069063          	bnez	a3,1d6a0 <__subdf3+0x360>
   1d684:	40c50633          	sub	a2,a0,a2
   1d688:	00c536b3          	sltu	a3,a0,a2
   1d68c:	40e78733          	sub	a4,a5,a4
   1d690:	00060513          	mv	a0,a2
   1d694:	40d707b3          	sub	a5,a4,a3
   1d698:	00100913          	li	s2,1
   1d69c:	0540006f          	j	1d6f0 <__subdf3+0x3b0>
   1d6a0:	7ff00593          	li	a1,2047
   1d6a4:	d4b80ae3          	beq	a6,a1,1d3f8 <__subdf3+0xb8>
   1d6a8:	03800593          	li	a1,56
   1d6ac:	0ad5c663          	blt	a1,a3,1d758 <__subdf3+0x418>
   1d6b0:	01f00593          	li	a1,31
   1d6b4:	06d5c863          	blt	a1,a3,1d724 <__subdf3+0x3e4>
   1d6b8:	02000813          	li	a6,32
   1d6bc:	40d80833          	sub	a6,a6,a3
   1d6c0:	00d658b3          	srl	a7,a2,a3
   1d6c4:	010715b3          	sll	a1,a4,a6
   1d6c8:	01061833          	sll	a6,a2,a6
   1d6cc:	0115e5b3          	or	a1,a1,a7
   1d6d0:	01003833          	snez	a6,a6
   1d6d4:	0105e633          	or	a2,a1,a6
   1d6d8:	00d756b3          	srl	a3,a4,a3
   1d6dc:	40c50633          	sub	a2,a0,a2
   1d6e0:	00c53733          	sltu	a4,a0,a2
   1d6e4:	40d786b3          	sub	a3,a5,a3
   1d6e8:	00060513          	mv	a0,a2
   1d6ec:	40e687b3          	sub	a5,a3,a4
   1d6f0:	00879713          	slli	a4,a5,0x8
   1d6f4:	f00750e3          	bgez	a4,1d5f4 <__subdf3+0x2b4>
   1d6f8:	00800437          	lui	s0,0x800
   1d6fc:	fff40413          	addi	s0,s0,-1 # 7fffff <__BSS_END__+0x7dd2cf>
   1d700:	0087f433          	and	s0,a5,s0
   1d704:	00050993          	mv	s3,a0
   1d708:	2100006f          	j	1d918 <__subdf3+0x5d8>
   1d70c:	7ff00693          	li	a3,2047
   1d710:	eed902e3          	beq	s2,a3,1d5f4 <__subdf3+0x2b4>
   1d714:	008006b7          	lui	a3,0x800
   1d718:	00d76733          	or	a4,a4,a3
   1d71c:	00080693          	mv	a3,a6
   1d720:	f89ff06f          	j	1d6a8 <__subdf3+0x368>
   1d724:	fe068593          	addi	a1,a3,-32 # 7fffe0 <__BSS_END__+0x7dd2b0>
   1d728:	02000893          	li	a7,32
   1d72c:	00b755b3          	srl	a1,a4,a1
   1d730:	00000813          	li	a6,0
   1d734:	01168863          	beq	a3,a7,1d744 <__subdf3+0x404>
   1d738:	04000813          	li	a6,64
   1d73c:	40d80833          	sub	a6,a6,a3
   1d740:	01071833          	sll	a6,a4,a6
   1d744:	00c86833          	or	a6,a6,a2
   1d748:	01003833          	snez	a6,a6
   1d74c:	0105e633          	or	a2,a1,a6
   1d750:	00000693          	li	a3,0
   1d754:	f89ff06f          	j	1d6dc <__subdf3+0x39c>
   1d758:	00c76633          	or	a2,a4,a2
   1d75c:	00c03633          	snez	a2,a2
   1d760:	ff1ff06f          	j	1d750 <__subdf3+0x410>
   1d764:	0e080863          	beqz	a6,1d854 <__subdf3+0x514>
   1d768:	41288833          	sub	a6,a7,s2
   1d76c:	04091263          	bnez	s2,1d7b0 <__subdf3+0x470>
   1d770:	00a7e5b3          	or	a1,a5,a0
   1d774:	2a058e63          	beqz	a1,1da30 <__subdf3+0x6f0>
   1d778:	fff80593          	addi	a1,a6,-1
   1d77c:	00059e63          	bnez	a1,1d798 <__subdf3+0x458>
   1d780:	40a60533          	sub	a0,a2,a0
   1d784:	40f70733          	sub	a4,a4,a5
   1d788:	00a63633          	sltu	a2,a2,a0
   1d78c:	40c707b3          	sub	a5,a4,a2
   1d790:	00068493          	mv	s1,a3
   1d794:	f05ff06f          	j	1d698 <__subdf3+0x358>
   1d798:	7ff00313          	li	t1,2047
   1d79c:	02681463          	bne	a6,t1,1d7c4 <__subdf3+0x484>
   1d7a0:	00070793          	mv	a5,a4
   1d7a4:	00060513          	mv	a0,a2
   1d7a8:	7ff00913          	li	s2,2047
   1d7ac:	0d00006f          	j	1d87c <__subdf3+0x53c>
   1d7b0:	7ff00593          	li	a1,2047
   1d7b4:	feb886e3          	beq	a7,a1,1d7a0 <__subdf3+0x460>
   1d7b8:	008005b7          	lui	a1,0x800
   1d7bc:	00b7e7b3          	or	a5,a5,a1
   1d7c0:	00080593          	mv	a1,a6
   1d7c4:	03800813          	li	a6,56
   1d7c8:	08b84063          	blt	a6,a1,1d848 <__subdf3+0x508>
   1d7cc:	01f00813          	li	a6,31
   1d7d0:	04b84263          	blt	a6,a1,1d814 <__subdf3+0x4d4>
   1d7d4:	02000313          	li	t1,32
   1d7d8:	40b30333          	sub	t1,t1,a1
   1d7dc:	00b55e33          	srl	t3,a0,a1
   1d7e0:	00679833          	sll	a6,a5,t1
   1d7e4:	00651333          	sll	t1,a0,t1
   1d7e8:	01c86833          	or	a6,a6,t3
   1d7ec:	00603333          	snez	t1,t1
   1d7f0:	00686533          	or	a0,a6,t1
   1d7f4:	00b7d5b3          	srl	a1,a5,a1
   1d7f8:	40a60533          	sub	a0,a2,a0
   1d7fc:	40b705b3          	sub	a1,a4,a1
   1d800:	00a63633          	sltu	a2,a2,a0
   1d804:	40c587b3          	sub	a5,a1,a2
   1d808:	00088913          	mv	s2,a7
   1d80c:	00068493          	mv	s1,a3
   1d810:	ee1ff06f          	j	1d6f0 <__subdf3+0x3b0>
   1d814:	fe058813          	addi	a6,a1,-32 # 7fffe0 <__BSS_END__+0x7dd2b0>
   1d818:	02000e13          	li	t3,32
   1d81c:	0107d833          	srl	a6,a5,a6
   1d820:	00000313          	li	t1,0
   1d824:	01c58863          	beq	a1,t3,1d834 <__subdf3+0x4f4>
   1d828:	04000313          	li	t1,64
   1d82c:	40b30333          	sub	t1,t1,a1
   1d830:	00679333          	sll	t1,a5,t1
   1d834:	00a36333          	or	t1,t1,a0
   1d838:	00603333          	snez	t1,t1
   1d83c:	00686533          	or	a0,a6,t1
   1d840:	00000593          	li	a1,0
   1d844:	fb5ff06f          	j	1d7f8 <__subdf3+0x4b8>
   1d848:	00a7e533          	or	a0,a5,a0
   1d84c:	00a03533          	snez	a0,a0
   1d850:	ff1ff06f          	j	1d840 <__subdf3+0x500>
   1d854:	00190593          	addi	a1,s2,1
   1d858:	7fe5f593          	andi	a1,a1,2046
   1d85c:	08059663          	bnez	a1,1d8e8 <__subdf3+0x5a8>
   1d860:	00c765b3          	or	a1,a4,a2
   1d864:	00a7e833          	or	a6,a5,a0
   1d868:	06091063          	bnez	s2,1d8c8 <__subdf3+0x588>
   1d86c:	00081c63          	bnez	a6,1d884 <__subdf3+0x544>
   1d870:	10058e63          	beqz	a1,1d98c <__subdf3+0x64c>
   1d874:	00070793          	mv	a5,a4
   1d878:	00060513          	mv	a0,a2
   1d87c:	00068493          	mv	s1,a3
   1d880:	d75ff06f          	j	1d5f4 <__subdf3+0x2b4>
   1d884:	d60588e3          	beqz	a1,1d5f4 <__subdf3+0x2b4>
   1d888:	40c50833          	sub	a6,a0,a2
   1d88c:	010538b3          	sltu	a7,a0,a6
   1d890:	40e785b3          	sub	a1,a5,a4
   1d894:	411585b3          	sub	a1,a1,a7
   1d898:	00859893          	slli	a7,a1,0x8
   1d89c:	0008dc63          	bgez	a7,1d8b4 <__subdf3+0x574>
   1d8a0:	40a60533          	sub	a0,a2,a0
   1d8a4:	40f70733          	sub	a4,a4,a5
   1d8a8:	00a63633          	sltu	a2,a2,a0
   1d8ac:	40c707b3          	sub	a5,a4,a2
   1d8b0:	fcdff06f          	j	1d87c <__subdf3+0x53c>
   1d8b4:	00b86533          	or	a0,a6,a1
   1d8b8:	18050463          	beqz	a0,1da40 <__subdf3+0x700>
   1d8bc:	00058793          	mv	a5,a1
   1d8c0:	00080513          	mv	a0,a6
   1d8c4:	d31ff06f          	j	1d5f4 <__subdf3+0x2b4>
   1d8c8:	00081c63          	bnez	a6,1d8e0 <__subdf3+0x5a0>
   1d8cc:	d4058ee3          	beqz	a1,1d628 <__subdf3+0x2e8>
   1d8d0:	00070793          	mv	a5,a4
   1d8d4:	00060513          	mv	a0,a2
   1d8d8:	00068493          	mv	s1,a3
   1d8dc:	b1dff06f          	j	1d3f8 <__subdf3+0xb8>
   1d8e0:	b0058ce3          	beqz	a1,1d3f8 <__subdf3+0xb8>
   1d8e4:	d45ff06f          	j	1d628 <__subdf3+0x2e8>
   1d8e8:	40c505b3          	sub	a1,a0,a2
   1d8ec:	00b53833          	sltu	a6,a0,a1
   1d8f0:	40e78433          	sub	s0,a5,a4
   1d8f4:	41040433          	sub	s0,s0,a6
   1d8f8:	00841813          	slli	a6,s0,0x8
   1d8fc:	00058993          	mv	s3,a1
   1d900:	08085063          	bgez	a6,1d980 <__subdf3+0x640>
   1d904:	40a609b3          	sub	s3,a2,a0
   1d908:	40f70433          	sub	s0,a4,a5
   1d90c:	01363633          	sltu	a2,a2,s3
   1d910:	40c40433          	sub	s0,s0,a2
   1d914:	00068493          	mv	s1,a3
   1d918:	06040e63          	beqz	s0,1d994 <__subdf3+0x654>
   1d91c:	00040513          	mv	a0,s0
   1d920:	795020ef          	jal	208b4 <__clzsi2>
   1d924:	ff850693          	addi	a3,a0,-8
   1d928:	02000793          	li	a5,32
   1d92c:	40d787b3          	sub	a5,a5,a3
   1d930:	00d41433          	sll	s0,s0,a3
   1d934:	00f9d7b3          	srl	a5,s3,a5
   1d938:	0087e7b3          	or	a5,a5,s0
   1d93c:	00d99433          	sll	s0,s3,a3
   1d940:	0b26c463          	blt	a3,s2,1d9e8 <__subdf3+0x6a8>
   1d944:	412686b3          	sub	a3,a3,s2
   1d948:	00168613          	addi	a2,a3,1
   1d94c:	01f00713          	li	a4,31
   1d950:	06c74263          	blt	a4,a2,1d9b4 <__subdf3+0x674>
   1d954:	02000713          	li	a4,32
   1d958:	40c70733          	sub	a4,a4,a2
   1d95c:	00e79533          	sll	a0,a5,a4
   1d960:	00c456b3          	srl	a3,s0,a2
   1d964:	00e41733          	sll	a4,s0,a4
   1d968:	00d56533          	or	a0,a0,a3
   1d96c:	00e03733          	snez	a4,a4
   1d970:	00e56533          	or	a0,a0,a4
   1d974:	00c7d7b3          	srl	a5,a5,a2
   1d978:	00000913          	li	s2,0
   1d97c:	c79ff06f          	j	1d5f4 <__subdf3+0x2b4>
   1d980:	0085e5b3          	or	a1,a1,s0
   1d984:	f8059ae3          	bnez	a1,1d918 <__subdf3+0x5d8>
   1d988:	00000913          	li	s2,0
   1d98c:	00000493          	li	s1,0
   1d990:	08c0006f          	j	1da1c <__subdf3+0x6dc>
   1d994:	00098513          	mv	a0,s3
   1d998:	71d020ef          	jal	208b4 <__clzsi2>
   1d99c:	01850693          	addi	a3,a0,24
   1d9a0:	01f00793          	li	a5,31
   1d9a4:	f8d7d2e3          	bge	a5,a3,1d928 <__subdf3+0x5e8>
   1d9a8:	ff850793          	addi	a5,a0,-8
   1d9ac:	00f997b3          	sll	a5,s3,a5
   1d9b0:	f91ff06f          	j	1d940 <__subdf3+0x600>
   1d9b4:	fe168693          	addi	a3,a3,-31
   1d9b8:	00d7d533          	srl	a0,a5,a3
   1d9bc:	02000693          	li	a3,32
   1d9c0:	00000713          	li	a4,0
   1d9c4:	00d60863          	beq	a2,a3,1d9d4 <__subdf3+0x694>
   1d9c8:	04000713          	li	a4,64
   1d9cc:	40c70733          	sub	a4,a4,a2
   1d9d0:	00e79733          	sll	a4,a5,a4
   1d9d4:	00e46733          	or	a4,s0,a4
   1d9d8:	00e03733          	snez	a4,a4
   1d9dc:	00e56533          	or	a0,a0,a4
   1d9e0:	00000793          	li	a5,0
   1d9e4:	f95ff06f          	j	1d978 <__subdf3+0x638>
   1d9e8:	ff800737          	lui	a4,0xff800
   1d9ec:	fff70713          	addi	a4,a4,-1 # ff7fffff <__BSS_END__+0xff7dd2cf>
   1d9f0:	40d90933          	sub	s2,s2,a3
   1d9f4:	00e7f7b3          	and	a5,a5,a4
   1d9f8:	00040513          	mv	a0,s0
   1d9fc:	bf9ff06f          	j	1d5f4 <__subdf3+0x2b4>
   1da00:	00070793          	mv	a5,a4
   1da04:	00060513          	mv	a0,a2
   1da08:	c5dff06f          	j	1d664 <__subdf3+0x324>
   1da0c:	00070793          	mv	a5,a4
   1da10:	00060513          	mv	a0,a2
   1da14:	be1ff06f          	j	1d5f4 <__subdf3+0x2b4>
   1da18:	7ff00913          	li	s2,2047
   1da1c:	00000793          	li	a5,0
   1da20:	00000513          	li	a0,0
   1da24:	0240006f          	j	1da48 <__subdf3+0x708>
   1da28:	00080913          	mv	s2,a6
   1da2c:	bc9ff06f          	j	1d5f4 <__subdf3+0x2b4>
   1da30:	00070793          	mv	a5,a4
   1da34:	00060513          	mv	a0,a2
   1da38:	00080913          	mv	s2,a6
   1da3c:	e41ff06f          	j	1d87c <__subdf3+0x53c>
   1da40:	00000793          	li	a5,0
   1da44:	00000493          	li	s1,0
   1da48:	00879713          	slli	a4,a5,0x8
   1da4c:	00075e63          	bgez	a4,1da68 <__subdf3+0x728>
   1da50:	00190913          	addi	s2,s2,1
   1da54:	7ff00713          	li	a4,2047
   1da58:	08e90263          	beq	s2,a4,1dadc <__subdf3+0x79c>
   1da5c:	ff800737          	lui	a4,0xff800
   1da60:	fff70713          	addi	a4,a4,-1 # ff7fffff <__BSS_END__+0xff7dd2cf>
   1da64:	00e7f7b3          	and	a5,a5,a4
   1da68:	01d79693          	slli	a3,a5,0x1d
   1da6c:	00355513          	srli	a0,a0,0x3
   1da70:	7ff00713          	li	a4,2047
   1da74:	00a6e6b3          	or	a3,a3,a0
   1da78:	0037d793          	srli	a5,a5,0x3
   1da7c:	00e91e63          	bne	s2,a4,1da98 <__subdf3+0x758>
   1da80:	00f6e6b3          	or	a3,a3,a5
   1da84:	00000793          	li	a5,0
   1da88:	00068863          	beqz	a3,1da98 <__subdf3+0x758>
   1da8c:	000807b7          	lui	a5,0x80
   1da90:	00000693          	li	a3,0
   1da94:	00000493          	li	s1,0
   1da98:	01491713          	slli	a4,s2,0x14
   1da9c:	7ff00637          	lui	a2,0x7ff00
   1daa0:	00c79793          	slli	a5,a5,0xc
   1daa4:	00c77733          	and	a4,a4,a2
   1daa8:	01c12083          	lw	ra,28(sp)
   1daac:	01812403          	lw	s0,24(sp)
   1dab0:	00c7d793          	srli	a5,a5,0xc
   1dab4:	00f767b3          	or	a5,a4,a5
   1dab8:	01f49713          	slli	a4,s1,0x1f
   1dabc:	00e7e633          	or	a2,a5,a4
   1dac0:	01412483          	lw	s1,20(sp)
   1dac4:	01012903          	lw	s2,16(sp)
   1dac8:	00c12983          	lw	s3,12(sp)
   1dacc:	00068513          	mv	a0,a3
   1dad0:	00060593          	mv	a1,a2
   1dad4:	02010113          	addi	sp,sp,32
   1dad8:	00008067          	ret
   1dadc:	00000793          	li	a5,0
   1dae0:	00000513          	li	a0,0
   1dae4:	f85ff06f          	j	1da68 <__subdf3+0x728>

0001dae8 <__fixdfsi>:
   1dae8:	0145d713          	srli	a4,a1,0x14
   1daec:	001006b7          	lui	a3,0x100
   1daf0:	fff68793          	addi	a5,a3,-1 # fffff <__BSS_END__+0xdd2cf>
   1daf4:	7ff77713          	andi	a4,a4,2047
   1daf8:	3fe00613          	li	a2,1022
   1dafc:	00b7f7b3          	and	a5,a5,a1
   1db00:	01f5d593          	srli	a1,a1,0x1f
   1db04:	04e65e63          	bge	a2,a4,1db60 <__fixdfsi+0x78>
   1db08:	41d00613          	li	a2,1053
   1db0c:	00e65a63          	bge	a2,a4,1db20 <__fixdfsi+0x38>
   1db10:	80000537          	lui	a0,0x80000
   1db14:	fff50513          	addi	a0,a0,-1 # 7fffffff <__BSS_END__+0x7ffdd2cf>
   1db18:	00a58533          	add	a0,a1,a0
   1db1c:	00008067          	ret
   1db20:	00d7e7b3          	or	a5,a5,a3
   1db24:	43300693          	li	a3,1075
   1db28:	40e686b3          	sub	a3,a3,a4
   1db2c:	01f00613          	li	a2,31
   1db30:	02d64063          	blt	a2,a3,1db50 <__fixdfsi+0x68>
   1db34:	bed70713          	addi	a4,a4,-1043
   1db38:	00e797b3          	sll	a5,a5,a4
   1db3c:	00d55533          	srl	a0,a0,a3
   1db40:	00a7e533          	or	a0,a5,a0
   1db44:	02058063          	beqz	a1,1db64 <__fixdfsi+0x7c>
   1db48:	40a00533          	neg	a0,a0
   1db4c:	00008067          	ret
   1db50:	41300693          	li	a3,1043
   1db54:	40e68733          	sub	a4,a3,a4
   1db58:	00e7d533          	srl	a0,a5,a4
   1db5c:	fe9ff06f          	j	1db44 <__fixdfsi+0x5c>
   1db60:	00000513          	li	a0,0
   1db64:	00008067          	ret

0001db68 <__floatsidf>:
   1db68:	ff010113          	addi	sp,sp,-16
   1db6c:	00112623          	sw	ra,12(sp)
   1db70:	00812423          	sw	s0,8(sp)
   1db74:	00912223          	sw	s1,4(sp)
   1db78:	08050663          	beqz	a0,1dc04 <__floatsidf+0x9c>
   1db7c:	41f55793          	srai	a5,a0,0x1f
   1db80:	00a7c433          	xor	s0,a5,a0
   1db84:	40f40433          	sub	s0,s0,a5
   1db88:	01f55493          	srli	s1,a0,0x1f
   1db8c:	00040513          	mv	a0,s0
   1db90:	525020ef          	jal	208b4 <__clzsi2>
   1db94:	41e00713          	li	a4,1054
   1db98:	00a00793          	li	a5,10
   1db9c:	40a70733          	sub	a4,a4,a0
   1dba0:	04a7c863          	blt	a5,a0,1dbf0 <__floatsidf+0x88>
   1dba4:	00b00793          	li	a5,11
   1dba8:	40a787b3          	sub	a5,a5,a0
   1dbac:	01550513          	addi	a0,a0,21
   1dbb0:	00f457b3          	srl	a5,s0,a5
   1dbb4:	00a41433          	sll	s0,s0,a0
   1dbb8:	00048513          	mv	a0,s1
   1dbbc:	00c79793          	slli	a5,a5,0xc
   1dbc0:	00c7d793          	srli	a5,a5,0xc
   1dbc4:	01471713          	slli	a4,a4,0x14
   1dbc8:	01f51513          	slli	a0,a0,0x1f
   1dbcc:	00f76733          	or	a4,a4,a5
   1dbd0:	00c12083          	lw	ra,12(sp)
   1dbd4:	00a767b3          	or	a5,a4,a0
   1dbd8:	00040513          	mv	a0,s0
   1dbdc:	00812403          	lw	s0,8(sp)
   1dbe0:	00412483          	lw	s1,4(sp)
   1dbe4:	00078593          	mv	a1,a5
   1dbe8:	01010113          	addi	sp,sp,16
   1dbec:	00008067          	ret
   1dbf0:	ff550513          	addi	a0,a0,-11
   1dbf4:	00a417b3          	sll	a5,s0,a0
   1dbf8:	00048513          	mv	a0,s1
   1dbfc:	00000413          	li	s0,0
   1dc00:	fbdff06f          	j	1dbbc <__floatsidf+0x54>
   1dc04:	00000713          	li	a4,0
   1dc08:	00000793          	li	a5,0
   1dc0c:	ff1ff06f          	j	1dbfc <__floatsidf+0x94>

0001dc10 <__eqtf2>:
   1dc10:	00c52783          	lw	a5,12(a0)
   1dc14:	0005af03          	lw	t5,0(a1)
   1dc18:	0045af83          	lw	t6,4(a1)
   1dc1c:	0085a283          	lw	t0,8(a1)
   1dc20:	00c5a583          	lw	a1,12(a1)
   1dc24:	00008737          	lui	a4,0x8
   1dc28:	0107d693          	srli	a3,a5,0x10
   1dc2c:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1dc30:	01079813          	slli	a6,a5,0x10
   1dc34:	01059e93          	slli	t4,a1,0x10
   1dc38:	01f7d613          	srli	a2,a5,0x1f
   1dc3c:	00e6f6b3          	and	a3,a3,a4
   1dc40:	0105d793          	srli	a5,a1,0x10
   1dc44:	00052883          	lw	a7,0(a0)
   1dc48:	00452303          	lw	t1,4(a0)
   1dc4c:	00852e03          	lw	t3,8(a0)
   1dc50:	ff010113          	addi	sp,sp,-16
   1dc54:	01085813          	srli	a6,a6,0x10
   1dc58:	010ede93          	srli	t4,t4,0x10
   1dc5c:	00e7f7b3          	and	a5,a5,a4
   1dc60:	01f5d593          	srli	a1,a1,0x1f
   1dc64:	02e69063          	bne	a3,a4,1dc84 <__eqtf2+0x74>
   1dc68:	0068e733          	or	a4,a7,t1
   1dc6c:	01c76733          	or	a4,a4,t3
   1dc70:	01076733          	or	a4,a4,a6
   1dc74:	00100513          	li	a0,1
   1dc78:	04071a63          	bnez	a4,1dccc <__eqtf2+0xbc>
   1dc7c:	04d79863          	bne	a5,a3,1dccc <__eqtf2+0xbc>
   1dc80:	0080006f          	j	1dc88 <__eqtf2+0x78>
   1dc84:	00e79c63          	bne	a5,a4,1dc9c <__eqtf2+0x8c>
   1dc88:	01ff6733          	or	a4,t5,t6
   1dc8c:	00576733          	or	a4,a4,t0
   1dc90:	01d76733          	or	a4,a4,t4
   1dc94:	00100513          	li	a0,1
   1dc98:	02071a63          	bnez	a4,1dccc <__eqtf2+0xbc>
   1dc9c:	00100513          	li	a0,1
   1dca0:	02d79663          	bne	a5,a3,1dccc <__eqtf2+0xbc>
   1dca4:	03e89463          	bne	a7,t5,1dccc <__eqtf2+0xbc>
   1dca8:	03f31263          	bne	t1,t6,1dccc <__eqtf2+0xbc>
   1dcac:	025e1063          	bne	t3,t0,1dccc <__eqtf2+0xbc>
   1dcb0:	01d81e63          	bne	a6,t4,1dccc <__eqtf2+0xbc>
   1dcb4:	02b60063          	beq	a2,a1,1dcd4 <__eqtf2+0xc4>
   1dcb8:	00079a63          	bnez	a5,1dccc <__eqtf2+0xbc>
   1dcbc:	0068e533          	or	a0,a7,t1
   1dcc0:	01c56533          	or	a0,a0,t3
   1dcc4:	01056533          	or	a0,a0,a6
   1dcc8:	00a03533          	snez	a0,a0
   1dccc:	01010113          	addi	sp,sp,16
   1dcd0:	00008067          	ret
   1dcd4:	00000513          	li	a0,0
   1dcd8:	ff5ff06f          	j	1dccc <__eqtf2+0xbc>

0001dcdc <__getf2>:
   1dcdc:	00c52783          	lw	a5,12(a0)
   1dce0:	00c5a683          	lw	a3,12(a1)
   1dce4:	00008737          	lui	a4,0x8
   1dce8:	0107d613          	srli	a2,a5,0x10
   1dcec:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1dcf0:	01079813          	slli	a6,a5,0x10
   1dcf4:	01069293          	slli	t0,a3,0x10
   1dcf8:	00052883          	lw	a7,0(a0)
   1dcfc:	00452303          	lw	t1,4(a0)
   1dd00:	00852e03          	lw	t3,8(a0)
   1dd04:	00e67633          	and	a2,a2,a4
   1dd08:	0106d513          	srli	a0,a3,0x10
   1dd0c:	0005ae83          	lw	t4,0(a1)
   1dd10:	0045af03          	lw	t5,4(a1)
   1dd14:	0085af83          	lw	t6,8(a1)
   1dd18:	ff010113          	addi	sp,sp,-16
   1dd1c:	01085813          	srli	a6,a6,0x10
   1dd20:	01f7d793          	srli	a5,a5,0x1f
   1dd24:	0102d293          	srli	t0,t0,0x10
   1dd28:	00e57533          	and	a0,a0,a4
   1dd2c:	01f6d693          	srli	a3,a3,0x1f
   1dd30:	00e61a63          	bne	a2,a4,1dd44 <__getf2+0x68>
   1dd34:	01136733          	or	a4,t1,a7
   1dd38:	01c76733          	or	a4,a4,t3
   1dd3c:	01076733          	or	a4,a4,a6
   1dd40:	0a071463          	bnez	a4,1dde8 <__getf2+0x10c>
   1dd44:	00008737          	lui	a4,0x8
   1dd48:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1dd4c:	00e51a63          	bne	a0,a4,1dd60 <__getf2+0x84>
   1dd50:	01eee733          	or	a4,t4,t5
   1dd54:	01f76733          	or	a4,a4,t6
   1dd58:	00576733          	or	a4,a4,t0
   1dd5c:	08071663          	bnez	a4,1dde8 <__getf2+0x10c>
   1dd60:	00000713          	li	a4,0
   1dd64:	00061a63          	bnez	a2,1dd78 <__getf2+0x9c>
   1dd68:	01136733          	or	a4,t1,a7
   1dd6c:	01c76733          	or	a4,a4,t3
   1dd70:	01076733          	or	a4,a4,a6
   1dd74:	00173713          	seqz	a4,a4
   1dd78:	06051c63          	bnez	a0,1ddf0 <__getf2+0x114>
   1dd7c:	01eee5b3          	or	a1,t4,t5
   1dd80:	01f5e5b3          	or	a1,a1,t6
   1dd84:	0055e5b3          	or	a1,a1,t0
   1dd88:	00070c63          	beqz	a4,1dda0 <__getf2+0xc4>
   1dd8c:	02058063          	beqz	a1,1ddac <__getf2+0xd0>
   1dd90:	00100513          	li	a0,1
   1dd94:	00069c63          	bnez	a3,1ddac <__getf2+0xd0>
   1dd98:	fff00513          	li	a0,-1
   1dd9c:	0100006f          	j	1ddac <__getf2+0xd0>
   1dda0:	04059a63          	bnez	a1,1ddf4 <__getf2+0x118>
   1dda4:	fff00513          	li	a0,-1
   1dda8:	04078e63          	beqz	a5,1de04 <__getf2+0x128>
   1ddac:	01010113          	addi	sp,sp,16
   1ddb0:	00008067          	ret
   1ddb4:	fca64ee3          	blt	a2,a0,1dd90 <__getf2+0xb4>
   1ddb8:	ff02e6e3          	bltu	t0,a6,1dda4 <__getf2+0xc8>
   1ddbc:	02581063          	bne	a6,t0,1dddc <__getf2+0x100>
   1ddc0:	ffcfe2e3          	bltu	t6,t3,1dda4 <__getf2+0xc8>
   1ddc4:	01fe1c63          	bne	t3,t6,1dddc <__getf2+0x100>
   1ddc8:	fc6f6ee3          	bltu	t5,t1,1dda4 <__getf2+0xc8>
   1ddcc:	01e31863          	bne	t1,t5,1dddc <__getf2+0x100>
   1ddd0:	fd1eeae3          	bltu	t4,a7,1dda4 <__getf2+0xc8>
   1ddd4:	00000513          	li	a0,0
   1ddd8:	fdd8fae3          	bgeu	a7,t4,1ddac <__getf2+0xd0>
   1dddc:	00100513          	li	a0,1
   1dde0:	fc0796e3          	bnez	a5,1ddac <__getf2+0xd0>
   1dde4:	fb5ff06f          	j	1dd98 <__getf2+0xbc>
   1dde8:	ffe00513          	li	a0,-2
   1ddec:	fc1ff06f          	j	1ddac <__getf2+0xd0>
   1ddf0:	fa0710e3          	bnez	a4,1dd90 <__getf2+0xb4>
   1ddf4:	faf698e3          	bne	a3,a5,1dda4 <__getf2+0xc8>
   1ddf8:	fac55ee3          	bge	a0,a2,1ddb4 <__getf2+0xd8>
   1ddfc:	fff00513          	li	a0,-1
   1de00:	fa0696e3          	bnez	a3,1ddac <__getf2+0xd0>
   1de04:	00100513          	li	a0,1
   1de08:	fa5ff06f          	j	1ddac <__getf2+0xd0>

0001de0c <__letf2>:
   1de0c:	00c52783          	lw	a5,12(a0)
   1de10:	00c5a683          	lw	a3,12(a1)
   1de14:	00008737          	lui	a4,0x8
   1de18:	0107d613          	srli	a2,a5,0x10
   1de1c:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1de20:	01079813          	slli	a6,a5,0x10
   1de24:	01069293          	slli	t0,a3,0x10
   1de28:	00052883          	lw	a7,0(a0)
   1de2c:	00452303          	lw	t1,4(a0)
   1de30:	00852e03          	lw	t3,8(a0)
   1de34:	00e67633          	and	a2,a2,a4
   1de38:	0106d513          	srli	a0,a3,0x10
   1de3c:	0005ae83          	lw	t4,0(a1)
   1de40:	0045af03          	lw	t5,4(a1)
   1de44:	0085af83          	lw	t6,8(a1)
   1de48:	ff010113          	addi	sp,sp,-16
   1de4c:	01085813          	srli	a6,a6,0x10
   1de50:	01f7d793          	srli	a5,a5,0x1f
   1de54:	0102d293          	srli	t0,t0,0x10
   1de58:	00e57533          	and	a0,a0,a4
   1de5c:	01f6d693          	srli	a3,a3,0x1f
   1de60:	00e61a63          	bne	a2,a4,1de74 <__letf2+0x68>
   1de64:	01136733          	or	a4,t1,a7
   1de68:	01c76733          	or	a4,a4,t3
   1de6c:	01076733          	or	a4,a4,a6
   1de70:	0a071463          	bnez	a4,1df18 <__letf2+0x10c>
   1de74:	00008737          	lui	a4,0x8
   1de78:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1de7c:	00e51a63          	bne	a0,a4,1de90 <__letf2+0x84>
   1de80:	01eee733          	or	a4,t4,t5
   1de84:	01f76733          	or	a4,a4,t6
   1de88:	00576733          	or	a4,a4,t0
   1de8c:	08071663          	bnez	a4,1df18 <__letf2+0x10c>
   1de90:	00000713          	li	a4,0
   1de94:	00061a63          	bnez	a2,1dea8 <__letf2+0x9c>
   1de98:	01136733          	or	a4,t1,a7
   1de9c:	01c76733          	or	a4,a4,t3
   1dea0:	01076733          	or	a4,a4,a6
   1dea4:	00173713          	seqz	a4,a4
   1dea8:	06051c63          	bnez	a0,1df20 <__letf2+0x114>
   1deac:	01eee5b3          	or	a1,t4,t5
   1deb0:	01f5e5b3          	or	a1,a1,t6
   1deb4:	0055e5b3          	or	a1,a1,t0
   1deb8:	00070c63          	beqz	a4,1ded0 <__letf2+0xc4>
   1debc:	02058063          	beqz	a1,1dedc <__letf2+0xd0>
   1dec0:	00100513          	li	a0,1
   1dec4:	00069c63          	bnez	a3,1dedc <__letf2+0xd0>
   1dec8:	fff00513          	li	a0,-1
   1decc:	0100006f          	j	1dedc <__letf2+0xd0>
   1ded0:	04059a63          	bnez	a1,1df24 <__letf2+0x118>
   1ded4:	fff00513          	li	a0,-1
   1ded8:	04078e63          	beqz	a5,1df34 <__letf2+0x128>
   1dedc:	01010113          	addi	sp,sp,16
   1dee0:	00008067          	ret
   1dee4:	fca64ee3          	blt	a2,a0,1dec0 <__letf2+0xb4>
   1dee8:	ff02e6e3          	bltu	t0,a6,1ded4 <__letf2+0xc8>
   1deec:	02581063          	bne	a6,t0,1df0c <__letf2+0x100>
   1def0:	ffcfe2e3          	bltu	t6,t3,1ded4 <__letf2+0xc8>
   1def4:	01fe1c63          	bne	t3,t6,1df0c <__letf2+0x100>
   1def8:	fc6f6ee3          	bltu	t5,t1,1ded4 <__letf2+0xc8>
   1defc:	01e31863          	bne	t1,t5,1df0c <__letf2+0x100>
   1df00:	fd1eeae3          	bltu	t4,a7,1ded4 <__letf2+0xc8>
   1df04:	00000513          	li	a0,0
   1df08:	fdd8fae3          	bgeu	a7,t4,1dedc <__letf2+0xd0>
   1df0c:	00100513          	li	a0,1
   1df10:	fc0796e3          	bnez	a5,1dedc <__letf2+0xd0>
   1df14:	fb5ff06f          	j	1dec8 <__letf2+0xbc>
   1df18:	00200513          	li	a0,2
   1df1c:	fc1ff06f          	j	1dedc <__letf2+0xd0>
   1df20:	fa0710e3          	bnez	a4,1dec0 <__letf2+0xb4>
   1df24:	faf698e3          	bne	a3,a5,1ded4 <__letf2+0xc8>
   1df28:	fac55ee3          	bge	a0,a2,1dee4 <__letf2+0xd8>
   1df2c:	fff00513          	li	a0,-1
   1df30:	fa0696e3          	bnez	a3,1dedc <__letf2+0xd0>
   1df34:	00100513          	li	a0,1
   1df38:	fa5ff06f          	j	1dedc <__letf2+0xd0>

0001df3c <__multf3>:
   1df3c:	f5010113          	addi	sp,sp,-176
   1df40:	09412c23          	sw	s4,152(sp)
   1df44:	00c5aa03          	lw	s4,12(a1)
   1df48:	0005a783          	lw	a5,0(a1)
   1df4c:	0085a683          	lw	a3,8(a1)
   1df50:	0a812423          	sw	s0,168(sp)
   1df54:	00050413          	mv	s0,a0
   1df58:	0045a503          	lw	a0,4(a1)
   1df5c:	010a1713          	slli	a4,s4,0x10
   1df60:	0b212023          	sw	s2,160(sp)
   1df64:	09312e23          	sw	s3,156(sp)
   1df68:	00062903          	lw	s2,0(a2) # 7ff00000 <__BSS_END__+0x7fedd2d0>
   1df6c:	00c62983          	lw	s3,12(a2)
   1df70:	09512a23          	sw	s5,148(sp)
   1df74:	09612823          	sw	s6,144(sp)
   1df78:	00862a83          	lw	s5,8(a2)
   1df7c:	00462b03          	lw	s6,4(a2)
   1df80:	00008637          	lui	a2,0x8
   1df84:	0a912223          	sw	s1,164(sp)
   1df88:	01075713          	srli	a4,a4,0x10
   1df8c:	010a5493          	srli	s1,s4,0x10
   1df90:	fff60613          	addi	a2,a2,-1 # 7fff <exit-0x80b5>
   1df94:	05412e23          	sw	s4,92(sp)
   1df98:	0a112623          	sw	ra,172(sp)
   1df9c:	09712623          	sw	s7,140(sp)
   1dfa0:	09812423          	sw	s8,136(sp)
   1dfa4:	09912223          	sw	s9,132(sp)
   1dfa8:	09a12023          	sw	s10,128(sp)
   1dfac:	07b12e23          	sw	s11,124(sp)
   1dfb0:	04f12823          	sw	a5,80(sp)
   1dfb4:	04a12a23          	sw	a0,84(sp)
   1dfb8:	04d12c23          	sw	a3,88(sp)
   1dfbc:	02f12023          	sw	a5,32(sp)
   1dfc0:	02a12223          	sw	a0,36(sp)
   1dfc4:	02d12423          	sw	a3,40(sp)
   1dfc8:	02e12623          	sw	a4,44(sp)
   1dfcc:	00c4f4b3          	and	s1,s1,a2
   1dfd0:	01fa5a13          	srli	s4,s4,0x1f
   1dfd4:	080482e3          	beqz	s1,1e858 <__multf3+0x91c>
   1dfd8:	1ac48ce3          	beq	s1,a2,1e990 <__multf3+0xa54>
   1dfdc:	000106b7          	lui	a3,0x10
   1dfe0:	00d76733          	or	a4,a4,a3
   1dfe4:	02e12623          	sw	a4,44(sp)
   1dfe8:	02010593          	addi	a1,sp,32
   1dfec:	02c10713          	addi	a4,sp,44
   1dff0:	00072683          	lw	a3,0(a4)
   1dff4:	ffc72603          	lw	a2,-4(a4)
   1dff8:	ffc70713          	addi	a4,a4,-4
   1dffc:	00369693          	slli	a3,a3,0x3
   1e000:	01d65613          	srli	a2,a2,0x1d
   1e004:	00c6e6b3          	or	a3,a3,a2
   1e008:	00d72223          	sw	a3,4(a4)
   1e00c:	fee592e3          	bne	a1,a4,1dff0 <__multf3+0xb4>
   1e010:	00379793          	slli	a5,a5,0x3
   1e014:	02f12023          	sw	a5,32(sp)
   1e018:	ffffc7b7          	lui	a5,0xffffc
   1e01c:	00178793          	addi	a5,a5,1 # ffffc001 <__BSS_END__+0xfffd92d1>
   1e020:	00f484b3          	add	s1,s1,a5
   1e024:	00000b93          	li	s7,0
   1e028:	01099513          	slli	a0,s3,0x10
   1e02c:	00008737          	lui	a4,0x8
   1e030:	0109d793          	srli	a5,s3,0x10
   1e034:	01055513          	srli	a0,a0,0x10
   1e038:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1e03c:	05312e23          	sw	s3,92(sp)
   1e040:	05212823          	sw	s2,80(sp)
   1e044:	05612a23          	sw	s6,84(sp)
   1e048:	05512c23          	sw	s5,88(sp)
   1e04c:	03212823          	sw	s2,48(sp)
   1e050:	03612a23          	sw	s6,52(sp)
   1e054:	03512c23          	sw	s5,56(sp)
   1e058:	02a12e23          	sw	a0,60(sp)
   1e05c:	00e7f7b3          	and	a5,a5,a4
   1e060:	01f9d993          	srli	s3,s3,0x1f
   1e064:	14078ae3          	beqz	a5,1e9b8 <__multf3+0xa7c>
   1e068:	28e784e3          	beq	a5,a4,1eaf0 <__multf3+0xbb4>
   1e06c:	00010737          	lui	a4,0x10
   1e070:	00e56533          	or	a0,a0,a4
   1e074:	02a12e23          	sw	a0,60(sp)
   1e078:	03010593          	addi	a1,sp,48
   1e07c:	03c10713          	addi	a4,sp,60
   1e080:	00072683          	lw	a3,0(a4) # 10000 <exit-0xb4>
   1e084:	ffc72603          	lw	a2,-4(a4)
   1e088:	ffc70713          	addi	a4,a4,-4
   1e08c:	00369693          	slli	a3,a3,0x3
   1e090:	01d65613          	srli	a2,a2,0x1d
   1e094:	00c6e6b3          	or	a3,a3,a2
   1e098:	00d72223          	sw	a3,4(a4)
   1e09c:	fee592e3          	bne	a1,a4,1e080 <__multf3+0x144>
   1e0a0:	ffffc737          	lui	a4,0xffffc
   1e0a4:	00391913          	slli	s2,s2,0x3
   1e0a8:	00170713          	addi	a4,a4,1 # ffffc001 <__BSS_END__+0xfffd92d1>
   1e0ac:	03212823          	sw	s2,48(sp)
   1e0b0:	00e787b3          	add	a5,a5,a4
   1e0b4:	00000693          	li	a3,0
   1e0b8:	009787b3          	add	a5,a5,s1
   1e0bc:	00f12623          	sw	a5,12(sp)
   1e0c0:	00178793          	addi	a5,a5,1
   1e0c4:	00f12423          	sw	a5,8(sp)
   1e0c8:	002b9793          	slli	a5,s7,0x2
   1e0cc:	00d7e7b3          	or	a5,a5,a3
   1e0d0:	00a00713          	li	a4,10
   1e0d4:	28f742e3          	blt	a4,a5,1eb58 <__multf3+0xc1c>
   1e0d8:	013a4733          	xor	a4,s4,s3
   1e0dc:	00e12223          	sw	a4,4(sp)
   1e0e0:	00200713          	li	a4,2
   1e0e4:	22f74ae3          	blt	a4,a5,1eb18 <__multf3+0xbdc>
   1e0e8:	fff78793          	addi	a5,a5,-1
   1e0ec:	00100713          	li	a4,1
   1e0f0:	28f770e3          	bgeu	a4,a5,1eb70 <__multf3+0xc34>
   1e0f4:	02012883          	lw	a7,32(sp)
   1e0f8:	03012f03          	lw	t5,48(sp)
   1e0fc:	000105b7          	lui	a1,0x10
   1e100:	fff58713          	addi	a4,a1,-1 # ffff <exit-0xb5>
   1e104:	0108d913          	srli	s2,a7,0x10
   1e108:	010f5493          	srli	s1,t5,0x10
   1e10c:	00e8f8b3          	and	a7,a7,a4
   1e110:	00ef7f33          	and	t5,t5,a4
   1e114:	031f0633          	mul	a2,t5,a7
   1e118:	03e90533          	mul	a0,s2,t5
   1e11c:	01065793          	srli	a5,a2,0x10
   1e120:	031486b3          	mul	a3,s1,a7
   1e124:	00a686b3          	add	a3,a3,a0
   1e128:	00d787b3          	add	a5,a5,a3
   1e12c:	02990bb3          	mul	s7,s2,s1
   1e130:	00a7f463          	bgeu	a5,a0,1e138 <__multf3+0x1fc>
   1e134:	00bb8bb3          	add	s7,s7,a1
   1e138:	03412e83          	lw	t4,52(sp)
   1e13c:	0107d693          	srli	a3,a5,0x10
   1e140:	00e7f7b3          	and	a5,a5,a4
   1e144:	00e67633          	and	a2,a2,a4
   1e148:	01079793          	slli	a5,a5,0x10
   1e14c:	00c787b3          	add	a5,a5,a2
   1e150:	010ed293          	srli	t0,t4,0x10
   1e154:	00eefeb3          	and	t4,t4,a4
   1e158:	03d90633          	mul	a2,s2,t4
   1e15c:	00f12823          	sw	a5,16(sp)
   1e160:	04f12823          	sw	a5,80(sp)
   1e164:	03128733          	mul	a4,t0,a7
   1e168:	031e87b3          	mul	a5,t4,a7
   1e16c:	00c70733          	add	a4,a4,a2
   1e170:	0107d313          	srli	t1,a5,0x10
   1e174:	00e30333          	add	t1,t1,a4
   1e178:	02590b33          	mul	s6,s2,t0
   1e17c:	00c37663          	bgeu	t1,a2,1e188 <__multf3+0x24c>
   1e180:	00010737          	lui	a4,0x10
   1e184:	00eb0b33          	add	s6,s6,a4
   1e188:	02412803          	lw	a6,36(sp)
   1e18c:	00010737          	lui	a4,0x10
   1e190:	fff70613          	addi	a2,a4,-1 # ffff <exit-0xb5>
   1e194:	01035513          	srli	a0,t1,0x10
   1e198:	00c37333          	and	t1,t1,a2
   1e19c:	00c7f7b3          	and	a5,a5,a2
   1e1a0:	01085393          	srli	t2,a6,0x10
   1e1a4:	01031313          	slli	t1,t1,0x10
   1e1a8:	00c87833          	and	a6,a6,a2
   1e1ac:	00f30333          	add	t1,t1,a5
   1e1b0:	03e38fb3          	mul	t6,t2,t5
   1e1b4:	006686b3          	add	a3,a3,t1
   1e1b8:	03e807b3          	mul	a5,a6,t5
   1e1bc:	03048e33          	mul	t3,s1,a6
   1e1c0:	0107d613          	srli	a2,a5,0x10
   1e1c4:	01fe0e33          	add	t3,t3,t6
   1e1c8:	01c60633          	add	a2,a2,t3
   1e1cc:	027485b3          	mul	a1,s1,t2
   1e1d0:	01f67463          	bgeu	a2,t6,1e1d8 <__multf3+0x29c>
   1e1d4:	00e585b3          	add	a1,a1,a4
   1e1d8:	01065a93          	srli	s5,a2,0x10
   1e1dc:	00010737          	lui	a4,0x10
   1e1e0:	00ba8ab3          	add	s5,s5,a1
   1e1e4:	fff70593          	addi	a1,a4,-1 # ffff <exit-0xb5>
   1e1e8:	00b7f7b3          	and	a5,a5,a1
   1e1ec:	00b67633          	and	a2,a2,a1
   1e1f0:	01061613          	slli	a2,a2,0x10
   1e1f4:	030e85b3          	mul	a1,t4,a6
   1e1f8:	00f60633          	add	a2,a2,a5
   1e1fc:	03d38fb3          	mul	t6,t2,t4
   1e200:	0105d793          	srli	a5,a1,0x10
   1e204:	03028e33          	mul	t3,t0,a6
   1e208:	01fe0e33          	add	t3,t3,t6
   1e20c:	01c787b3          	add	a5,a5,t3
   1e210:	027289b3          	mul	s3,t0,t2
   1e214:	01f7f463          	bgeu	a5,t6,1e21c <__multf3+0x2e0>
   1e218:	00e989b3          	add	s3,s3,a4
   1e21c:	0107d713          	srli	a4,a5,0x10
   1e220:	01370733          	add	a4,a4,s3
   1e224:	00010a37          	lui	s4,0x10
   1e228:	00e12a23          	sw	a4,20(sp)
   1e22c:	fffa0713          	addi	a4,s4,-1 # ffff <exit-0xb5>
   1e230:	03812e03          	lw	t3,56(sp)
   1e234:	00e7f7b3          	and	a5,a5,a4
   1e238:	00e5f5b3          	and	a1,a1,a4
   1e23c:	01079793          	slli	a5,a5,0x10
   1e240:	00b787b3          	add	a5,a5,a1
   1e244:	00f12c23          	sw	a5,24(sp)
   1e248:	010e5793          	srli	a5,t3,0x10
   1e24c:	00ee7e33          	and	t3,t3,a4
   1e250:	031e05b3          	mul	a1,t3,a7
   1e254:	03c90c33          	mul	s8,s2,t3
   1e258:	0105d713          	srli	a4,a1,0x10
   1e25c:	031789b3          	mul	s3,a5,a7
   1e260:	018989b3          	add	s3,s3,s8
   1e264:	01370733          	add	a4,a4,s3
   1e268:	02f90fb3          	mul	t6,s2,a5
   1e26c:	01877463          	bgeu	a4,s8,1e274 <__multf3+0x338>
   1e270:	014f8fb3          	add	t6,t6,s4
   1e274:	01075993          	srli	s3,a4,0x10
   1e278:	00010cb7          	lui	s9,0x10
   1e27c:	01f98fb3          	add	t6,s3,t6
   1e280:	fffc8993          	addi	s3,s9,-1 # ffff <exit-0xb5>
   1e284:	01377733          	and	a4,a4,s3
   1e288:	0135f5b3          	and	a1,a1,s3
   1e28c:	01071713          	slli	a4,a4,0x10
   1e290:	00b70733          	add	a4,a4,a1
   1e294:	02812583          	lw	a1,40(sp)
   1e298:	01f12e23          	sw	t6,28(sp)
   1e29c:	0105df93          	srli	t6,a1,0x10
   1e2a0:	0135f5b3          	and	a1,a1,s3
   1e2a4:	03e58a33          	mul	s4,a1,t5
   1e2a8:	03ef8d33          	mul	s10,t6,t5
   1e2ac:	010a5d93          	srli	s11,s4,0x10
   1e2b0:	02b489b3          	mul	s3,s1,a1
   1e2b4:	01a989b3          	add	s3,s3,s10
   1e2b8:	013d89b3          	add	s3,s11,s3
   1e2bc:	03f48c33          	mul	s8,s1,t6
   1e2c0:	01a9f463          	bgeu	s3,s10,1e2c8 <__multf3+0x38c>
   1e2c4:	019c0c33          	add	s8,s8,s9
   1e2c8:	00db86b3          	add	a3,s7,a3
   1e2cc:	0066b333          	sltu	t1,a3,t1
   1e2d0:	0109dd13          	srli	s10,s3,0x10
   1e2d4:	00650533          	add	a0,a0,t1
   1e2d8:	00010cb7          	lui	s9,0x10
   1e2dc:	01650533          	add	a0,a0,s6
   1e2e0:	018d0d33          	add	s10,s10,s8
   1e2e4:	00c68633          	add	a2,a3,a2
   1e2e8:	fffc8c13          	addi	s8,s9,-1 # ffff <exit-0xb5>
   1e2ec:	01550ab3          	add	s5,a0,s5
   1e2f0:	0189f9b3          	and	s3,s3,s8
   1e2f4:	00d636b3          	sltu	a3,a2,a3
   1e2f8:	00da86b3          	add	a3,s5,a3
   1e2fc:	01099993          	slli	s3,s3,0x10
   1e300:	018a7a33          	and	s4,s4,s8
   1e304:	01498a33          	add	s4,s3,s4
   1e308:	00aab9b3          	sltu	s3,s5,a0
   1e30c:	0156bab3          	sltu	s5,a3,s5
   1e310:	0159e9b3          	or	s3,s3,s5
   1e314:	00653533          	sltu	a0,a0,t1
   1e318:	00a98533          	add	a0,s3,a0
   1e31c:	01812303          	lw	t1,24(sp)
   1e320:	01412983          	lw	s3,20(sp)
   1e324:	04c12a23          	sw	a2,84(sp)
   1e328:	00668333          	add	t1,a3,t1
   1e32c:	01350ab3          	add	s5,a0,s3
   1e330:	01c12983          	lw	s3,28(sp)
   1e334:	00d336b3          	sltu	a3,t1,a3
   1e338:	00da86b3          	add	a3,s5,a3
   1e33c:	00e30733          	add	a4,t1,a4
   1e340:	01368b33          	add	s6,a3,s3
   1e344:	00673333          	sltu	t1,a4,t1
   1e348:	006b0333          	add	t1,s6,t1
   1e34c:	01470a33          	add	s4,a4,s4
   1e350:	01a30d33          	add	s10,t1,s10
   1e354:	00ea3733          	sltu	a4,s4,a4
   1e358:	00aab533          	sltu	a0,s5,a0
   1e35c:	00ed0733          	add	a4,s10,a4
   1e360:	0156bab3          	sltu	s5,a3,s5
   1e364:	00db36b3          	sltu	a3,s6,a3
   1e368:	01633b33          	sltu	s6,t1,s6
   1e36c:	0166e6b3          	or	a3,a3,s6
   1e370:	006d39b3          	sltu	s3,s10,t1
   1e374:	01556ab3          	or	s5,a0,s5
   1e378:	01a73d33          	sltu	s10,a4,s10
   1e37c:	00da8ab3          	add	s5,s5,a3
   1e380:	01a9e9b3          	or	s3,s3,s10
   1e384:	015989b3          	add	s3,s3,s5
   1e388:	03c12a83          	lw	s5,60(sp)
   1e38c:	05412c23          	sw	s4,88(sp)
   1e390:	010adb13          	srli	s6,s5,0x10
   1e394:	018afab3          	and	s5,s5,s8
   1e398:	031a86b3          	mul	a3,s5,a7
   1e39c:	03590533          	mul	a0,s2,s5
   1e3a0:	031b08b3          	mul	a7,s6,a7
   1e3a4:	00a88333          	add	t1,a7,a0
   1e3a8:	0106d893          	srli	a7,a3,0x10
   1e3ac:	006888b3          	add	a7,a7,t1
   1e3b0:	03690933          	mul	s2,s2,s6
   1e3b4:	00a8f463          	bgeu	a7,a0,1e3bc <__multf3+0x480>
   1e3b8:	01990933          	add	s2,s2,s9
   1e3bc:	02c12b83          	lw	s7,44(sp)
   1e3c0:	0108d513          	srli	a0,a7,0x10
   1e3c4:	01250533          	add	a0,a0,s2
   1e3c8:	00010c37          	lui	s8,0x10
   1e3cc:	00a12a23          	sw	a0,20(sp)
   1e3d0:	fffc0513          	addi	a0,s8,-1 # ffff <exit-0xb5>
   1e3d4:	010bd913          	srli	s2,s7,0x10
   1e3d8:	00abfbb3          	and	s7,s7,a0
   1e3dc:	00a6f6b3          	and	a3,a3,a0
   1e3e0:	00a8f8b3          	and	a7,a7,a0
   1e3e4:	03248333          	mul	t1,s1,s2
   1e3e8:	01089893          	slli	a7,a7,0x10
   1e3ec:	00d888b3          	add	a7,a7,a3
   1e3f0:	03eb8533          	mul	a0,s7,t5
   1e3f4:	037484b3          	mul	s1,s1,s7
   1e3f8:	01055693          	srli	a3,a0,0x10
   1e3fc:	03e90f33          	mul	t5,s2,t5
   1e400:	01e484b3          	add	s1,s1,t5
   1e404:	009686b3          	add	a3,a3,s1
   1e408:	01e6f463          	bgeu	a3,t5,1e410 <__multf3+0x4d4>
   1e40c:	01830333          	add	t1,t1,s8
   1e410:	0106df13          	srli	t5,a3,0x10
   1e414:	006f0333          	add	t1,t5,t1
   1e418:	00010cb7          	lui	s9,0x10
   1e41c:	00612c23          	sw	t1,24(sp)
   1e420:	fffc8313          	addi	t1,s9,-1 # ffff <exit-0xb5>
   1e424:	00657533          	and	a0,a0,t1
   1e428:	0066f6b3          	and	a3,a3,t1
   1e42c:	03c38f33          	mul	t5,t2,t3
   1e430:	01069693          	slli	a3,a3,0x10
   1e434:	00a686b3          	add	a3,a3,a0
   1e438:	03c80333          	mul	t1,a6,t3
   1e43c:	030784b3          	mul	s1,a5,a6
   1e440:	01035513          	srli	a0,t1,0x10
   1e444:	01e484b3          	add	s1,s1,t5
   1e448:	00950533          	add	a0,a0,s1
   1e44c:	02f38c33          	mul	s8,t2,a5
   1e450:	01e57463          	bgeu	a0,t5,1e458 <__multf3+0x51c>
   1e454:	019c0c33          	add	s8,s8,s9
   1e458:	00010d37          	lui	s10,0x10
   1e45c:	fffd0f13          	addi	t5,s10,-1 # ffff <exit-0xb5>
   1e460:	01055493          	srli	s1,a0,0x10
   1e464:	01e57533          	and	a0,a0,t5
   1e468:	01e37333          	and	t1,t1,t5
   1e46c:	01051513          	slli	a0,a0,0x10
   1e470:	018484b3          	add	s1,s1,s8
   1e474:	02b28f33          	mul	t5,t0,a1
   1e478:	00650533          	add	a0,a0,t1
   1e47c:	03df8c33          	mul	s8,t6,t4
   1e480:	02be8333          	mul	t1,t4,a1
   1e484:	018f0f33          	add	t5,t5,s8
   1e488:	01035d93          	srli	s11,t1,0x10
   1e48c:	01ed8f33          	add	t5,s11,t5
   1e490:	03f28cb3          	mul	s9,t0,t6
   1e494:	018f7463          	bgeu	t5,s8,1e49c <__multf3+0x560>
   1e498:	01ac8cb3          	add	s9,s9,s10
   1e49c:	010f5c13          	srli	s8,t5,0x10
   1e4a0:	019c0c33          	add	s8,s8,s9
   1e4a4:	00010cb7          	lui	s9,0x10
   1e4a8:	fffc8d13          	addi	s10,s9,-1 # ffff <exit-0xb5>
   1e4ac:	01af7f33          	and	t5,t5,s10
   1e4b0:	010f1f13          	slli	t5,t5,0x10
   1e4b4:	01a37333          	and	t1,t1,s10
   1e4b8:	006f0333          	add	t1,t5,t1
   1e4bc:	01412f03          	lw	t5,20(sp)
   1e4c0:	011708b3          	add	a7,a4,a7
   1e4c4:	01812d03          	lw	s10,24(sp)
   1e4c8:	01e98f33          	add	t5,s3,t5
   1e4cc:	00e8b733          	sltu	a4,a7,a4
   1e4d0:	00ef0733          	add	a4,t5,a4
   1e4d4:	00d886b3          	add	a3,a7,a3
   1e4d8:	01a70d33          	add	s10,a4,s10
   1e4dc:	0116b8b3          	sltu	a7,a3,a7
   1e4e0:	011d08b3          	add	a7,s10,a7
   1e4e4:	00a68533          	add	a0,a3,a0
   1e4e8:	009884b3          	add	s1,a7,s1
   1e4ec:	00d536b3          	sltu	a3,a0,a3
   1e4f0:	00d486b3          	add	a3,s1,a3
   1e4f4:	013f39b3          	sltu	s3,t5,s3
   1e4f8:	01e73f33          	sltu	t5,a4,t5
   1e4fc:	00ed3733          	sltu	a4,s10,a4
   1e500:	01a8bd33          	sltu	s10,a7,s10
   1e504:	01e9ef33          	or	t5,s3,t5
   1e508:	0114b8b3          	sltu	a7,s1,a7
   1e50c:	01a76733          	or	a4,a4,s10
   1e510:	0096b4b3          	sltu	s1,a3,s1
   1e514:	01868c33          	add	s8,a3,s8
   1e518:	00ef0733          	add	a4,t5,a4
   1e51c:	0098e8b3          	or	a7,a7,s1
   1e520:	00e884b3          	add	s1,a7,a4
   1e524:	03cf8f33          	mul	t5,t6,t3
   1e528:	00dc38b3          	sltu	a7,s8,a3
   1e52c:	00650333          	add	t1,a0,t1
   1e530:	00a33533          	sltu	a0,t1,a0
   1e534:	00ac0533          	add	a0,s8,a0
   1e538:	01853c33          	sltu	s8,a0,s8
   1e53c:	0188e8b3          	or	a7,a7,s8
   1e540:	04612e23          	sw	t1,92(sp)
   1e544:	009888b3          	add	a7,a7,s1
   1e548:	02be06b3          	mul	a3,t3,a1
   1e54c:	02b789b3          	mul	s3,a5,a1
   1e550:	0106d713          	srli	a4,a3,0x10
   1e554:	01e989b3          	add	s3,s3,t5
   1e558:	01370733          	add	a4,a4,s3
   1e55c:	03f784b3          	mul	s1,a5,t6
   1e560:	01e77463          	bgeu	a4,t5,1e568 <__multf3+0x62c>
   1e564:	019484b3          	add	s1,s1,s9
   1e568:	01075f13          	srli	t5,a4,0x10
   1e56c:	009f0f33          	add	t5,t5,s1
   1e570:	000104b7          	lui	s1,0x10
   1e574:	fff48993          	addi	s3,s1,-1 # ffff <exit-0xb5>
   1e578:	01377733          	and	a4,a4,s3
   1e57c:	0136f6b3          	and	a3,a3,s3
   1e580:	01071713          	slli	a4,a4,0x10
   1e584:	035389b3          	mul	s3,t2,s5
   1e588:	00d70733          	add	a4,a4,a3
   1e58c:	030a86b3          	mul	a3,s5,a6
   1e590:	030b0833          	mul	a6,s6,a6
   1e594:	01380c33          	add	s8,a6,s3
   1e598:	0106d813          	srli	a6,a3,0x10
   1e59c:	01880833          	add	a6,a6,s8
   1e5a0:	036383b3          	mul	t2,t2,s6
   1e5a4:	01387463          	bgeu	a6,s3,1e5ac <__multf3+0x670>
   1e5a8:	009383b3          	add	t2,t2,s1
   1e5ac:	01085493          	srli	s1,a6,0x10
   1e5b0:	00010c37          	lui	s8,0x10
   1e5b4:	007483b3          	add	t2,s1,t2
   1e5b8:	fffc0493          	addi	s1,s8,-1 # ffff <exit-0xb5>
   1e5bc:	0096f6b3          	and	a3,a3,s1
   1e5c0:	00987833          	and	a6,a6,s1
   1e5c4:	01081813          	slli	a6,a6,0x10
   1e5c8:	03db89b3          	mul	s3,s7,t4
   1e5cc:	00d80833          	add	a6,a6,a3
   1e5d0:	03d90eb3          	mul	t4,s2,t4
   1e5d4:	0109d693          	srli	a3,s3,0x10
   1e5d8:	032284b3          	mul	s1,t0,s2
   1e5dc:	037282b3          	mul	t0,t0,s7
   1e5e0:	01d282b3          	add	t0,t0,t4
   1e5e4:	005686b3          	add	a3,a3,t0
   1e5e8:	01d6f463          	bgeu	a3,t4,1e5f0 <__multf3+0x6b4>
   1e5ec:	018484b3          	add	s1,s1,s8
   1e5f0:	0106de93          	srli	t4,a3,0x10
   1e5f4:	009e8eb3          	add	t4,t4,s1
   1e5f8:	000104b7          	lui	s1,0x10
   1e5fc:	fff48293          	addi	t0,s1,-1 # ffff <exit-0xb5>
   1e600:	0056f6b3          	and	a3,a3,t0
   1e604:	0059f9b3          	and	s3,s3,t0
   1e608:	01069693          	slli	a3,a3,0x10
   1e60c:	02ba82b3          	mul	t0,s5,a1
   1e610:	013686b3          	add	a3,a3,s3
   1e614:	02bb05b3          	mul	a1,s6,a1
   1e618:	035f89b3          	mul	s3,t6,s5
   1e61c:	01358c33          	add	s8,a1,s3
   1e620:	0102d593          	srli	a1,t0,0x10
   1e624:	018585b3          	add	a1,a1,s8
   1e628:	036f8fb3          	mul	t6,t6,s6
   1e62c:	0135f463          	bgeu	a1,s3,1e634 <__multf3+0x6f8>
   1e630:	009f8fb3          	add	t6,t6,s1
   1e634:	0105d493          	srli	s1,a1,0x10
   1e638:	01f48fb3          	add	t6,s1,t6
   1e63c:	000104b7          	lui	s1,0x10
   1e640:	fff48993          	addi	s3,s1,-1 # ffff <exit-0xb5>
   1e644:	0135f5b3          	and	a1,a1,s3
   1e648:	0132f2b3          	and	t0,t0,s3
   1e64c:	01059593          	slli	a1,a1,0x10
   1e650:	032789b3          	mul	s3,a5,s2
   1e654:	005585b3          	add	a1,a1,t0
   1e658:	037787b3          	mul	a5,a5,s7
   1e65c:	03cb82b3          	mul	t0,s7,t3
   1e660:	03c90e33          	mul	t3,s2,t3
   1e664:	0102dc13          	srli	s8,t0,0x10
   1e668:	01c787b3          	add	a5,a5,t3
   1e66c:	00fc07b3          	add	a5,s8,a5
   1e670:	01c7f463          	bgeu	a5,t3,1e678 <__multf3+0x73c>
   1e674:	009989b3          	add	s3,s3,s1
   1e678:	00e50733          	add	a4,a0,a4
   1e67c:	01070833          	add	a6,a4,a6
   1e680:	01e88f33          	add	t5,a7,t5
   1e684:	00a73533          	sltu	a0,a4,a0
   1e688:	00af0533          	add	a0,t5,a0
   1e68c:	00d806b3          	add	a3,a6,a3
   1e690:	007503b3          	add	t2,a0,t2
   1e694:	00e83733          	sltu	a4,a6,a4
   1e698:	06d12023          	sw	a3,96(sp)
   1e69c:	0106b6b3          	sltu	a3,a3,a6
   1e6a0:	037a8833          	mul	a6,s5,s7
   1e6a4:	00e38733          	add	a4,t2,a4
   1e6a8:	01d70eb3          	add	t4,a4,t4
   1e6ac:	00de86b3          	add	a3,t4,a3
   1e6b0:	0107de13          	srli	t3,a5,0x10
   1e6b4:	011f38b3          	sltu	a7,t5,a7
   1e6b8:	000104b7          	lui	s1,0x10
   1e6bc:	01e53f33          	sltu	t5,a0,t5
   1e6c0:	00a3b533          	sltu	a0,t2,a0
   1e6c4:	007733b3          	sltu	t2,a4,t2
   1e6c8:	03590ab3          	mul	s5,s2,s5
   1e6cc:	013e0e33          	add	t3,t3,s3
   1e6d0:	00756533          	or	a0,a0,t2
   1e6d4:	fff48993          	addi	s3,s1,-1 # ffff <exit-0xb5>
   1e6d8:	00eeb733          	sltu	a4,t4,a4
   1e6dc:	01e8e8b3          	or	a7,a7,t5
   1e6e0:	01d6beb3          	sltu	t4,a3,t4
   1e6e4:	00a888b3          	add	a7,a7,a0
   1e6e8:	0137f7b3          	and	a5,a5,s3
   1e6ec:	01d76733          	or	a4,a4,t4
   1e6f0:	032b0933          	mul	s2,s6,s2
   1e6f4:	00b685b3          	add	a1,a3,a1
   1e6f8:	01170733          	add	a4,a4,a7
   1e6fc:	01079793          	slli	a5,a5,0x10
   1e700:	0132f2b3          	and	t0,t0,s3
   1e704:	01f70fb3          	add	t6,a4,t6
   1e708:	005787b3          	add	a5,a5,t0
   1e70c:	00d5b6b3          	sltu	a3,a1,a3
   1e710:	00df86b3          	add	a3,t6,a3
   1e714:	00f587b3          	add	a5,a1,a5
   1e718:	037b0b33          	mul	s6,s6,s7
   1e71c:	00efb733          	sltu	a4,t6,a4
   1e720:	01c68e33          	add	t3,a3,t3
   1e724:	01f6bfb3          	sltu	t6,a3,t6
   1e728:	06f12223          	sw	a5,100(sp)
   1e72c:	00b7b7b3          	sltu	a5,a5,a1
   1e730:	01f76533          	or	a0,a4,t6
   1e734:	00fe07b3          	add	a5,t3,a5
   1e738:	01085713          	srli	a4,a6,0x10
   1e73c:	00de36b3          	sltu	a3,t3,a3
   1e740:	015b0b33          	add	s6,s6,s5
   1e744:	01c7be33          	sltu	t3,a5,t3
   1e748:	01670733          	add	a4,a4,s6
   1e74c:	01c6e6b3          	or	a3,a3,t3
   1e750:	01577463          	bgeu	a4,s5,1e758 <__multf3+0x81c>
   1e754:	00990933          	add	s2,s2,s1
   1e758:	01075593          	srli	a1,a4,0x10
   1e75c:	00a585b3          	add	a1,a1,a0
   1e760:	00010537          	lui	a0,0x10
   1e764:	fff50513          	addi	a0,a0,-1 # ffff <exit-0xb5>
   1e768:	00a77733          	and	a4,a4,a0
   1e76c:	01071713          	slli	a4,a4,0x10
   1e770:	00a87833          	and	a6,a6,a0
   1e774:	01070733          	add	a4,a4,a6
   1e778:	00e78733          	add	a4,a5,a4
   1e77c:	00d586b3          	add	a3,a1,a3
   1e780:	00f737b3          	sltu	a5,a4,a5
   1e784:	00f687b3          	add	a5,a3,a5
   1e788:	012787b3          	add	a5,a5,s2
   1e78c:	06f12623          	sw	a5,108(sp)
   1e790:	01012783          	lw	a5,16(sp)
   1e794:	00d31313          	slli	t1,t1,0xd
   1e798:	06e12423          	sw	a4,104(sp)
   1e79c:	00c7e7b3          	or	a5,a5,a2
   1e7a0:	0147e7b3          	or	a5,a5,s4
   1e7a4:	00f36333          	or	t1,t1,a5
   1e7a8:	06010613          	addi	a2,sp,96
   1e7ac:	05010793          	addi	a5,sp,80
   1e7b0:	00c7a703          	lw	a4,12(a5)
   1e7b4:	0107a683          	lw	a3,16(a5)
   1e7b8:	00478793          	addi	a5,a5,4
   1e7bc:	01375713          	srli	a4,a4,0x13
   1e7c0:	00d69693          	slli	a3,a3,0xd
   1e7c4:	00d76733          	or	a4,a4,a3
   1e7c8:	fee7ae23          	sw	a4,-4(a5)
   1e7cc:	fef612e3          	bne	a2,a5,1e7b0 <__multf3+0x874>
   1e7d0:	05012783          	lw	a5,80(sp)
   1e7d4:	00603333          	snez	t1,t1
   1e7d8:	05c12703          	lw	a4,92(sp)
   1e7dc:	00f36333          	or	t1,t1,a5
   1e7e0:	05812783          	lw	a5,88(sp)
   1e7e4:	04e12623          	sw	a4,76(sp)
   1e7e8:	04612023          	sw	t1,64(sp)
   1e7ec:	04f12423          	sw	a5,72(sp)
   1e7f0:	05412783          	lw	a5,84(sp)
   1e7f4:	04f12223          	sw	a5,68(sp)
   1e7f8:	00b71793          	slli	a5,a4,0xb
   1e7fc:	0407d863          	bgez	a5,1e84c <__multf3+0x910>
   1e800:	01f31313          	slli	t1,t1,0x1f
   1e804:	04010793          	addi	a5,sp,64
   1e808:	04c10593          	addi	a1,sp,76
   1e80c:	0007a683          	lw	a3,0(a5)
   1e810:	0047a603          	lw	a2,4(a5)
   1e814:	00478793          	addi	a5,a5,4
   1e818:	0016d693          	srli	a3,a3,0x1
   1e81c:	01f61613          	slli	a2,a2,0x1f
   1e820:	00c6e6b3          	or	a3,a3,a2
   1e824:	fed7ae23          	sw	a3,-4(a5)
   1e828:	fef592e3          	bne	a1,a5,1e80c <__multf3+0x8d0>
   1e82c:	04012783          	lw	a5,64(sp)
   1e830:	00603333          	snez	t1,t1
   1e834:	00175713          	srli	a4,a4,0x1
   1e838:	0067e7b3          	or	a5,a5,t1
   1e83c:	04f12023          	sw	a5,64(sp)
   1e840:	00812783          	lw	a5,8(sp)
   1e844:	04e12623          	sw	a4,76(sp)
   1e848:	00f12623          	sw	a5,12(sp)
   1e84c:	00c12783          	lw	a5,12(sp)
   1e850:	00f12423          	sw	a5,8(sp)
   1e854:	3780006f          	j	1ebcc <__multf3+0xc90>
   1e858:	00a7e633          	or	a2,a5,a0
   1e85c:	00d66633          	or	a2,a2,a3
   1e860:	00e66633          	or	a2,a2,a4
   1e864:	14060463          	beqz	a2,1e9ac <__multf3+0xa70>
   1e868:	0a070063          	beqz	a4,1e908 <__multf3+0x9cc>
   1e86c:	00070513          	mv	a0,a4
   1e870:	044020ef          	jal	208b4 <__clzsi2>
   1e874:	ff450713          	addi	a4,a0,-12
   1e878:	40575593          	srai	a1,a4,0x5
   1e87c:	01f77713          	andi	a4,a4,31
   1e880:	0a070e63          	beqz	a4,1e93c <__multf3+0xa00>
   1e884:	ffc00693          	li	a3,-4
   1e888:	02d586b3          	mul	a3,a1,a3
   1e88c:	02000813          	li	a6,32
   1e890:	02010313          	addi	t1,sp,32
   1e894:	40e80833          	sub	a6,a6,a4
   1e898:	00c68793          	addi	a5,a3,12 # 1000c <exit-0xa8>
   1e89c:	00f307b3          	add	a5,t1,a5
   1e8a0:	40d006b3          	neg	a3,a3
   1e8a4:	0cf31463          	bne	t1,a5,1e96c <__multf3+0xa30>
   1e8a8:	fff58793          	addi	a5,a1,-1
   1e8ac:	00259593          	slli	a1,a1,0x2
   1e8b0:	05058693          	addi	a3,a1,80
   1e8b4:	02010613          	addi	a2,sp,32
   1e8b8:	00c685b3          	add	a1,a3,a2
   1e8bc:	02012683          	lw	a3,32(sp)
   1e8c0:	00e69733          	sll	a4,a3,a4
   1e8c4:	fae5a823          	sw	a4,-80(a1)
   1e8c8:	00178793          	addi	a5,a5,1
   1e8cc:	00279793          	slli	a5,a5,0x2
   1e8d0:	00800693          	li	a3,8
   1e8d4:	02010713          	addi	a4,sp,32
   1e8d8:	00d7ea63          	bltu	a5,a3,1e8ec <__multf3+0x9b0>
   1e8dc:	02012023          	sw	zero,32(sp)
   1e8e0:	00072223          	sw	zero,4(a4)
   1e8e4:	ff878793          	addi	a5,a5,-8
   1e8e8:	02810713          	addi	a4,sp,40
   1e8ec:	00400693          	li	a3,4
   1e8f0:	00d7e463          	bltu	a5,a3,1e8f8 <__multf3+0x9bc>
   1e8f4:	00072023          	sw	zero,0(a4)
   1e8f8:	ffffc4b7          	lui	s1,0xffffc
   1e8fc:	01148493          	addi	s1,s1,17 # ffffc011 <__BSS_END__+0xfffd92e1>
   1e900:	40a484b3          	sub	s1,s1,a0
   1e904:	f20ff06f          	j	1e024 <__multf3+0xe8>
   1e908:	00068a63          	beqz	a3,1e91c <__multf3+0x9e0>
   1e90c:	00068513          	mv	a0,a3
   1e910:	7a5010ef          	jal	208b4 <__clzsi2>
   1e914:	02050513          	addi	a0,a0,32
   1e918:	f5dff06f          	j	1e874 <__multf3+0x938>
   1e91c:	00050863          	beqz	a0,1e92c <__multf3+0x9f0>
   1e920:	795010ef          	jal	208b4 <__clzsi2>
   1e924:	04050513          	addi	a0,a0,64
   1e928:	f4dff06f          	j	1e874 <__multf3+0x938>
   1e92c:	00078513          	mv	a0,a5
   1e930:	785010ef          	jal	208b4 <__clzsi2>
   1e934:	06050513          	addi	a0,a0,96
   1e938:	f3dff06f          	j	1e874 <__multf3+0x938>
   1e93c:	ffc00693          	li	a3,-4
   1e940:	02d586b3          	mul	a3,a1,a3
   1e944:	02c10793          	addi	a5,sp,44
   1e948:	00300713          	li	a4,3
   1e94c:	00d78633          	add	a2,a5,a3
   1e950:	00062603          	lw	a2,0(a2)
   1e954:	fff70713          	addi	a4,a4,-1
   1e958:	ffc78793          	addi	a5,a5,-4
   1e95c:	00c7a223          	sw	a2,4(a5)
   1e960:	feb756e3          	bge	a4,a1,1e94c <__multf3+0xa10>
   1e964:	fff58793          	addi	a5,a1,-1
   1e968:	f61ff06f          	j	1e8c8 <__multf3+0x98c>
   1e96c:	0007a603          	lw	a2,0(a5)
   1e970:	ffc7a883          	lw	a7,-4(a5)
   1e974:	00d78e33          	add	t3,a5,a3
   1e978:	00e61633          	sll	a2,a2,a4
   1e97c:	0108d8b3          	srl	a7,a7,a6
   1e980:	01166633          	or	a2,a2,a7
   1e984:	00ce2023          	sw	a2,0(t3)
   1e988:	ffc78793          	addi	a5,a5,-4
   1e98c:	f19ff06f          	j	1e8a4 <__multf3+0x968>
   1e990:	00a7e7b3          	or	a5,a5,a0
   1e994:	00d7e7b3          	or	a5,a5,a3
   1e998:	00e7e7b3          	or	a5,a5,a4
   1e99c:	00200b93          	li	s7,2
   1e9a0:	e8078463          	beqz	a5,1e028 <__multf3+0xec>
   1e9a4:	00300b93          	li	s7,3
   1e9a8:	e80ff06f          	j	1e028 <__multf3+0xec>
   1e9ac:	00000493          	li	s1,0
   1e9b0:	00100b93          	li	s7,1
   1e9b4:	e74ff06f          	j	1e028 <__multf3+0xec>
   1e9b8:	016967b3          	or	a5,s2,s6
   1e9bc:	0157e7b3          	or	a5,a5,s5
   1e9c0:	00a7e7b3          	or	a5,a5,a0
   1e9c4:	14078463          	beqz	a5,1eb0c <__multf3+0xbd0>
   1e9c8:	08050e63          	beqz	a0,1ea64 <__multf3+0xb28>
   1e9cc:	6e9010ef          	jal	208b4 <__clzsi2>
   1e9d0:	ff450693          	addi	a3,a0,-12
   1e9d4:	4056d793          	srai	a5,a3,0x5
   1e9d8:	01f6f693          	andi	a3,a3,31
   1e9dc:	0c068063          	beqz	a3,1ea9c <__multf3+0xb60>
   1e9e0:	ffc00613          	li	a2,-4
   1e9e4:	02c78633          	mul	a2,a5,a2
   1e9e8:	02000813          	li	a6,32
   1e9ec:	03010313          	addi	t1,sp,48
   1e9f0:	40d80833          	sub	a6,a6,a3
   1e9f4:	00c60713          	addi	a4,a2,12
   1e9f8:	00e30733          	add	a4,t1,a4
   1e9fc:	40c00633          	neg	a2,a2
   1ea00:	0ce31663          	bne	t1,a4,1eacc <__multf3+0xb90>
   1ea04:	fff78713          	addi	a4,a5,-1
   1ea08:	00279793          	slli	a5,a5,0x2
   1ea0c:	02010613          	addi	a2,sp,32
   1ea10:	05078793          	addi	a5,a5,80
   1ea14:	00c787b3          	add	a5,a5,a2
   1ea18:	03012603          	lw	a2,48(sp)
   1ea1c:	00d616b3          	sll	a3,a2,a3
   1ea20:	fcd7a023          	sw	a3,-64(a5)
   1ea24:	00170793          	addi	a5,a4,1
   1ea28:	00279793          	slli	a5,a5,0x2
   1ea2c:	00800693          	li	a3,8
   1ea30:	03010713          	addi	a4,sp,48
   1ea34:	00d7ea63          	bltu	a5,a3,1ea48 <__multf3+0xb0c>
   1ea38:	02012823          	sw	zero,48(sp)
   1ea3c:	00072223          	sw	zero,4(a4)
   1ea40:	ff878793          	addi	a5,a5,-8
   1ea44:	03810713          	addi	a4,sp,56
   1ea48:	00400693          	li	a3,4
   1ea4c:	00d7e463          	bltu	a5,a3,1ea54 <__multf3+0xb18>
   1ea50:	00072023          	sw	zero,0(a4)
   1ea54:	ffffc7b7          	lui	a5,0xffffc
   1ea58:	01178793          	addi	a5,a5,17 # ffffc011 <__BSS_END__+0xfffd92e1>
   1ea5c:	40a787b3          	sub	a5,a5,a0
   1ea60:	e54ff06f          	j	1e0b4 <__multf3+0x178>
   1ea64:	000a8a63          	beqz	s5,1ea78 <__multf3+0xb3c>
   1ea68:	000a8513          	mv	a0,s5
   1ea6c:	649010ef          	jal	208b4 <__clzsi2>
   1ea70:	02050513          	addi	a0,a0,32
   1ea74:	f5dff06f          	j	1e9d0 <__multf3+0xa94>
   1ea78:	000b0a63          	beqz	s6,1ea8c <__multf3+0xb50>
   1ea7c:	000b0513          	mv	a0,s6
   1ea80:	635010ef          	jal	208b4 <__clzsi2>
   1ea84:	04050513          	addi	a0,a0,64
   1ea88:	f49ff06f          	j	1e9d0 <__multf3+0xa94>
   1ea8c:	00090513          	mv	a0,s2
   1ea90:	625010ef          	jal	208b4 <__clzsi2>
   1ea94:	06050513          	addi	a0,a0,96
   1ea98:	f39ff06f          	j	1e9d0 <__multf3+0xa94>
   1ea9c:	ffc00613          	li	a2,-4
   1eaa0:	02c78633          	mul	a2,a5,a2
   1eaa4:	03c10713          	addi	a4,sp,60
   1eaa8:	00300693          	li	a3,3
   1eaac:	00c705b3          	add	a1,a4,a2
   1eab0:	0005a583          	lw	a1,0(a1)
   1eab4:	fff68693          	addi	a3,a3,-1
   1eab8:	ffc70713          	addi	a4,a4,-4
   1eabc:	00b72223          	sw	a1,4(a4)
   1eac0:	fef6d6e3          	bge	a3,a5,1eaac <__multf3+0xb70>
   1eac4:	fff78713          	addi	a4,a5,-1
   1eac8:	f5dff06f          	j	1ea24 <__multf3+0xae8>
   1eacc:	00072583          	lw	a1,0(a4)
   1ead0:	ffc72883          	lw	a7,-4(a4)
   1ead4:	00c70e33          	add	t3,a4,a2
   1ead8:	00d595b3          	sll	a1,a1,a3
   1eadc:	0108d8b3          	srl	a7,a7,a6
   1eae0:	0115e5b3          	or	a1,a1,a7
   1eae4:	00be2023          	sw	a1,0(t3)
   1eae8:	ffc70713          	addi	a4,a4,-4
   1eaec:	f15ff06f          	j	1ea00 <__multf3+0xac4>
   1eaf0:	01696933          	or	s2,s2,s6
   1eaf4:	01596933          	or	s2,s2,s5
   1eaf8:	00a96933          	or	s2,s2,a0
   1eafc:	00200693          	li	a3,2
   1eb00:	da090c63          	beqz	s2,1e0b8 <__multf3+0x17c>
   1eb04:	00300693          	li	a3,3
   1eb08:	db0ff06f          	j	1e0b8 <__multf3+0x17c>
   1eb0c:	00000793          	li	a5,0
   1eb10:	00100693          	li	a3,1
   1eb14:	da4ff06f          	j	1e0b8 <__multf3+0x17c>
   1eb18:	00100713          	li	a4,1
   1eb1c:	00f717b3          	sll	a5,a4,a5
   1eb20:	5307f713          	andi	a4,a5,1328
   1eb24:	06071863          	bnez	a4,1eb94 <__multf3+0xc58>
   1eb28:	0887f713          	andi	a4,a5,136
   1eb2c:	04071063          	bnez	a4,1eb6c <__multf3+0xc30>
   1eb30:	2407f793          	andi	a5,a5,576
   1eb34:	dc078063          	beqz	a5,1e0f4 <__multf3+0x1b8>
   1eb38:	000087b7          	lui	a5,0x8
   1eb3c:	04f12623          	sw	a5,76(sp)
   1eb40:	04012423          	sw	zero,72(sp)
   1eb44:	04012223          	sw	zero,68(sp)
   1eb48:	04012023          	sw	zero,64(sp)
   1eb4c:	fff78793          	addi	a5,a5,-1 # 7fff <exit-0x80b5>
   1eb50:	00012223          	sw	zero,4(sp)
   1eb54:	1440006f          	j	1ec98 <__multf3+0xd5c>
   1eb58:	00f00713          	li	a4,15
   1eb5c:	fce78ee3          	beq	a5,a4,1eb38 <__multf3+0xbfc>
   1eb60:	00b00713          	li	a4,11
   1eb64:	01412223          	sw	s4,4(sp)
   1eb68:	02e79663          	bne	a5,a4,1eb94 <__multf3+0xc58>
   1eb6c:	01312223          	sw	s3,4(sp)
   1eb70:	03012783          	lw	a5,48(sp)
   1eb74:	00068b93          	mv	s7,a3
   1eb78:	04f12023          	sw	a5,64(sp)
   1eb7c:	03412783          	lw	a5,52(sp)
   1eb80:	04f12223          	sw	a5,68(sp)
   1eb84:	03812783          	lw	a5,56(sp)
   1eb88:	04f12423          	sw	a5,72(sp)
   1eb8c:	03c12783          	lw	a5,60(sp)
   1eb90:	0200006f          	j	1ebb0 <__multf3+0xc74>
   1eb94:	02012783          	lw	a5,32(sp)
   1eb98:	04f12023          	sw	a5,64(sp)
   1eb9c:	02412783          	lw	a5,36(sp)
   1eba0:	04f12223          	sw	a5,68(sp)
   1eba4:	02812783          	lw	a5,40(sp)
   1eba8:	04f12423          	sw	a5,72(sp)
   1ebac:	02c12783          	lw	a5,44(sp)
   1ebb0:	04f12623          	sw	a5,76(sp)
   1ebb4:	00200793          	li	a5,2
   1ebb8:	36fb8663          	beq	s7,a5,1ef24 <__multf3+0xfe8>
   1ebbc:	00300793          	li	a5,3
   1ebc0:	f6fb8ce3          	beq	s7,a5,1eb38 <__multf3+0xbfc>
   1ebc4:	00100793          	li	a5,1
   1ebc8:	34fb8463          	beq	s7,a5,1ef10 <__multf3+0xfd4>
   1ebcc:	00812703          	lw	a4,8(sp)
   1ebd0:	000047b7          	lui	a5,0x4
   1ebd4:	fff78793          	addi	a5,a5,-1 # 3fff <exit-0xc0b5>
   1ebd8:	00f707b3          	add	a5,a4,a5
   1ebdc:	12f05a63          	blez	a5,1ed10 <__multf3+0xdd4>
   1ebe0:	04012703          	lw	a4,64(sp)
   1ebe4:	00777693          	andi	a3,a4,7
   1ebe8:	04068463          	beqz	a3,1ec30 <__multf3+0xcf4>
   1ebec:	00f77693          	andi	a3,a4,15
   1ebf0:	00400613          	li	a2,4
   1ebf4:	02c68e63          	beq	a3,a2,1ec30 <__multf3+0xcf4>
   1ebf8:	00470713          	addi	a4,a4,4
   1ebfc:	00473693          	sltiu	a3,a4,4
   1ec00:	04e12023          	sw	a4,64(sp)
   1ec04:	04412703          	lw	a4,68(sp)
   1ec08:	00e68733          	add	a4,a3,a4
   1ec0c:	04e12223          	sw	a4,68(sp)
   1ec10:	00d73733          	sltu	a4,a4,a3
   1ec14:	04812683          	lw	a3,72(sp)
   1ec18:	00e68733          	add	a4,a3,a4
   1ec1c:	04e12423          	sw	a4,72(sp)
   1ec20:	00d73733          	sltu	a4,a4,a3
   1ec24:	04c12683          	lw	a3,76(sp)
   1ec28:	00d70733          	add	a4,a4,a3
   1ec2c:	04e12623          	sw	a4,76(sp)
   1ec30:	04c12703          	lw	a4,76(sp)
   1ec34:	00b71693          	slli	a3,a4,0xb
   1ec38:	0206d063          	bgez	a3,1ec58 <__multf3+0xd1c>
   1ec3c:	fff007b7          	lui	a5,0xfff00
   1ec40:	fff78793          	addi	a5,a5,-1 # ffefffff <__BSS_END__+0xffedd2cf>
   1ec44:	00f77733          	and	a4,a4,a5
   1ec48:	04e12623          	sw	a4,76(sp)
   1ec4c:	00812703          	lw	a4,8(sp)
   1ec50:	000047b7          	lui	a5,0x4
   1ec54:	00f707b3          	add	a5,a4,a5
   1ec58:	04010713          	addi	a4,sp,64
   1ec5c:	04c10593          	addi	a1,sp,76
   1ec60:	00072683          	lw	a3,0(a4)
   1ec64:	00472603          	lw	a2,4(a4)
   1ec68:	00470713          	addi	a4,a4,4
   1ec6c:	0036d693          	srli	a3,a3,0x3
   1ec70:	01d61613          	slli	a2,a2,0x1d
   1ec74:	00c6e6b3          	or	a3,a3,a2
   1ec78:	fed72e23          	sw	a3,-4(a4)
   1ec7c:	fee592e3          	bne	a1,a4,1ec60 <__multf3+0xd24>
   1ec80:	000086b7          	lui	a3,0x8
   1ec84:	ffe68693          	addi	a3,a3,-2 # 7ffe <exit-0x80b6>
   1ec88:	04c12703          	lw	a4,76(sp)
   1ec8c:	28f6cc63          	blt	a3,a5,1ef24 <__multf3+0xfe8>
   1ec90:	00375713          	srli	a4,a4,0x3
   1ec94:	04e12623          	sw	a4,76(sp)
   1ec98:	04c12703          	lw	a4,76(sp)
   1ec9c:	0ac12083          	lw	ra,172(sp)
   1eca0:	00040513          	mv	a0,s0
   1eca4:	04e11e23          	sh	a4,92(sp)
   1eca8:	00412703          	lw	a4,4(sp)
   1ecac:	0a412483          	lw	s1,164(sp)
   1ecb0:	0a012903          	lw	s2,160(sp)
   1ecb4:	00f71713          	slli	a4,a4,0xf
   1ecb8:	00f767b3          	or	a5,a4,a5
   1ecbc:	04f11f23          	sh	a5,94(sp)
   1ecc0:	04012783          	lw	a5,64(sp)
   1ecc4:	09c12983          	lw	s3,156(sp)
   1ecc8:	09812a03          	lw	s4,152(sp)
   1eccc:	00f42023          	sw	a5,0(s0)
   1ecd0:	04412783          	lw	a5,68(sp)
   1ecd4:	09412a83          	lw	s5,148(sp)
   1ecd8:	09012b03          	lw	s6,144(sp)
   1ecdc:	00f42223          	sw	a5,4(s0)
   1ece0:	04812783          	lw	a5,72(sp)
   1ece4:	08c12b83          	lw	s7,140(sp)
   1ece8:	08812c03          	lw	s8,136(sp)
   1ecec:	00f42423          	sw	a5,8(s0)
   1ecf0:	05c12783          	lw	a5,92(sp)
   1ecf4:	08412c83          	lw	s9,132(sp)
   1ecf8:	08012d03          	lw	s10,128(sp)
   1ecfc:	00f42623          	sw	a5,12(s0)
   1ed00:	0a812403          	lw	s0,168(sp)
   1ed04:	07c12d83          	lw	s11,124(sp)
   1ed08:	0b010113          	addi	sp,sp,176
   1ed0c:	00008067          	ret
   1ed10:	00100693          	li	a3,1
   1ed14:	40f686b3          	sub	a3,a3,a5
   1ed18:	07400793          	li	a5,116
   1ed1c:	1cd7ca63          	blt	a5,a3,1eef0 <__multf3+0xfb4>
   1ed20:	04010613          	addi	a2,sp,64
   1ed24:	4056d713          	srai	a4,a3,0x5
   1ed28:	00060513          	mv	a0,a2
   1ed2c:	01f6f693          	andi	a3,a3,31
   1ed30:	00000793          	li	a5,0
   1ed34:	00000593          	li	a1,0
   1ed38:	02e59e63          	bne	a1,a4,1ed74 <__multf3+0xe38>
   1ed3c:	00300593          	li	a1,3
   1ed40:	40e585b3          	sub	a1,a1,a4
   1ed44:	00271513          	slli	a0,a4,0x2
   1ed48:	04069063          	bnez	a3,1ed88 <__multf3+0xe4c>
   1ed4c:	00060813          	mv	a6,a2
   1ed50:	00a808b3          	add	a7,a6,a0
   1ed54:	0008a883          	lw	a7,0(a7) # 10000 <exit-0xb4>
   1ed58:	00168693          	addi	a3,a3,1
   1ed5c:	00480813          	addi	a6,a6,4
   1ed60:	ff182e23          	sw	a7,-4(a6)
   1ed64:	fed5d6e3          	bge	a1,a3,1ed50 <__multf3+0xe14>
   1ed68:	00400693          	li	a3,4
   1ed6c:	40e68733          	sub	a4,a3,a4
   1ed70:	06c0006f          	j	1eddc <__multf3+0xea0>
   1ed74:	00052803          	lw	a6,0(a0)
   1ed78:	00158593          	addi	a1,a1,1
   1ed7c:	00450513          	addi	a0,a0,4
   1ed80:	0107e7b3          	or	a5,a5,a6
   1ed84:	fb5ff06f          	j	1ed38 <__multf3+0xdfc>
   1ed88:	05050813          	addi	a6,a0,80
   1ed8c:	02010893          	addi	a7,sp,32
   1ed90:	01180833          	add	a6,a6,a7
   1ed94:	fd082803          	lw	a6,-48(a6)
   1ed98:	02000313          	li	t1,32
   1ed9c:	40d30333          	sub	t1,t1,a3
   1eda0:	00681833          	sll	a6,a6,t1
   1eda4:	0107e7b3          	or	a5,a5,a6
   1eda8:	00000e13          	li	t3,0
   1edac:	00a60833          	add	a6,a2,a0
   1edb0:	40a00533          	neg	a0,a0
   1edb4:	0ebe4063          	blt	t3,a1,1ee94 <__multf3+0xf58>
   1edb8:	00400513          	li	a0,4
   1edbc:	00259593          	slli	a1,a1,0x2
   1edc0:	40e50733          	sub	a4,a0,a4
   1edc4:	05058593          	addi	a1,a1,80
   1edc8:	02010513          	addi	a0,sp,32
   1edcc:	00a585b3          	add	a1,a1,a0
   1edd0:	04c12503          	lw	a0,76(sp)
   1edd4:	00d556b3          	srl	a3,a0,a3
   1edd8:	fcd5a823          	sw	a3,-48(a1)
   1eddc:	00400693          	li	a3,4
   1ede0:	40e686b3          	sub	a3,a3,a4
   1ede4:	00269693          	slli	a3,a3,0x2
   1ede8:	00271713          	slli	a4,a4,0x2
   1edec:	00800593          	li	a1,8
   1edf0:	00e60733          	add	a4,a2,a4
   1edf4:	00b6ea63          	bltu	a3,a1,1ee08 <__multf3+0xecc>
   1edf8:	00072023          	sw	zero,0(a4)
   1edfc:	00072223          	sw	zero,4(a4)
   1ee00:	ff868693          	addi	a3,a3,-8
   1ee04:	00870713          	addi	a4,a4,8
   1ee08:	00400593          	li	a1,4
   1ee0c:	00b6e463          	bltu	a3,a1,1ee14 <__multf3+0xed8>
   1ee10:	00072023          	sw	zero,0(a4)
   1ee14:	04012703          	lw	a4,64(sp)
   1ee18:	00f037b3          	snez	a5,a5
   1ee1c:	00e7e7b3          	or	a5,a5,a4
   1ee20:	04f12023          	sw	a5,64(sp)
   1ee24:	0077f713          	andi	a4,a5,7
   1ee28:	04070463          	beqz	a4,1ee70 <__multf3+0xf34>
   1ee2c:	00f7f713          	andi	a4,a5,15
   1ee30:	00400693          	li	a3,4
   1ee34:	02d70e63          	beq	a4,a3,1ee70 <__multf3+0xf34>
   1ee38:	04412703          	lw	a4,68(sp)
   1ee3c:	00478793          	addi	a5,a5,4 # 4004 <exit-0xc0b0>
   1ee40:	04f12023          	sw	a5,64(sp)
   1ee44:	0047b793          	sltiu	a5,a5,4
   1ee48:	00f707b3          	add	a5,a4,a5
   1ee4c:	04f12223          	sw	a5,68(sp)
   1ee50:	00e7b7b3          	sltu	a5,a5,a4
   1ee54:	04812703          	lw	a4,72(sp)
   1ee58:	00f707b3          	add	a5,a4,a5
   1ee5c:	04f12423          	sw	a5,72(sp)
   1ee60:	00e7b7b3          	sltu	a5,a5,a4
   1ee64:	04c12703          	lw	a4,76(sp)
   1ee68:	00e787b3          	add	a5,a5,a4
   1ee6c:	04f12623          	sw	a5,76(sp)
   1ee70:	04c12703          	lw	a4,76(sp)
   1ee74:	00c71793          	slli	a5,a4,0xc
   1ee78:	0407d263          	bgez	a5,1eebc <__multf3+0xf80>
   1ee7c:	04012623          	sw	zero,76(sp)
   1ee80:	04012423          	sw	zero,72(sp)
   1ee84:	04012223          	sw	zero,68(sp)
   1ee88:	04012023          	sw	zero,64(sp)
   1ee8c:	00100793          	li	a5,1
   1ee90:	e09ff06f          	j	1ec98 <__multf3+0xd5c>
   1ee94:	00082883          	lw	a7,0(a6)
   1ee98:	00482e83          	lw	t4,4(a6)
   1ee9c:	00a80f33          	add	t5,a6,a0
   1eea0:	00d8d8b3          	srl	a7,a7,a3
   1eea4:	006e9eb3          	sll	t4,t4,t1
   1eea8:	01d8e8b3          	or	a7,a7,t4
   1eeac:	011f2023          	sw	a7,0(t5)
   1eeb0:	001e0e13          	addi	t3,t3,1
   1eeb4:	00480813          	addi	a6,a6,4
   1eeb8:	efdff06f          	j	1edb4 <__multf3+0xe78>
   1eebc:	00c60593          	addi	a1,a2,12
   1eec0:	00062783          	lw	a5,0(a2)
   1eec4:	00462683          	lw	a3,4(a2)
   1eec8:	00460613          	addi	a2,a2,4
   1eecc:	0037d793          	srli	a5,a5,0x3
   1eed0:	01d69693          	slli	a3,a3,0x1d
   1eed4:	00d7e7b3          	or	a5,a5,a3
   1eed8:	fef62e23          	sw	a5,-4(a2)
   1eedc:	fec592e3          	bne	a1,a2,1eec0 <__multf3+0xf84>
   1eee0:	00375713          	srli	a4,a4,0x3
   1eee4:	04e12623          	sw	a4,76(sp)
   1eee8:	00000793          	li	a5,0
   1eeec:	dadff06f          	j	1ec98 <__multf3+0xd5c>
   1eef0:	04412703          	lw	a4,68(sp)
   1eef4:	04012783          	lw	a5,64(sp)
   1eef8:	00e7e7b3          	or	a5,a5,a4
   1eefc:	04812703          	lw	a4,72(sp)
   1ef00:	00e7e7b3          	or	a5,a5,a4
   1ef04:	04c12703          	lw	a4,76(sp)
   1ef08:	00e7e7b3          	or	a5,a5,a4
   1ef0c:	fc078ee3          	beqz	a5,1eee8 <__multf3+0xfac>
   1ef10:	04012623          	sw	zero,76(sp)
   1ef14:	04012423          	sw	zero,72(sp)
   1ef18:	04012223          	sw	zero,68(sp)
   1ef1c:	04012023          	sw	zero,64(sp)
   1ef20:	fc9ff06f          	j	1eee8 <__multf3+0xfac>
   1ef24:	000087b7          	lui	a5,0x8
   1ef28:	04012623          	sw	zero,76(sp)
   1ef2c:	04012423          	sw	zero,72(sp)
   1ef30:	04012223          	sw	zero,68(sp)
   1ef34:	04012023          	sw	zero,64(sp)
   1ef38:	fff78793          	addi	a5,a5,-1 # 7fff <exit-0x80b5>
   1ef3c:	d5dff06f          	j	1ec98 <__multf3+0xd5c>

0001ef40 <__subtf3>:
   1ef40:	f9010113          	addi	sp,sp,-112
   1ef44:	0085a703          	lw	a4,8(a1)
   1ef48:	05512a23          	sw	s5,84(sp)
   1ef4c:	00c5aa83          	lw	s5,12(a1)
   1ef50:	0005a783          	lw	a5,0(a1)
   1ef54:	0045a683          	lw	a3,4(a1)
   1ef58:	02e12c23          	sw	a4,56(sp)
   1ef5c:	00e12c23          	sw	a4,24(sp)
   1ef60:	010a9713          	slli	a4,s5,0x10
   1ef64:	06912223          	sw	s1,100(sp)
   1ef68:	01075713          	srli	a4,a4,0x10
   1ef6c:	001a9493          	slli	s1,s5,0x1
   1ef70:	00462803          	lw	a6,4(a2)
   1ef74:	00862583          	lw	a1,8(a2)
   1ef78:	06812423          	sw	s0,104(sp)
   1ef7c:	07212023          	sw	s2,96(sp)
   1ef80:	00062403          	lw	s0,0(a2)
   1ef84:	00c62903          	lw	s2,12(a2)
   1ef88:	05412c23          	sw	s4,88(sp)
   1ef8c:	03512e23          	sw	s5,60(sp)
   1ef90:	00050a13          	mv	s4,a0
   1ef94:	06112623          	sw	ra,108(sp)
   1ef98:	05312e23          	sw	s3,92(sp)
   1ef9c:	05612823          	sw	s6,80(sp)
   1efa0:	05712623          	sw	s7,76(sp)
   1efa4:	05812423          	sw	s8,72(sp)
   1efa8:	02f12823          	sw	a5,48(sp)
   1efac:	02d12a23          	sw	a3,52(sp)
   1efb0:	00f12823          	sw	a5,16(sp)
   1efb4:	00d12a23          	sw	a3,20(sp)
   1efb8:	00e12e23          	sw	a4,28(sp)
   1efbc:	0114d493          	srli	s1,s1,0x11
   1efc0:	01fada93          	srli	s5,s5,0x1f
   1efc4:	01010513          	addi	a0,sp,16
   1efc8:	01c10613          	addi	a2,sp,28
   1efcc:	00062703          	lw	a4,0(a2)
   1efd0:	ffc62683          	lw	a3,-4(a2)
   1efd4:	ffc60613          	addi	a2,a2,-4
   1efd8:	00371713          	slli	a4,a4,0x3
   1efdc:	01d6d693          	srli	a3,a3,0x1d
   1efe0:	00d76733          	or	a4,a4,a3
   1efe4:	00e62223          	sw	a4,4(a2)
   1efe8:	fec512e3          	bne	a0,a2,1efcc <__subtf3+0x8c>
   1efec:	01091713          	slli	a4,s2,0x10
   1eff0:	00191b93          	slli	s7,s2,0x1
   1eff4:	00379793          	slli	a5,a5,0x3
   1eff8:	01075713          	srli	a4,a4,0x10
   1effc:	03012a23          	sw	a6,52(sp)
   1f000:	03212e23          	sw	s2,60(sp)
   1f004:	03012223          	sw	a6,36(sp)
   1f008:	00f12823          	sw	a5,16(sp)
   1f00c:	02812823          	sw	s0,48(sp)
   1f010:	02b12c23          	sw	a1,56(sp)
   1f014:	02812023          	sw	s0,32(sp)
   1f018:	02b12423          	sw	a1,40(sp)
   1f01c:	02e12623          	sw	a4,44(sp)
   1f020:	011bdb93          	srli	s7,s7,0x11
   1f024:	01f95913          	srli	s2,s2,0x1f
   1f028:	02010813          	addi	a6,sp,32
   1f02c:	02c10313          	addi	t1,sp,44
   1f030:	00032703          	lw	a4,0(t1)
   1f034:	ffc32683          	lw	a3,-4(t1)
   1f038:	ffc30313          	addi	t1,t1,-4
   1f03c:	00371713          	slli	a4,a4,0x3
   1f040:	01d6d693          	srli	a3,a3,0x1d
   1f044:	00d76733          	or	a4,a4,a3
   1f048:	00e32223          	sw	a4,4(t1)
   1f04c:	fe6812e3          	bne	a6,t1,1f030 <__subtf3+0xf0>
   1f050:	00341413          	slli	s0,s0,0x3
   1f054:	00008737          	lui	a4,0x8
   1f058:	02812023          	sw	s0,32(sp)
   1f05c:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1f060:	02eb9063          	bne	s7,a4,1f080 <__subtf3+0x140>
   1f064:	02812683          	lw	a3,40(sp)
   1f068:	02412703          	lw	a4,36(sp)
   1f06c:	00d76733          	or	a4,a4,a3
   1f070:	02c12683          	lw	a3,44(sp)
   1f074:	00d76733          	or	a4,a4,a3
   1f078:	00876733          	or	a4,a4,s0
   1f07c:	00071463          	bnez	a4,1f084 <__subtf3+0x144>
   1f080:	00194913          	xori	s2,s2,1
   1f084:	417488b3          	sub	a7,s1,s7
   1f088:	095916e3          	bne	s2,s5,1f914 <__subtf3+0x9d4>
   1f08c:	45105263          	blez	a7,1f4d0 <__subtf3+0x590>
   1f090:	01412903          	lw	s2,20(sp)
   1f094:	01812983          	lw	s3,24(sp)
   1f098:	01c12b03          	lw	s6,28(sp)
   1f09c:	0a0b9263          	bnez	s7,1f140 <__subtf3+0x200>
   1f0a0:	02412683          	lw	a3,36(sp)
   1f0a4:	02812703          	lw	a4,40(sp)
   1f0a8:	02c12583          	lw	a1,44(sp)
   1f0ac:	00e6e633          	or	a2,a3,a4
   1f0b0:	00b66633          	or	a2,a2,a1
   1f0b4:	00866633          	or	a2,a2,s0
   1f0b8:	00061e63          	bnez	a2,1f0d4 <__subtf3+0x194>
   1f0bc:	02f12823          	sw	a5,48(sp)
   1f0c0:	03212a23          	sw	s2,52(sp)
   1f0c4:	03312c23          	sw	s3,56(sp)
   1f0c8:	03612e23          	sw	s6,60(sp)
   1f0cc:	00088493          	mv	s1,a7
   1f0d0:	08c0006f          	j	1f15c <__subtf3+0x21c>
   1f0d4:	fff88613          	addi	a2,a7,-1
   1f0d8:	04061863          	bnez	a2,1f128 <__subtf3+0x1e8>
   1f0dc:	00878433          	add	s0,a5,s0
   1f0e0:	01268933          	add	s2,a3,s2
   1f0e4:	02812823          	sw	s0,48(sp)
   1f0e8:	00f43433          	sltu	s0,s0,a5
   1f0ec:	00890433          	add	s0,s2,s0
   1f0f0:	02812a23          	sw	s0,52(sp)
   1f0f4:	00d936b3          	sltu	a3,s2,a3
   1f0f8:	01243433          	sltu	s0,s0,s2
   1f0fc:	013709b3          	add	s3,a4,s3
   1f100:	0086e6b3          	or	a3,a3,s0
   1f104:	00d986b3          	add	a3,s3,a3
   1f108:	02d12c23          	sw	a3,56(sp)
   1f10c:	00e9b7b3          	sltu	a5,s3,a4
   1f110:	0136b6b3          	sltu	a3,a3,s3
   1f114:	00d7e7b3          	or	a5,a5,a3
   1f118:	016585b3          	add	a1,a1,s6
   1f11c:	00b787b3          	add	a5,a5,a1
   1f120:	00100493          	li	s1,1
   1f124:	2fc0006f          	j	1f420 <__subtf3+0x4e0>
   1f128:	00008737          	lui	a4,0x8
   1f12c:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1f130:	00e88463          	beq	a7,a4,1f138 <__subtf3+0x1f8>
   1f134:	2500106f          	j	20384 <__subtf3+0x1444>
   1f138:	02f12823          	sw	a5,48(sp)
   1f13c:	4400006f          	j	1f57c <__subtf3+0x63c>
   1f140:	00008737          	lui	a4,0x8
   1f144:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1f148:	16e49a63          	bne	s1,a4,1f2bc <__subtf3+0x37c>
   1f14c:	02f12823          	sw	a5,48(sp)
   1f150:	03212a23          	sw	s2,52(sp)
   1f154:	03312c23          	sw	s3,56(sp)
   1f158:	03612e23          	sw	s6,60(sp)
   1f15c:	03012783          	lw	a5,48(sp)
   1f160:	0077f713          	andi	a4,a5,7
   1f164:	04070463          	beqz	a4,1f1ac <__subtf3+0x26c>
   1f168:	00f7f713          	andi	a4,a5,15
   1f16c:	00400693          	li	a3,4
   1f170:	02d70e63          	beq	a4,a3,1f1ac <__subtf3+0x26c>
   1f174:	03412703          	lw	a4,52(sp)
   1f178:	00478793          	addi	a5,a5,4
   1f17c:	02f12823          	sw	a5,48(sp)
   1f180:	0047b793          	sltiu	a5,a5,4
   1f184:	00f707b3          	add	a5,a4,a5
   1f188:	02f12a23          	sw	a5,52(sp)
   1f18c:	00e7b7b3          	sltu	a5,a5,a4
   1f190:	03812703          	lw	a4,56(sp)
   1f194:	00f707b3          	add	a5,a4,a5
   1f198:	02f12c23          	sw	a5,56(sp)
   1f19c:	00e7b7b3          	sltu	a5,a5,a4
   1f1a0:	03c12703          	lw	a4,60(sp)
   1f1a4:	00e787b3          	add	a5,a5,a4
   1f1a8:	02f12e23          	sw	a5,60(sp)
   1f1ac:	03c12783          	lw	a5,60(sp)
   1f1b0:	00c79713          	slli	a4,a5,0xc
   1f1b4:	02075463          	bgez	a4,1f1dc <__subtf3+0x29c>
   1f1b8:	00008737          	lui	a4,0x8
   1f1bc:	00148493          	addi	s1,s1,1
   1f1c0:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1f1c4:	00e49463          	bne	s1,a4,1f1cc <__subtf3+0x28c>
   1f1c8:	1a80106f          	j	20370 <__subtf3+0x1430>
   1f1cc:	fff80737          	lui	a4,0xfff80
   1f1d0:	fff70713          	addi	a4,a4,-1 # fff7ffff <__BSS_END__+0xfff5d2cf>
   1f1d4:	00e7f7b3          	and	a5,a5,a4
   1f1d8:	02f12e23          	sw	a5,60(sp)
   1f1dc:	03010793          	addi	a5,sp,48
   1f1e0:	03c10613          	addi	a2,sp,60
   1f1e4:	0007a703          	lw	a4,0(a5)
   1f1e8:	0047a683          	lw	a3,4(a5)
   1f1ec:	00478793          	addi	a5,a5,4
   1f1f0:	00375713          	srli	a4,a4,0x3
   1f1f4:	01d69693          	slli	a3,a3,0x1d
   1f1f8:	00d76733          	or	a4,a4,a3
   1f1fc:	fee7ae23          	sw	a4,-4(a5)
   1f200:	fec792e3          	bne	a5,a2,1f1e4 <__subtf3+0x2a4>
   1f204:	03c12703          	lw	a4,60(sp)
   1f208:	000086b7          	lui	a3,0x8
   1f20c:	fff68793          	addi	a5,a3,-1 # 7fff <exit-0x80b5>
   1f210:	00375713          	srli	a4,a4,0x3
   1f214:	02e12e23          	sw	a4,60(sp)
   1f218:	02f49a63          	bne	s1,a5,1f24c <__subtf3+0x30c>
   1f21c:	03412603          	lw	a2,52(sp)
   1f220:	03012783          	lw	a5,48(sp)
   1f224:	00c7e7b3          	or	a5,a5,a2
   1f228:	03812603          	lw	a2,56(sp)
   1f22c:	00c7e7b3          	or	a5,a5,a2
   1f230:	00e7e7b3          	or	a5,a5,a4
   1f234:	00078c63          	beqz	a5,1f24c <__subtf3+0x30c>
   1f238:	02d12e23          	sw	a3,60(sp)
   1f23c:	02012c23          	sw	zero,56(sp)
   1f240:	02012a23          	sw	zero,52(sp)
   1f244:	02012823          	sw	zero,48(sp)
   1f248:	00000a93          	li	s5,0
   1f24c:	03c12783          	lw	a5,60(sp)
   1f250:	01149493          	slli	s1,s1,0x11
   1f254:	0114d493          	srli	s1,s1,0x11
   1f258:	00f11623          	sh	a5,12(sp)
   1f25c:	03012783          	lw	a5,48(sp)
   1f260:	00fa9a93          	slli	s5,s5,0xf
   1f264:	009aeab3          	or	s5,s5,s1
   1f268:	00fa2023          	sw	a5,0(s4)
   1f26c:	03412783          	lw	a5,52(sp)
   1f270:	01511723          	sh	s5,14(sp)
   1f274:	06c12083          	lw	ra,108(sp)
   1f278:	00fa2223          	sw	a5,4(s4)
   1f27c:	03812783          	lw	a5,56(sp)
   1f280:	06812403          	lw	s0,104(sp)
   1f284:	06412483          	lw	s1,100(sp)
   1f288:	00fa2423          	sw	a5,8(s4)
   1f28c:	00c12783          	lw	a5,12(sp)
   1f290:	06012903          	lw	s2,96(sp)
   1f294:	05c12983          	lw	s3,92(sp)
   1f298:	00fa2623          	sw	a5,12(s4)
   1f29c:	05412a83          	lw	s5,84(sp)
   1f2a0:	05012b03          	lw	s6,80(sp)
   1f2a4:	04c12b83          	lw	s7,76(sp)
   1f2a8:	04812c03          	lw	s8,72(sp)
   1f2ac:	000a0513          	mv	a0,s4
   1f2b0:	05812a03          	lw	s4,88(sp)
   1f2b4:	07010113          	addi	sp,sp,112
   1f2b8:	00008067          	ret
   1f2bc:	02c12703          	lw	a4,44(sp)
   1f2c0:	000806b7          	lui	a3,0x80
   1f2c4:	00d76733          	or	a4,a4,a3
   1f2c8:	02e12623          	sw	a4,44(sp)
   1f2cc:	07400713          	li	a4,116
   1f2d0:	01175463          	bge	a4,a7,1f2d8 <__subtf3+0x398>
   1f2d4:	0bc0106f          	j	20390 <__subtf3+0x1450>
   1f2d8:	00088613          	mv	a2,a7
   1f2dc:	40565693          	srai	a3,a2,0x5
   1f2e0:	00030513          	mv	a0,t1
   1f2e4:	01f67613          	andi	a2,a2,31
   1f2e8:	00000713          	li	a4,0
   1f2ec:	00000593          	li	a1,0
   1f2f0:	02d59c63          	bne	a1,a3,1f328 <__subtf3+0x3e8>
   1f2f4:	00300593          	li	a1,3
   1f2f8:	40d585b3          	sub	a1,a1,a3
   1f2fc:	00269513          	slli	a0,a3,0x2
   1f300:	02061e63          	bnez	a2,1f33c <__subtf3+0x3fc>
   1f304:	00a308b3          	add	a7,t1,a0
   1f308:	0008a883          	lw	a7,0(a7)
   1f30c:	00160613          	addi	a2,a2,1
   1f310:	00430313          	addi	t1,t1,4
   1f314:	ff132e23          	sw	a7,-4(t1)
   1f318:	fec5d6e3          	bge	a1,a2,1f304 <__subtf3+0x3c4>
   1f31c:	00400613          	li	a2,4
   1f320:	40d606b3          	sub	a3,a2,a3
   1f324:	0640006f          	j	1f388 <__subtf3+0x448>
   1f328:	00052883          	lw	a7,0(a0)
   1f32c:	00158593          	addi	a1,a1,1
   1f330:	00450513          	addi	a0,a0,4
   1f334:	01176733          	or	a4,a4,a7
   1f338:	fb9ff06f          	j	1f2f0 <__subtf3+0x3b0>
   1f33c:	04050893          	addi	a7,a0,64
   1f340:	002888b3          	add	a7,a7,sp
   1f344:	fe08a883          	lw	a7,-32(a7)
   1f348:	02000e13          	li	t3,32
   1f34c:	40ce0e33          	sub	t3,t3,a2
   1f350:	01c898b3          	sll	a7,a7,t3
   1f354:	01176733          	or	a4,a4,a7
   1f358:	00000e93          	li	t4,0
   1f35c:	00a808b3          	add	a7,a6,a0
   1f360:	40a00533          	neg	a0,a0
   1f364:	14bec263          	blt	t4,a1,1f4a8 <__subtf3+0x568>
   1f368:	00400513          	li	a0,4
   1f36c:	40d506b3          	sub	a3,a0,a3
   1f370:	02c12503          	lw	a0,44(sp)
   1f374:	00259593          	slli	a1,a1,0x2
   1f378:	04058593          	addi	a1,a1,64
   1f37c:	002585b3          	add	a1,a1,sp
   1f380:	00c55633          	srl	a2,a0,a2
   1f384:	fec5a023          	sw	a2,-32(a1)
   1f388:	00400613          	li	a2,4
   1f38c:	40d60633          	sub	a2,a2,a3
   1f390:	00261613          	slli	a2,a2,0x2
   1f394:	00269693          	slli	a3,a3,0x2
   1f398:	00800593          	li	a1,8
   1f39c:	00d806b3          	add	a3,a6,a3
   1f3a0:	00b66a63          	bltu	a2,a1,1f3b4 <__subtf3+0x474>
   1f3a4:	0006a023          	sw	zero,0(a3) # 80000 <__BSS_END__+0x5d2d0>
   1f3a8:	0006a223          	sw	zero,4(a3)
   1f3ac:	ff860613          	addi	a2,a2,-8
   1f3b0:	00868693          	addi	a3,a3,8
   1f3b4:	00400593          	li	a1,4
   1f3b8:	00b66463          	bltu	a2,a1,1f3c0 <__subtf3+0x480>
   1f3bc:	0006a023          	sw	zero,0(a3)
   1f3c0:	02012683          	lw	a3,32(sp)
   1f3c4:	00e03733          	snez	a4,a4
   1f3c8:	00d76733          	or	a4,a4,a3
   1f3cc:	02412683          	lw	a3,36(sp)
   1f3d0:	02e12023          	sw	a4,32(sp)
   1f3d4:	00e78733          	add	a4,a5,a4
   1f3d8:	01268933          	add	s2,a3,s2
   1f3dc:	02e12823          	sw	a4,48(sp)
   1f3e0:	00f73733          	sltu	a4,a4,a5
   1f3e4:	02812783          	lw	a5,40(sp)
   1f3e8:	00e90733          	add	a4,s2,a4
   1f3ec:	02e12a23          	sw	a4,52(sp)
   1f3f0:	00d936b3          	sltu	a3,s2,a3
   1f3f4:	01273733          	sltu	a4,a4,s2
   1f3f8:	013789b3          	add	s3,a5,s3
   1f3fc:	00e6e733          	or	a4,a3,a4
   1f400:	00e98733          	add	a4,s3,a4
   1f404:	02e12c23          	sw	a4,56(sp)
   1f408:	00f9b7b3          	sltu	a5,s3,a5
   1f40c:	01373733          	sltu	a4,a4,s3
   1f410:	00e7e7b3          	or	a5,a5,a4
   1f414:	02c12703          	lw	a4,44(sp)
   1f418:	00eb0733          	add	a4,s6,a4
   1f41c:	00e787b3          	add	a5,a5,a4
   1f420:	02f12e23          	sw	a5,60(sp)
   1f424:	00c79713          	slli	a4,a5,0xc
   1f428:	d2075ae3          	bgez	a4,1f15c <__subtf3+0x21c>
   1f42c:	03012683          	lw	a3,48(sp)
   1f430:	fff80737          	lui	a4,0xfff80
   1f434:	fff70713          	addi	a4,a4,-1 # fff7ffff <__BSS_END__+0xfff5d2cf>
   1f438:	00e7f7b3          	and	a5,a5,a4
   1f43c:	02f12e23          	sw	a5,60(sp)
   1f440:	00148493          	addi	s1,s1,1
   1f444:	01f69693          	slli	a3,a3,0x1f
   1f448:	03010713          	addi	a4,sp,48
   1f44c:	03c10513          	addi	a0,sp,60
   1f450:	00072603          	lw	a2,0(a4)
   1f454:	00472583          	lw	a1,4(a4)
   1f458:	00470713          	addi	a4,a4,4
   1f45c:	00165613          	srli	a2,a2,0x1
   1f460:	01f59593          	slli	a1,a1,0x1f
   1f464:	00b66633          	or	a2,a2,a1
   1f468:	fec72e23          	sw	a2,-4(a4)
   1f46c:	fee512e3          	bne	a0,a4,1f450 <__subtf3+0x510>
   1f470:	03012703          	lw	a4,48(sp)
   1f474:	0017d793          	srli	a5,a5,0x1
   1f478:	02f12e23          	sw	a5,60(sp)
   1f47c:	00d037b3          	snez	a5,a3
   1f480:	00f767b3          	or	a5,a4,a5
   1f484:	02f12823          	sw	a5,48(sp)
   1f488:	000087b7          	lui	a5,0x8
   1f48c:	fff78793          	addi	a5,a5,-1 # 7fff <exit-0x80b5>
   1f490:	ccf496e3          	bne	s1,a5,1f15c <__subtf3+0x21c>
   1f494:	02012e23          	sw	zero,60(sp)
   1f498:	02012c23          	sw	zero,56(sp)
   1f49c:	02012a23          	sw	zero,52(sp)
   1f4a0:	02012823          	sw	zero,48(sp)
   1f4a4:	cb9ff06f          	j	1f15c <__subtf3+0x21c>
   1f4a8:	0008a303          	lw	t1,0(a7)
   1f4ac:	0048af03          	lw	t5,4(a7)
   1f4b0:	00a88fb3          	add	t6,a7,a0
   1f4b4:	00c35333          	srl	t1,t1,a2
   1f4b8:	01cf1f33          	sll	t5,t5,t3
   1f4bc:	01e36333          	or	t1,t1,t5
   1f4c0:	006fa023          	sw	t1,0(t6)
   1f4c4:	001e8e93          	addi	t4,t4,1
   1f4c8:	00488893          	addi	a7,a7,4
   1f4cc:	e99ff06f          	j	1f364 <__subtf3+0x424>
   1f4d0:	02412903          	lw	s2,36(sp)
   1f4d4:	02812983          	lw	s3,40(sp)
   1f4d8:	02c12b03          	lw	s6,44(sp)
   1f4dc:	26088263          	beqz	a7,1f740 <__subtf3+0x800>
   1f4e0:	409b8833          	sub	a6,s7,s1
   1f4e4:	0a049c63          	bnez	s1,1f59c <__subtf3+0x65c>
   1f4e8:	01412683          	lw	a3,20(sp)
   1f4ec:	01812703          	lw	a4,24(sp)
   1f4f0:	01c12883          	lw	a7,28(sp)
   1f4f4:	00e6e5b3          	or	a1,a3,a4
   1f4f8:	0115e5b3          	or	a1,a1,a7
   1f4fc:	00f5e5b3          	or	a1,a1,a5
   1f500:	00059e63          	bnez	a1,1f51c <__subtf3+0x5dc>
   1f504:	02812823          	sw	s0,48(sp)
   1f508:	03212a23          	sw	s2,52(sp)
   1f50c:	03312c23          	sw	s3,56(sp)
   1f510:	03612e23          	sw	s6,60(sp)
   1f514:	00080493          	mv	s1,a6
   1f518:	c45ff06f          	j	1f15c <__subtf3+0x21c>
   1f51c:	fff80593          	addi	a1,a6,-1
   1f520:	04059663          	bnez	a1,1f56c <__subtf3+0x62c>
   1f524:	00878433          	add	s0,a5,s0
   1f528:	01268933          	add	s2,a3,s2
   1f52c:	02812823          	sw	s0,48(sp)
   1f530:	00f43433          	sltu	s0,s0,a5
   1f534:	00890433          	add	s0,s2,s0
   1f538:	02812a23          	sw	s0,52(sp)
   1f53c:	00d936b3          	sltu	a3,s2,a3
   1f540:	01243433          	sltu	s0,s0,s2
   1f544:	013709b3          	add	s3,a4,s3
   1f548:	0086e6b3          	or	a3,a3,s0
   1f54c:	00d986b3          	add	a3,s3,a3
   1f550:	02d12c23          	sw	a3,56(sp)
   1f554:	00e9b7b3          	sltu	a5,s3,a4
   1f558:	0136b6b3          	sltu	a3,a3,s3
   1f55c:	00d7e7b3          	or	a5,a5,a3
   1f560:	016888b3          	add	a7,a7,s6
   1f564:	011787b3          	add	a5,a5,a7
   1f568:	bb9ff06f          	j	1f120 <__subtf3+0x1e0>
   1f56c:	000087b7          	lui	a5,0x8
   1f570:	fff78793          	addi	a5,a5,-1 # 7fff <exit-0x80b5>
   1f574:	62f818e3          	bne	a6,a5,203a4 <__subtf3+0x1464>
   1f578:	02812823          	sw	s0,48(sp)
   1f57c:	03212a23          	sw	s2,52(sp)
   1f580:	03312c23          	sw	s3,56(sp)
   1f584:	03612e23          	sw	s6,60(sp)
   1f588:	000084b7          	lui	s1,0x8
   1f58c:	fff48493          	addi	s1,s1,-1 # 7fff <exit-0x80b5>
   1f590:	bcdff06f          	j	1f15c <__subtf3+0x21c>
   1f594:	00078413          	mv	s0,a5
   1f598:	fe1ff06f          	j	1f578 <__subtf3+0x638>
   1f59c:	000087b7          	lui	a5,0x8
   1f5a0:	fff78793          	addi	a5,a5,-1 # 7fff <exit-0x80b5>
   1f5a4:	fcfb8ae3          	beq	s7,a5,1f578 <__subtf3+0x638>
   1f5a8:	01c12783          	lw	a5,28(sp)
   1f5ac:	00080737          	lui	a4,0x80
   1f5b0:	00e7e7b3          	or	a5,a5,a4
   1f5b4:	00f12e23          	sw	a5,28(sp)
   1f5b8:	07400793          	li	a5,116
   1f5bc:	5f07c8e3          	blt	a5,a6,203ac <__subtf3+0x146c>
   1f5c0:	00080593          	mv	a1,a6
   1f5c4:	02000713          	li	a4,32
   1f5c8:	02e5c733          	div	a4,a1,a4
   1f5cc:	00060693          	mv	a3,a2
   1f5d0:	00000493          	li	s1,0
   1f5d4:	00000793          	li	a5,0
   1f5d8:	02e7ce63          	blt	a5,a4,1f614 <__subtf3+0x6d4>
   1f5dc:	00300793          	li	a5,3
   1f5e0:	01f5f893          	andi	a7,a1,31
   1f5e4:	40e787b3          	sub	a5,a5,a4
   1f5e8:	00271813          	slli	a6,a4,0x2
   1f5ec:	02089e63          	bnez	a7,1f628 <__subtf3+0x6e8>
   1f5f0:	010606b3          	add	a3,a2,a6
   1f5f4:	0006a683          	lw	a3,0(a3)
   1f5f8:	00188893          	addi	a7,a7,1
   1f5fc:	00460613          	addi	a2,a2,4
   1f600:	fed62e23          	sw	a3,-4(a2)
   1f604:	ff17d6e3          	bge	a5,a7,1f5f0 <__subtf3+0x6b0>
   1f608:	00400793          	li	a5,4
   1f60c:	40e78733          	sub	a4,a5,a4
   1f610:	0780006f          	j	1f688 <__subtf3+0x748>
   1f614:	0006a803          	lw	a6,0(a3)
   1f618:	00178793          	addi	a5,a5,1
   1f61c:	00468693          	addi	a3,a3,4
   1f620:	0104e4b3          	or	s1,s1,a6
   1f624:	fb5ff06f          	j	1f5d8 <__subtf3+0x698>
   1f628:	02000693          	li	a3,32
   1f62c:	02d5e5b3          	rem	a1,a1,a3
   1f630:	40b685b3          	sub	a1,a3,a1
   1f634:	00070693          	mv	a3,a4
   1f638:	00075463          	bgez	a4,1f640 <__subtf3+0x700>
   1f63c:	00000693          	li	a3,0
   1f640:	00269693          	slli	a3,a3,0x2
   1f644:	04068693          	addi	a3,a3,64
   1f648:	002686b3          	add	a3,a3,sp
   1f64c:	fd06a683          	lw	a3,-48(a3)
   1f650:	00000313          	li	t1,0
   1f654:	00b696b3          	sll	a3,a3,a1
   1f658:	00d4e4b3          	or	s1,s1,a3
   1f65c:	010506b3          	add	a3,a0,a6
   1f660:	41000833          	neg	a6,a6
   1f664:	0af34a63          	blt	t1,a5,1f718 <__subtf3+0x7d8>
   1f668:	00400693          	li	a3,4
   1f66c:	40e68733          	sub	a4,a3,a4
   1f670:	01c12683          	lw	a3,28(sp)
   1f674:	00279793          	slli	a5,a5,0x2
   1f678:	04078793          	addi	a5,a5,64
   1f67c:	002787b3          	add	a5,a5,sp
   1f680:	0116d6b3          	srl	a3,a3,a7
   1f684:	fcd7a823          	sw	a3,-48(a5)
   1f688:	00572793          	slti	a5,a4,5
   1f68c:	00000613          	li	a2,0
   1f690:	00078863          	beqz	a5,1f6a0 <__subtf3+0x760>
   1f694:	00400613          	li	a2,4
   1f698:	40e60633          	sub	a2,a2,a4
   1f69c:	00261613          	slli	a2,a2,0x2
   1f6a0:	00271713          	slli	a4,a4,0x2
   1f6a4:	00e50533          	add	a0,a0,a4
   1f6a8:	00000593          	li	a1,0
   1f6ac:	fd4f10ef          	jal	10e80 <memset>
   1f6b0:	01012703          	lw	a4,16(sp)
   1f6b4:	009037b3          	snez	a5,s1
   1f6b8:	00e7e7b3          	or	a5,a5,a4
   1f6bc:	01412683          	lw	a3,20(sp)
   1f6c0:	00f12823          	sw	a5,16(sp)
   1f6c4:	00f407b3          	add	a5,s0,a5
   1f6c8:	01268933          	add	s2,a3,s2
   1f6cc:	02f12823          	sw	a5,48(sp)
   1f6d0:	0087b7b3          	sltu	a5,a5,s0
   1f6d4:	00f90733          	add	a4,s2,a5
   1f6d8:	01812783          	lw	a5,24(sp)
   1f6dc:	02e12a23          	sw	a4,52(sp)
   1f6e0:	00d936b3          	sltu	a3,s2,a3
   1f6e4:	01273733          	sltu	a4,a4,s2
   1f6e8:	013789b3          	add	s3,a5,s3
   1f6ec:	00e6e733          	or	a4,a3,a4
   1f6f0:	00e98733          	add	a4,s3,a4
   1f6f4:	02e12c23          	sw	a4,56(sp)
   1f6f8:	00f9b7b3          	sltu	a5,s3,a5
   1f6fc:	01373733          	sltu	a4,a4,s3
   1f700:	00e7e7b3          	or	a5,a5,a4
   1f704:	01c12703          	lw	a4,28(sp)
   1f708:	000b8493          	mv	s1,s7
   1f70c:	00eb0733          	add	a4,s6,a4
   1f710:	00e787b3          	add	a5,a5,a4
   1f714:	d0dff06f          	j	1f420 <__subtf3+0x4e0>
   1f718:	0006a603          	lw	a2,0(a3)
   1f71c:	0046ae03          	lw	t3,4(a3)
   1f720:	01068eb3          	add	t4,a3,a6
   1f724:	01165633          	srl	a2,a2,a7
   1f728:	00be1e33          	sll	t3,t3,a1
   1f72c:	01c66633          	or	a2,a2,t3
   1f730:	00cea023          	sw	a2,0(t4)
   1f734:	00130313          	addi	t1,t1,1
   1f738:	00468693          	addi	a3,a3,4
   1f73c:	f29ff06f          	j	1f664 <__subtf3+0x724>
   1f740:	00148813          	addi	a6,s1,1
   1f744:	01181893          	slli	a7,a6,0x11
   1f748:	0128d893          	srli	a7,a7,0x12
   1f74c:	01412683          	lw	a3,20(sp)
   1f750:	01812703          	lw	a4,24(sp)
   1f754:	01c12603          	lw	a2,28(sp)
   1f758:	03010593          	addi	a1,sp,48
   1f75c:	03c10513          	addi	a0,sp,60
   1f760:	10089e63          	bnez	a7,1f87c <__subtf3+0x93c>
   1f764:	00e6e833          	or	a6,a3,a4
   1f768:	00c86833          	or	a6,a6,a2
   1f76c:	00f86833          	or	a6,a6,a5
   1f770:	0a049863          	bnez	s1,1f820 <__subtf3+0x8e0>
   1f774:	00081e63          	bnez	a6,1f790 <__subtf3+0x850>
   1f778:	02812823          	sw	s0,48(sp)
   1f77c:	03212a23          	sw	s2,52(sp)
   1f780:	03312c23          	sw	s3,56(sp)
   1f784:	03612e23          	sw	s6,60(sp)
   1f788:	00000493          	li	s1,0
   1f78c:	9d1ff06f          	j	1f15c <__subtf3+0x21c>
   1f790:	013965b3          	or	a1,s2,s3
   1f794:	0165e5b3          	or	a1,a1,s6
   1f798:	0085e5b3          	or	a1,a1,s0
   1f79c:	00059c63          	bnez	a1,1f7b4 <__subtf3+0x874>
   1f7a0:	02f12823          	sw	a5,48(sp)
   1f7a4:	02d12a23          	sw	a3,52(sp)
   1f7a8:	02e12c23          	sw	a4,56(sp)
   1f7ac:	02c12e23          	sw	a2,60(sp)
   1f7b0:	9adff06f          	j	1f15c <__subtf3+0x21c>
   1f7b4:	00878433          	add	s0,a5,s0
   1f7b8:	01268933          	add	s2,a3,s2
   1f7bc:	02812823          	sw	s0,48(sp)
   1f7c0:	00f43433          	sltu	s0,s0,a5
   1f7c4:	00890433          	add	s0,s2,s0
   1f7c8:	02812a23          	sw	s0,52(sp)
   1f7cc:	00d936b3          	sltu	a3,s2,a3
   1f7d0:	01243433          	sltu	s0,s0,s2
   1f7d4:	013709b3          	add	s3,a4,s3
   1f7d8:	0086e6b3          	or	a3,a3,s0
   1f7dc:	00d986b3          	add	a3,s3,a3
   1f7e0:	02d12c23          	sw	a3,56(sp)
   1f7e4:	00e9b733          	sltu	a4,s3,a4
   1f7e8:	0136b6b3          	sltu	a3,a3,s3
   1f7ec:	00d76733          	or	a4,a4,a3
   1f7f0:	01660633          	add	a2,a2,s6
   1f7f4:	00c70733          	add	a4,a4,a2
   1f7f8:	00c71793          	slli	a5,a4,0xc
   1f7fc:	0007c663          	bltz	a5,1f808 <__subtf3+0x8c8>
   1f800:	02e12e23          	sw	a4,60(sp)
   1f804:	959ff06f          	j	1f15c <__subtf3+0x21c>
   1f808:	fff807b7          	lui	a5,0xfff80
   1f80c:	fff78793          	addi	a5,a5,-1 # fff7ffff <__BSS_END__+0xfff5d2cf>
   1f810:	00f77733          	and	a4,a4,a5
   1f814:	02e12e23          	sw	a4,60(sp)
   1f818:	00100493          	li	s1,1
   1f81c:	941ff06f          	j	1f15c <__subtf3+0x21c>
   1f820:	1a080463          	beqz	a6,1f9c8 <__subtf3+0xa88>
   1f824:	01396933          	or	s2,s2,s3
   1f828:	01696933          	or	s2,s2,s6
   1f82c:	00896933          	or	s2,s2,s0
   1f830:	1a090063          	beqz	s2,1f9d0 <__subtf3+0xa90>
   1f834:	000087b7          	lui	a5,0x8
   1f838:	02f12e23          	sw	a5,60(sp)
   1f83c:	02012c23          	sw	zero,56(sp)
   1f840:	02012a23          	sw	zero,52(sp)
   1f844:	02012823          	sw	zero,48(sp)
   1f848:	00050793          	mv	a5,a0
   1f84c:	0007a703          	lw	a4,0(a5) # 8000 <exit-0x80b4>
   1f850:	ffc7a683          	lw	a3,-4(a5)
   1f854:	ffc78793          	addi	a5,a5,-4
   1f858:	00371713          	slli	a4,a4,0x3
   1f85c:	01d6d693          	srli	a3,a3,0x1d
   1f860:	00d76733          	or	a4,a4,a3
   1f864:	00e7a223          	sw	a4,4(a5)
   1f868:	fef592e3          	bne	a1,a5,1f84c <__subtf3+0x90c>
   1f86c:	000084b7          	lui	s1,0x8
   1f870:	fff48493          	addi	s1,s1,-1 # 7fff <exit-0x80b5>
   1f874:	00000a93          	li	s5,0
   1f878:	8e5ff06f          	j	1f15c <__subtf3+0x21c>
   1f87c:	00878433          	add	s0,a5,s0
   1f880:	01268933          	add	s2,a3,s2
   1f884:	02812823          	sw	s0,48(sp)
   1f888:	00f43433          	sltu	s0,s0,a5
   1f88c:	00890433          	add	s0,s2,s0
   1f890:	02812a23          	sw	s0,52(sp)
   1f894:	00d936b3          	sltu	a3,s2,a3
   1f898:	01243433          	sltu	s0,s0,s2
   1f89c:	013709b3          	add	s3,a4,s3
   1f8a0:	0086e6b3          	or	a3,a3,s0
   1f8a4:	00d986b3          	add	a3,s3,a3
   1f8a8:	02d12c23          	sw	a3,56(sp)
   1f8ac:	00e9b733          	sltu	a4,s3,a4
   1f8b0:	0136b6b3          	sltu	a3,a3,s3
   1f8b4:	00d76733          	or	a4,a4,a3
   1f8b8:	01660633          	add	a2,a2,s6
   1f8bc:	00c70733          	add	a4,a4,a2
   1f8c0:	02e12e23          	sw	a4,60(sp)
   1f8c4:	00058793          	mv	a5,a1
   1f8c8:	0007a683          	lw	a3,0(a5)
   1f8cc:	0047a603          	lw	a2,4(a5)
   1f8d0:	00478793          	addi	a5,a5,4
   1f8d4:	0016d693          	srli	a3,a3,0x1
   1f8d8:	01f61613          	slli	a2,a2,0x1f
   1f8dc:	00c6e6b3          	or	a3,a3,a2
   1f8e0:	fed7ae23          	sw	a3,-4(a5)
   1f8e4:	fef512e3          	bne	a0,a5,1f8c8 <__subtf3+0x988>
   1f8e8:	000087b7          	lui	a5,0x8
   1f8ec:	fff78793          	addi	a5,a5,-1 # 7fff <exit-0x80b5>
   1f8f0:	00f80863          	beq	a6,a5,1f900 <__subtf3+0x9c0>
   1f8f4:	00175713          	srli	a4,a4,0x1
   1f8f8:	02e12e23          	sw	a4,60(sp)
   1f8fc:	c19ff06f          	j	1f514 <__subtf3+0x5d4>
   1f900:	02012e23          	sw	zero,60(sp)
   1f904:	02012c23          	sw	zero,56(sp)
   1f908:	02012a23          	sw	zero,52(sp)
   1f90c:	02012823          	sw	zero,48(sp)
   1f910:	c05ff06f          	j	1f514 <__subtf3+0x5d4>
   1f914:	29105c63          	blez	a7,1fbac <__subtf3+0xc6c>
   1f918:	01412903          	lw	s2,20(sp)
   1f91c:	01812983          	lw	s3,24(sp)
   1f920:	01c12b03          	lw	s6,28(sp)
   1f924:	0a0b9e63          	bnez	s7,1f9e0 <__subtf3+0xaa0>
   1f928:	02412e03          	lw	t3,36(sp)
   1f92c:	02812503          	lw	a0,40(sp)
   1f930:	02c12683          	lw	a3,44(sp)
   1f934:	00ae6733          	or	a4,t3,a0
   1f938:	00d76733          	or	a4,a4,a3
   1f93c:	00876733          	or	a4,a4,s0
   1f940:	f6070e63          	beqz	a4,1f0bc <__subtf3+0x17c>
   1f944:	fff88e93          	addi	t4,a7,-1
   1f948:	040e9c63          	bnez	t4,1f9a0 <__subtf3+0xa60>
   1f94c:	40878733          	sub	a4,a5,s0
   1f950:	41c90633          	sub	a2,s2,t3
   1f954:	00e7b5b3          	sltu	a1,a5,a4
   1f958:	00c93833          	sltu	a6,s2,a2
   1f95c:	40b60633          	sub	a2,a2,a1
   1f960:	00000593          	li	a1,0
   1f964:	00e7f663          	bgeu	a5,a4,1f970 <__subtf3+0xa30>
   1f968:	412e0e33          	sub	t3,t3,s2
   1f96c:	001e3593          	seqz	a1,t3
   1f970:	0105e7b3          	or	a5,a1,a6
   1f974:	40a985b3          	sub	a1,s3,a0
   1f978:	00b9b833          	sltu	a6,s3,a1
   1f97c:	40f585b3          	sub	a1,a1,a5
   1f980:	00078663          	beqz	a5,1f98c <__subtf3+0xa4c>
   1f984:	41350533          	sub	a0,a0,s3
   1f988:	00153e93          	seqz	t4,a0
   1f98c:	40db07b3          	sub	a5,s6,a3
   1f990:	010ee6b3          	or	a3,t4,a6
   1f994:	40d787b3          	sub	a5,a5,a3
   1f998:	00100493          	li	s1,1
   1f99c:	1bc0006f          	j	1fb58 <__subtf3+0xc18>
   1f9a0:	00008737          	lui	a4,0x8
   1f9a4:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1f9a8:	f8e88863          	beq	a7,a4,1f138 <__subtf3+0x1f8>
   1f9ac:	07400713          	li	a4,116
   1f9b0:	05d75c63          	bge	a4,t4,1fa08 <__subtf3+0xac8>
   1f9b4:	02012623          	sw	zero,44(sp)
   1f9b8:	02012423          	sw	zero,40(sp)
   1f9bc:	02012223          	sw	zero,36(sp)
   1f9c0:	00100713          	li	a4,1
   1f9c4:	1340006f          	j	1faf8 <__subtf3+0xbb8>
   1f9c8:	00040793          	mv	a5,s0
   1f9cc:	f6cff06f          	j	1f138 <__subtf3+0x1f8>
   1f9d0:	00068913          	mv	s2,a3
   1f9d4:	00070993          	mv	s3,a4
   1f9d8:	00060b13          	mv	s6,a2
   1f9dc:	f5cff06f          	j	1f138 <__subtf3+0x1f8>
   1f9e0:	00008737          	lui	a4,0x8
   1f9e4:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1f9e8:	bae486e3          	beq	s1,a4,1f594 <__subtf3+0x654>
   1f9ec:	02c12703          	lw	a4,44(sp)
   1f9f0:	000806b7          	lui	a3,0x80
   1f9f4:	00d76733          	or	a4,a4,a3
   1f9f8:	02e12623          	sw	a4,44(sp)
   1f9fc:	07400713          	li	a4,116
   1fa00:	fb174ae3          	blt	a4,a7,1f9b4 <__subtf3+0xa74>
   1fa04:	00088e93          	mv	t4,a7
   1fa08:	405ed693          	srai	a3,t4,0x5
   1fa0c:	00030593          	mv	a1,t1
   1fa10:	01fefe93          	andi	t4,t4,31
   1fa14:	00000713          	li	a4,0
   1fa18:	00000613          	li	a2,0
   1fa1c:	02d61c63          	bne	a2,a3,1fa54 <__subtf3+0xb14>
   1fa20:	00300613          	li	a2,3
   1fa24:	40d60633          	sub	a2,a2,a3
   1fa28:	00269593          	slli	a1,a3,0x2
   1fa2c:	020e9e63          	bnez	t4,1fa68 <__subtf3+0xb28>
   1fa30:	00b30533          	add	a0,t1,a1
   1fa34:	00052503          	lw	a0,0(a0)
   1fa38:	001e8e93          	addi	t4,t4,1
   1fa3c:	00430313          	addi	t1,t1,4
   1fa40:	fea32e23          	sw	a0,-4(t1)
   1fa44:	ffd656e3          	bge	a2,t4,1fa30 <__subtf3+0xaf0>
   1fa48:	00400613          	li	a2,4
   1fa4c:	40d606b3          	sub	a3,a2,a3
   1fa50:	0640006f          	j	1fab4 <__subtf3+0xb74>
   1fa54:	0005a503          	lw	a0,0(a1)
   1fa58:	00160613          	addi	a2,a2,1
   1fa5c:	00458593          	addi	a1,a1,4
   1fa60:	00a76733          	or	a4,a4,a0
   1fa64:	fb9ff06f          	j	1fa1c <__subtf3+0xadc>
   1fa68:	04058513          	addi	a0,a1,64
   1fa6c:	00250533          	add	a0,a0,sp
   1fa70:	fe052503          	lw	a0,-32(a0)
   1fa74:	02000313          	li	t1,32
   1fa78:	41d30333          	sub	t1,t1,t4
   1fa7c:	00651533          	sll	a0,a0,t1
   1fa80:	00a76733          	or	a4,a4,a0
   1fa84:	00000e13          	li	t3,0
   1fa88:	00b80533          	add	a0,a6,a1
   1fa8c:	40b005b3          	neg	a1,a1
   1fa90:	0ece4a63          	blt	t3,a2,1fb84 <__subtf3+0xc44>
   1fa94:	00400593          	li	a1,4
   1fa98:	40d586b3          	sub	a3,a1,a3
   1fa9c:	02c12583          	lw	a1,44(sp)
   1faa0:	00261613          	slli	a2,a2,0x2
   1faa4:	04060613          	addi	a2,a2,64
   1faa8:	00260633          	add	a2,a2,sp
   1faac:	01d5d5b3          	srl	a1,a1,t4
   1fab0:	feb62023          	sw	a1,-32(a2)
   1fab4:	00400613          	li	a2,4
   1fab8:	40d60633          	sub	a2,a2,a3
   1fabc:	00261613          	slli	a2,a2,0x2
   1fac0:	00269693          	slli	a3,a3,0x2
   1fac4:	00800593          	li	a1,8
   1fac8:	00d806b3          	add	a3,a6,a3
   1facc:	00b66a63          	bltu	a2,a1,1fae0 <__subtf3+0xba0>
   1fad0:	0006a023          	sw	zero,0(a3) # 80000 <__BSS_END__+0x5d2d0>
   1fad4:	0006a223          	sw	zero,4(a3)
   1fad8:	ff860613          	addi	a2,a2,-8
   1fadc:	00868693          	addi	a3,a3,8
   1fae0:	00400593          	li	a1,4
   1fae4:	00b66463          	bltu	a2,a1,1faec <__subtf3+0xbac>
   1fae8:	0006a023          	sw	zero,0(a3)
   1faec:	02012683          	lw	a3,32(sp)
   1faf0:	00e03733          	snez	a4,a4
   1faf4:	00d76733          	or	a4,a4,a3
   1faf8:	02412583          	lw	a1,36(sp)
   1fafc:	02e12023          	sw	a4,32(sp)
   1fb00:	40e78733          	sub	a4,a5,a4
   1fb04:	40b90633          	sub	a2,s2,a1
   1fb08:	00e7b6b3          	sltu	a3,a5,a4
   1fb0c:	00c93533          	sltu	a0,s2,a2
   1fb10:	40d60633          	sub	a2,a2,a3
   1fb14:	00000693          	li	a3,0
   1fb18:	00e7f663          	bgeu	a5,a4,1fb24 <__subtf3+0xbe4>
   1fb1c:	412585b3          	sub	a1,a1,s2
   1fb20:	0015b693          	seqz	a3,a1
   1fb24:	00a6e7b3          	or	a5,a3,a0
   1fb28:	02812503          	lw	a0,40(sp)
   1fb2c:	00000693          	li	a3,0
   1fb30:	40a985b3          	sub	a1,s3,a0
   1fb34:	00b9b833          	sltu	a6,s3,a1
   1fb38:	40f585b3          	sub	a1,a1,a5
   1fb3c:	00078663          	beqz	a5,1fb48 <__subtf3+0xc08>
   1fb40:	41350533          	sub	a0,a0,s3
   1fb44:	00153693          	seqz	a3,a0
   1fb48:	02c12783          	lw	a5,44(sp)
   1fb4c:	0106e6b3          	or	a3,a3,a6
   1fb50:	40fb07b3          	sub	a5,s6,a5
   1fb54:	40d787b3          	sub	a5,a5,a3
   1fb58:	02e12823          	sw	a4,48(sp)
   1fb5c:	02f12e23          	sw	a5,60(sp)
   1fb60:	02b12c23          	sw	a1,56(sp)
   1fb64:	02c12a23          	sw	a2,52(sp)
   1fb68:	00c79713          	slli	a4,a5,0xc
   1fb6c:	de075863          	bgez	a4,1f15c <__subtf3+0x21c>
   1fb70:	00080737          	lui	a4,0x80
   1fb74:	fff70713          	addi	a4,a4,-1 # 7ffff <__BSS_END__+0x5d2cf>
   1fb78:	00e7f7b3          	and	a5,a5,a4
   1fb7c:	02f12e23          	sw	a5,60(sp)
   1fb80:	5700006f          	j	200f0 <__subtf3+0x11b0>
   1fb84:	00052883          	lw	a7,0(a0)
   1fb88:	00452f03          	lw	t5,4(a0)
   1fb8c:	00b50fb3          	add	t6,a0,a1
   1fb90:	01d8d8b3          	srl	a7,a7,t4
   1fb94:	006f1f33          	sll	t5,t5,t1
   1fb98:	01e8e8b3          	or	a7,a7,t5
   1fb9c:	011fa023          	sw	a7,0(t6)
   1fba0:	001e0e13          	addi	t3,t3,1
   1fba4:	00450513          	addi	a0,a0,4
   1fba8:	ee9ff06f          	j	1fa90 <__subtf3+0xb50>
   1fbac:	02412c03          	lw	s8,36(sp)
   1fbb0:	02812b03          	lw	s6,40(sp)
   1fbb4:	02c12983          	lw	s3,44(sp)
   1fbb8:	28088463          	beqz	a7,1fe40 <__subtf3+0xf00>
   1fbbc:	409b8333          	sub	t1,s7,s1
   1fbc0:	0a049e63          	bnez	s1,1fc7c <__subtf3+0xd3c>
   1fbc4:	01412583          	lw	a1,20(sp)
   1fbc8:	01812803          	lw	a6,24(sp)
   1fbcc:	01c12683          	lw	a3,28(sp)
   1fbd0:	0105e8b3          	or	a7,a1,a6
   1fbd4:	00d8e8b3          	or	a7,a7,a3
   1fbd8:	00f8e8b3          	or	a7,a7,a5
   1fbdc:	02089063          	bnez	a7,1fbfc <__subtf3+0xcbc>
   1fbe0:	02812823          	sw	s0,48(sp)
   1fbe4:	03812a23          	sw	s8,52(sp)
   1fbe8:	03612c23          	sw	s6,56(sp)
   1fbec:	03312e23          	sw	s3,60(sp)
   1fbf0:	00030493          	mv	s1,t1
   1fbf4:	00090a93          	mv	s5,s2
   1fbf8:	d64ff06f          	j	1f15c <__subtf3+0x21c>
   1fbfc:	fff30893          	addi	a7,t1,-1
   1fc00:	04089c63          	bnez	a7,1fc58 <__subtf3+0xd18>
   1fc04:	40f40733          	sub	a4,s0,a5
   1fc08:	40bc0633          	sub	a2,s8,a1
   1fc0c:	00e437b3          	sltu	a5,s0,a4
   1fc10:	00cc3533          	sltu	a0,s8,a2
   1fc14:	40f60633          	sub	a2,a2,a5
   1fc18:	00000793          	li	a5,0
   1fc1c:	00e47663          	bgeu	s0,a4,1fc28 <__subtf3+0xce8>
   1fc20:	418585b3          	sub	a1,a1,s8
   1fc24:	0015b793          	seqz	a5,a1
   1fc28:	00a7e7b3          	or	a5,a5,a0
   1fc2c:	410b05b3          	sub	a1,s6,a6
   1fc30:	00bb3533          	sltu	a0,s6,a1
   1fc34:	40f585b3          	sub	a1,a1,a5
   1fc38:	00078663          	beqz	a5,1fc44 <__subtf3+0xd04>
   1fc3c:	41680833          	sub	a6,a6,s6
   1fc40:	00183893          	seqz	a7,a6
   1fc44:	40d987b3          	sub	a5,s3,a3
   1fc48:	00a8e6b3          	or	a3,a7,a0
   1fc4c:	40d787b3          	sub	a5,a5,a3
   1fc50:	00090a93          	mv	s5,s2
   1fc54:	d45ff06f          	j	1f998 <__subtf3+0xa58>
   1fc58:	000087b7          	lui	a5,0x8
   1fc5c:	fff78793          	addi	a5,a5,-1 # 7fff <exit-0x80b5>
   1fc60:	76f31063          	bne	t1,a5,203c0 <__subtf3+0x1480>
   1fc64:	02812823          	sw	s0,48(sp)
   1fc68:	03812a23          	sw	s8,52(sp)
   1fc6c:	03612c23          	sw	s6,56(sp)
   1fc70:	03312e23          	sw	s3,60(sp)
   1fc74:	00090a93          	mv	s5,s2
   1fc78:	911ff06f          	j	1f588 <__subtf3+0x648>
   1fc7c:	000087b7          	lui	a5,0x8
   1fc80:	fff78793          	addi	a5,a5,-1 # 7fff <exit-0x80b5>
   1fc84:	fefb80e3          	beq	s7,a5,1fc64 <__subtf3+0xd24>
   1fc88:	01c12783          	lw	a5,28(sp)
   1fc8c:	00080737          	lui	a4,0x80
   1fc90:	00e7e7b3          	or	a5,a5,a4
   1fc94:	00f12e23          	sw	a5,28(sp)
   1fc98:	07400793          	li	a5,116
   1fc9c:	1867c863          	blt	a5,t1,1fe2c <__subtf3+0xeec>
   1fca0:	02000793          	li	a5,32
   1fca4:	02f347b3          	div	a5,t1,a5
   1fca8:	00060693          	mv	a3,a2
   1fcac:	00000493          	li	s1,0
   1fcb0:	00000713          	li	a4,0
   1fcb4:	02f74e63          	blt	a4,a5,1fcf0 <__subtf3+0xdb0>
   1fcb8:	00300713          	li	a4,3
   1fcbc:	01f37893          	andi	a7,t1,31
   1fcc0:	40f70e33          	sub	t3,a4,a5
   1fcc4:	00279593          	slli	a1,a5,0x2
   1fcc8:	02089e63          	bnez	a7,1fd04 <__subtf3+0xdc4>
   1fccc:	00b60733          	add	a4,a2,a1
   1fcd0:	00072703          	lw	a4,0(a4) # 80000 <__BSS_END__+0x5d2d0>
   1fcd4:	00188893          	addi	a7,a7,1
   1fcd8:	00460613          	addi	a2,a2,4
   1fcdc:	fee62e23          	sw	a4,-4(a2)
   1fce0:	ff1e56e3          	bge	t3,a7,1fccc <__subtf3+0xd8c>
   1fce4:	00400713          	li	a4,4
   1fce8:	40f707b3          	sub	a5,a4,a5
   1fcec:	0780006f          	j	1fd64 <__subtf3+0xe24>
   1fcf0:	0006a583          	lw	a1,0(a3)
   1fcf4:	00170713          	addi	a4,a4,1
   1fcf8:	00468693          	addi	a3,a3,4
   1fcfc:	00b4e4b3          	or	s1,s1,a1
   1fd00:	fb5ff06f          	j	1fcb4 <__subtf3+0xd74>
   1fd04:	02000613          	li	a2,32
   1fd08:	02c36733          	rem	a4,t1,a2
   1fd0c:	00078693          	mv	a3,a5
   1fd10:	40e60633          	sub	a2,a2,a4
   1fd14:	0007d463          	bgez	a5,1fd1c <__subtf3+0xddc>
   1fd18:	00000693          	li	a3,0
   1fd1c:	00269693          	slli	a3,a3,0x2
   1fd20:	04068713          	addi	a4,a3,64
   1fd24:	002706b3          	add	a3,a4,sp
   1fd28:	fd06a703          	lw	a4,-48(a3)
   1fd2c:	00b506b3          	add	a3,a0,a1
   1fd30:	40b005b3          	neg	a1,a1
   1fd34:	00c71733          	sll	a4,a4,a2
   1fd38:	00e4e4b3          	or	s1,s1,a4
   1fd3c:	00000713          	li	a4,0
   1fd40:	0dc74263          	blt	a4,t3,1fe04 <__subtf3+0xec4>
   1fd44:	01c12683          	lw	a3,28(sp)
   1fd48:	00400713          	li	a4,4
   1fd4c:	40f707b3          	sub	a5,a4,a5
   1fd50:	002e1713          	slli	a4,t3,0x2
   1fd54:	04070713          	addi	a4,a4,64
   1fd58:	00270733          	add	a4,a4,sp
   1fd5c:	0116d6b3          	srl	a3,a3,a7
   1fd60:	fcd72823          	sw	a3,-48(a4)
   1fd64:	0057a713          	slti	a4,a5,5
   1fd68:	00000613          	li	a2,0
   1fd6c:	00070863          	beqz	a4,1fd7c <__subtf3+0xe3c>
   1fd70:	00400613          	li	a2,4
   1fd74:	40f60633          	sub	a2,a2,a5
   1fd78:	00261613          	slli	a2,a2,0x2
   1fd7c:	00279793          	slli	a5,a5,0x2
   1fd80:	00f50533          	add	a0,a0,a5
   1fd84:	00000593          	li	a1,0
   1fd88:	8f8f10ef          	jal	10e80 <memset>
   1fd8c:	01012783          	lw	a5,16(sp)
   1fd90:	00903733          	snez	a4,s1
   1fd94:	00f76733          	or	a4,a4,a5
   1fd98:	01412683          	lw	a3,20(sp)
   1fd9c:	00e12823          	sw	a4,16(sp)
   1fda0:	40e40733          	sub	a4,s0,a4
   1fda4:	40dc0633          	sub	a2,s8,a3
   1fda8:	00e437b3          	sltu	a5,s0,a4
   1fdac:	00cc35b3          	sltu	a1,s8,a2
   1fdb0:	40f60633          	sub	a2,a2,a5
   1fdb4:	00000793          	li	a5,0
   1fdb8:	00e47663          	bgeu	s0,a4,1fdc4 <__subtf3+0xe84>
   1fdbc:	418686b3          	sub	a3,a3,s8
   1fdc0:	0016b793          	seqz	a5,a3
   1fdc4:	01812503          	lw	a0,24(sp)
   1fdc8:	00b7e7b3          	or	a5,a5,a1
   1fdcc:	00000693          	li	a3,0
   1fdd0:	40ab05b3          	sub	a1,s6,a0
   1fdd4:	00bb3833          	sltu	a6,s6,a1
   1fdd8:	40f585b3          	sub	a1,a1,a5
   1fddc:	00078663          	beqz	a5,1fde8 <__subtf3+0xea8>
   1fde0:	41650533          	sub	a0,a0,s6
   1fde4:	00153693          	seqz	a3,a0
   1fde8:	01c12783          	lw	a5,28(sp)
   1fdec:	0106e6b3          	or	a3,a3,a6
   1fdf0:	000b8493          	mv	s1,s7
   1fdf4:	40f987b3          	sub	a5,s3,a5
   1fdf8:	40d787b3          	sub	a5,a5,a3
   1fdfc:	00090a93          	mv	s5,s2
   1fe00:	d59ff06f          	j	1fb58 <__subtf3+0xc18>
   1fe04:	0006a803          	lw	a6,0(a3)
   1fe08:	0046a303          	lw	t1,4(a3)
   1fe0c:	00b68eb3          	add	t4,a3,a1
   1fe10:	01185833          	srl	a6,a6,a7
   1fe14:	00c31333          	sll	t1,t1,a2
   1fe18:	00686833          	or	a6,a6,t1
   1fe1c:	010ea023          	sw	a6,0(t4)
   1fe20:	00170713          	addi	a4,a4,1
   1fe24:	00468693          	addi	a3,a3,4
   1fe28:	f19ff06f          	j	1fd40 <__subtf3+0xe00>
   1fe2c:	00012e23          	sw	zero,28(sp)
   1fe30:	00012c23          	sw	zero,24(sp)
   1fe34:	00012a23          	sw	zero,20(sp)
   1fe38:	00100713          	li	a4,1
   1fe3c:	f5dff06f          	j	1fd98 <__subtf3+0xe58>
   1fe40:	00148593          	addi	a1,s1,1
   1fe44:	01159513          	slli	a0,a1,0x11
   1fe48:	01255513          	srli	a0,a0,0x12
   1fe4c:	01412683          	lw	a3,20(sp)
   1fe50:	01812603          	lw	a2,24(sp)
   1fe54:	01c12703          	lw	a4,28(sp)
   1fe58:	00008837          	lui	a6,0x8
   1fe5c:	1c051e63          	bnez	a0,20038 <__subtf3+0x10f8>
   1fe60:	016c6533          	or	a0,s8,s6
   1fe64:	00c6e5b3          	or	a1,a3,a2
   1fe68:	01356533          	or	a0,a0,s3
   1fe6c:	00e5e5b3          	or	a1,a1,a4
   1fe70:	00856533          	or	a0,a0,s0
   1fe74:	00f5e5b3          	or	a1,a1,a5
   1fe78:	10049863          	bnez	s1,1ff88 <__subtf3+0x1048>
   1fe7c:	02059263          	bnez	a1,1fea0 <__subtf3+0xf60>
   1fe80:	02812823          	sw	s0,48(sp)
   1fe84:	03812a23          	sw	s8,52(sp)
   1fe88:	03612c23          	sw	s6,56(sp)
   1fe8c:	03312e23          	sw	s3,60(sp)
   1fe90:	00090a93          	mv	s5,s2
   1fe94:	ac051463          	bnez	a0,1f15c <__subtf3+0x21c>
   1fe98:	00000493          	li	s1,0
   1fe9c:	9d9ff06f          	j	1f874 <__subtf3+0x934>
   1fea0:	00051c63          	bnez	a0,1feb8 <__subtf3+0xf78>
   1fea4:	02f12823          	sw	a5,48(sp)
   1fea8:	02d12a23          	sw	a3,52(sp)
   1feac:	02c12c23          	sw	a2,56(sp)
   1feb0:	02e12e23          	sw	a4,60(sp)
   1feb4:	8d5ff06f          	j	1f788 <__subtf3+0x848>
   1feb8:	40878533          	sub	a0,a5,s0
   1febc:	41868e33          	sub	t3,a3,s8
   1fec0:	00a7b833          	sltu	a6,a5,a0
   1fec4:	01c6b8b3          	sltu	a7,a3,t3
   1fec8:	410e0833          	sub	a6,t3,a6
   1fecc:	00000593          	li	a1,0
   1fed0:	00a7f463          	bgeu	a5,a0,1fed8 <__subtf3+0xf98>
   1fed4:	001e3593          	seqz	a1,t3
   1fed8:	0115e5b3          	or	a1,a1,a7
   1fedc:	416608b3          	sub	a7,a2,s6
   1fee0:	01163f33          	sltu	t5,a2,a7
   1fee4:	40b88eb3          	sub	t4,a7,a1
   1fee8:	00000313          	li	t1,0
   1feec:	00058463          	beqz	a1,1fef4 <__subtf3+0xfb4>
   1fef0:	0018b313          	seqz	t1,a7
   1fef4:	01e36333          	or	t1,t1,t5
   1fef8:	413705b3          	sub	a1,a4,s3
   1fefc:	406585b3          	sub	a1,a1,t1
   1ff00:	02b12e23          	sw	a1,60(sp)
   1ff04:	03d12c23          	sw	t4,56(sp)
   1ff08:	03012a23          	sw	a6,52(sp)
   1ff0c:	02a12823          	sw	a0,48(sp)
   1ff10:	00c59313          	slli	t1,a1,0xc
   1ff14:	06035063          	bgez	t1,1ff74 <__subtf3+0x1034>
   1ff18:	40f407b3          	sub	a5,s0,a5
   1ff1c:	40dc06b3          	sub	a3,s8,a3
   1ff20:	00f435b3          	sltu	a1,s0,a5
   1ff24:	00dc3c33          	sltu	s8,s8,a3
   1ff28:	40b686b3          	sub	a3,a3,a1
   1ff2c:	00000593          	li	a1,0
   1ff30:	00f47463          	bgeu	s0,a5,1ff38 <__subtf3+0xff8>
   1ff34:	001e3593          	seqz	a1,t3
   1ff38:	40cb0633          	sub	a2,s6,a2
   1ff3c:	0185ec33          	or	s8,a1,s8
   1ff40:	00cb3b33          	sltu	s6,s6,a2
   1ff44:	00000513          	li	a0,0
   1ff48:	41860633          	sub	a2,a2,s8
   1ff4c:	000c0463          	beqz	s8,1ff54 <__subtf3+0x1014>
   1ff50:	0018b513          	seqz	a0,a7
   1ff54:	40e98733          	sub	a4,s3,a4
   1ff58:	01656533          	or	a0,a0,s6
   1ff5c:	40a70733          	sub	a4,a4,a0
   1ff60:	02e12e23          	sw	a4,60(sp)
   1ff64:	02c12c23          	sw	a2,56(sp)
   1ff68:	02d12a23          	sw	a3,52(sp)
   1ff6c:	02f12823          	sw	a5,48(sp)
   1ff70:	c85ff06f          	j	1fbf4 <__subtf3+0xcb4>
   1ff74:	01056533          	or	a0,a0,a6
   1ff78:	01d56533          	or	a0,a0,t4
   1ff7c:	00b56533          	or	a0,a0,a1
   1ff80:	f0050ce3          	beqz	a0,1fe98 <__subtf3+0xf58>
   1ff84:	805ff06f          	j	1f788 <__subtf3+0x848>
   1ff88:	03010893          	addi	a7,sp,48
   1ff8c:	04059e63          	bnez	a1,1ffe8 <__subtf3+0x10a8>
   1ff90:	02051e63          	bnez	a0,1ffcc <__subtf3+0x108c>
   1ff94:	03012e23          	sw	a6,60(sp)
   1ff98:	02012c23          	sw	zero,56(sp)
   1ff9c:	02012a23          	sw	zero,52(sp)
   1ffa0:	02012823          	sw	zero,48(sp)
   1ffa4:	03c10793          	addi	a5,sp,60
   1ffa8:	0007a703          	lw	a4,0(a5)
   1ffac:	ffc7a683          	lw	a3,-4(a5)
   1ffb0:	ffc78793          	addi	a5,a5,-4
   1ffb4:	00371713          	slli	a4,a4,0x3
   1ffb8:	01d6d693          	srli	a3,a3,0x1d
   1ffbc:	00d76733          	or	a4,a4,a3
   1ffc0:	00e7a223          	sw	a4,4(a5)
   1ffc4:	fef892e3          	bne	a7,a5,1ffa8 <__subtf3+0x1068>
   1ffc8:	8a5ff06f          	j	1f86c <__subtf3+0x92c>
   1ffcc:	02812823          	sw	s0,48(sp)
   1ffd0:	03812a23          	sw	s8,52(sp)
   1ffd4:	03612c23          	sw	s6,56(sp)
   1ffd8:	03312e23          	sw	s3,60(sp)
   1ffdc:	00090a93          	mv	s5,s2
   1ffe0:	fff80493          	addi	s1,a6,-1 # 7fff <exit-0x80b5>
   1ffe4:	978ff06f          	j	1f15c <__subtf3+0x21c>
   1ffe8:	00051c63          	bnez	a0,20000 <__subtf3+0x10c0>
   1ffec:	02f12823          	sw	a5,48(sp)
   1fff0:	02d12a23          	sw	a3,52(sp)
   1fff4:	02c12c23          	sw	a2,56(sp)
   1fff8:	02e12e23          	sw	a4,60(sp)
   1fffc:	fe5ff06f          	j	1ffe0 <__subtf3+0x10a0>
   20000:	03012e23          	sw	a6,60(sp)
   20004:	02012c23          	sw	zero,56(sp)
   20008:	02012a23          	sw	zero,52(sp)
   2000c:	02012823          	sw	zero,48(sp)
   20010:	03c10793          	addi	a5,sp,60
   20014:	0007a703          	lw	a4,0(a5)
   20018:	ffc7a683          	lw	a3,-4(a5)
   2001c:	ffc78793          	addi	a5,a5,-4
   20020:	00371713          	slli	a4,a4,0x3
   20024:	01d6d693          	srli	a3,a3,0x1d
   20028:	00d76733          	or	a4,a4,a3
   2002c:	00e7a223          	sw	a4,4(a5)
   20030:	fef892e3          	bne	a7,a5,20014 <__subtf3+0x10d4>
   20034:	839ff06f          	j	1f86c <__subtf3+0x92c>
   20038:	40878533          	sub	a0,a5,s0
   2003c:	41868eb3          	sub	t4,a3,s8
   20040:	00a7b833          	sltu	a6,a5,a0
   20044:	01d6b333          	sltu	t1,a3,t4
   20048:	410e8833          	sub	a6,t4,a6
   2004c:	00000593          	li	a1,0
   20050:	00a7f463          	bgeu	a5,a0,20058 <__subtf3+0x1118>
   20054:	001eb593          	seqz	a1,t4
   20058:	0065e5b3          	or	a1,a1,t1
   2005c:	41660333          	sub	t1,a2,s6
   20060:	00663fb3          	sltu	t6,a2,t1
   20064:	40b30f33          	sub	t5,t1,a1
   20068:	00000e13          	li	t3,0
   2006c:	00058463          	beqz	a1,20074 <__subtf3+0x1134>
   20070:	00133e13          	seqz	t3,t1
   20074:	01fe6e33          	or	t3,t3,t6
   20078:	413705b3          	sub	a1,a4,s3
   2007c:	41c585b3          	sub	a1,a1,t3
   20080:	02b12e23          	sw	a1,60(sp)
   20084:	03e12c23          	sw	t5,56(sp)
   20088:	03012a23          	sw	a6,52(sp)
   2008c:	02a12823          	sw	a0,48(sp)
   20090:	00c59e13          	slli	t3,a1,0xc
   20094:	140e5663          	bgez	t3,201e0 <__subtf3+0x12a0>
   20098:	40f407b3          	sub	a5,s0,a5
   2009c:	40dc06b3          	sub	a3,s8,a3
   200a0:	00f435b3          	sltu	a1,s0,a5
   200a4:	00dc3c33          	sltu	s8,s8,a3
   200a8:	40b686b3          	sub	a3,a3,a1
   200ac:	00000593          	li	a1,0
   200b0:	00f47463          	bgeu	s0,a5,200b8 <__subtf3+0x1178>
   200b4:	001eb593          	seqz	a1,t4
   200b8:	40cb0633          	sub	a2,s6,a2
   200bc:	0185ec33          	or	s8,a1,s8
   200c0:	00cb3b33          	sltu	s6,s6,a2
   200c4:	41860633          	sub	a2,a2,s8
   200c8:	000c0463          	beqz	s8,200d0 <__subtf3+0x1190>
   200cc:	00133893          	seqz	a7,t1
   200d0:	40e985b3          	sub	a1,s3,a4
   200d4:	0168e733          	or	a4,a7,s6
   200d8:	40e58733          	sub	a4,a1,a4
   200dc:	02e12e23          	sw	a4,60(sp)
   200e0:	02c12c23          	sw	a2,56(sp)
   200e4:	02d12a23          	sw	a3,52(sp)
   200e8:	02f12823          	sw	a5,48(sp)
   200ec:	00090a93          	mv	s5,s2
   200f0:	03c12503          	lw	a0,60(sp)
   200f4:	10050063          	beqz	a0,201f4 <__subtf3+0x12b4>
   200f8:	7bc000ef          	jal	208b4 <__clzsi2>
   200fc:	ff450513          	addi	a0,a0,-12
   20100:	02000713          	li	a4,32
   20104:	01f57813          	andi	a6,a0,31
   20108:	02e547b3          	div	a5,a0,a4
   2010c:	12080063          	beqz	a6,2022c <__subtf3+0x12ec>
   20110:	03010313          	addi	t1,sp,48
   20114:	02e566b3          	rem	a3,a0,a4
   20118:	40d70633          	sub	a2,a4,a3
   2011c:	ffc00693          	li	a3,-4
   20120:	02d786b3          	mul	a3,a5,a3
   20124:	00c68713          	addi	a4,a3,12
   20128:	00e30733          	add	a4,t1,a4
   2012c:	40d006b3          	neg	a3,a3
   20130:	12e31663          	bne	t1,a4,2025c <__subtf3+0x131c>
   20134:	03012683          	lw	a3,48(sp)
   20138:	fff78713          	addi	a4,a5,-1
   2013c:	00279793          	slli	a5,a5,0x2
   20140:	04078793          	addi	a5,a5,64
   20144:	002787b3          	add	a5,a5,sp
   20148:	010696b3          	sll	a3,a3,a6
   2014c:	fed7a823          	sw	a3,-16(a5)
   20150:	00170713          	addi	a4,a4,1
   20154:	03010793          	addi	a5,sp,48
   20158:	00271713          	slli	a4,a4,0x2
   2015c:	00800693          	li	a3,8
   20160:	00078893          	mv	a7,a5
   20164:	00d76a63          	bltu	a4,a3,20178 <__subtf3+0x1238>
   20168:	02012823          	sw	zero,48(sp)
   2016c:	0007a223          	sw	zero,4(a5)
   20170:	ff870713          	addi	a4,a4,-8
   20174:	03810793          	addi	a5,sp,56
   20178:	00400693          	li	a3,4
   2017c:	00d76463          	bltu	a4,a3,20184 <__subtf3+0x1244>
   20180:	0007a023          	sw	zero,0(a5)
   20184:	1c954863          	blt	a0,s1,20354 <__subtf3+0x1414>
   20188:	40950533          	sub	a0,a0,s1
   2018c:	00150513          	addi	a0,a0,1
   20190:	40555713          	srai	a4,a0,0x5
   20194:	01f57793          	andi	a5,a0,31
   20198:	00088593          	mv	a1,a7
   2019c:	00088613          	mv	a2,a7
   201a0:	00000313          	li	t1,0
   201a4:	00000693          	li	a3,0
   201a8:	0ce69c63          	bne	a3,a4,20280 <__subtf3+0x1340>
   201ac:	00300693          	li	a3,3
   201b0:	40e686b3          	sub	a3,a3,a4
   201b4:	00271613          	slli	a2,a4,0x2
   201b8:	0c079e63          	bnez	a5,20294 <__subtf3+0x1354>
   201bc:	00c58533          	add	a0,a1,a2
   201c0:	00052503          	lw	a0,0(a0)
   201c4:	00178793          	addi	a5,a5,1
   201c8:	00458593          	addi	a1,a1,4
   201cc:	fea5ae23          	sw	a0,-4(a1)
   201d0:	fef6d6e3          	bge	a3,a5,201bc <__subtf3+0x127c>
   201d4:	00400793          	li	a5,4
   201d8:	40e78733          	sub	a4,a5,a4
   201dc:	1040006f          	j	202e0 <__subtf3+0x13a0>
   201e0:	01056533          	or	a0,a0,a6
   201e4:	01e56533          	or	a0,a0,t5
   201e8:	00b56533          	or	a0,a0,a1
   201ec:	ca0506e3          	beqz	a0,1fe98 <__subtf3+0xf58>
   201f0:	f01ff06f          	j	200f0 <__subtf3+0x11b0>
   201f4:	03812503          	lw	a0,56(sp)
   201f8:	00050863          	beqz	a0,20208 <__subtf3+0x12c8>
   201fc:	6b8000ef          	jal	208b4 <__clzsi2>
   20200:	02050513          	addi	a0,a0,32
   20204:	ef9ff06f          	j	200fc <__subtf3+0x11bc>
   20208:	03412503          	lw	a0,52(sp)
   2020c:	00050863          	beqz	a0,2021c <__subtf3+0x12dc>
   20210:	6a4000ef          	jal	208b4 <__clzsi2>
   20214:	04050513          	addi	a0,a0,64
   20218:	ee5ff06f          	j	200fc <__subtf3+0x11bc>
   2021c:	03012503          	lw	a0,48(sp)
   20220:	694000ef          	jal	208b4 <__clzsi2>
   20224:	06050513          	addi	a0,a0,96
   20228:	ed5ff06f          	j	200fc <__subtf3+0x11bc>
   2022c:	ffc00613          	li	a2,-4
   20230:	02c78633          	mul	a2,a5,a2
   20234:	03c10713          	addi	a4,sp,60
   20238:	00300693          	li	a3,3
   2023c:	00c705b3          	add	a1,a4,a2
   20240:	0005a583          	lw	a1,0(a1)
   20244:	fff68693          	addi	a3,a3,-1
   20248:	ffc70713          	addi	a4,a4,-4
   2024c:	00b72223          	sw	a1,4(a4)
   20250:	fef6d6e3          	bge	a3,a5,2023c <__subtf3+0x12fc>
   20254:	fff78713          	addi	a4,a5,-1
   20258:	ef9ff06f          	j	20150 <__subtf3+0x1210>
   2025c:	00072583          	lw	a1,0(a4)
   20260:	ffc72883          	lw	a7,-4(a4)
   20264:	00d70e33          	add	t3,a4,a3
   20268:	010595b3          	sll	a1,a1,a6
   2026c:	00c8d8b3          	srl	a7,a7,a2
   20270:	0115e5b3          	or	a1,a1,a7
   20274:	00be2023          	sw	a1,0(t3)
   20278:	ffc70713          	addi	a4,a4,-4
   2027c:	eb5ff06f          	j	20130 <__subtf3+0x11f0>
   20280:	00062503          	lw	a0,0(a2)
   20284:	00168693          	addi	a3,a3,1
   20288:	00460613          	addi	a2,a2,4
   2028c:	00a36333          	or	t1,t1,a0
   20290:	f19ff06f          	j	201a8 <__subtf3+0x1268>
   20294:	04060593          	addi	a1,a2,64
   20298:	002585b3          	add	a1,a1,sp
   2029c:	ff05a583          	lw	a1,-16(a1)
   202a0:	02000813          	li	a6,32
   202a4:	40f80833          	sub	a6,a6,a5
   202a8:	010595b3          	sll	a1,a1,a6
   202ac:	00b36333          	or	t1,t1,a1
   202b0:	00000e13          	li	t3,0
   202b4:	00c885b3          	add	a1,a7,a2
   202b8:	40c00633          	neg	a2,a2
   202bc:	06de4863          	blt	t3,a3,2032c <__subtf3+0x13ec>
   202c0:	00400613          	li	a2,4
   202c4:	40e60733          	sub	a4,a2,a4
   202c8:	03c12603          	lw	a2,60(sp)
   202cc:	00269693          	slli	a3,a3,0x2
   202d0:	04068693          	addi	a3,a3,64
   202d4:	002686b3          	add	a3,a3,sp
   202d8:	00f657b3          	srl	a5,a2,a5
   202dc:	fef6a823          	sw	a5,-16(a3)
   202e0:	00400693          	li	a3,4
   202e4:	40e686b3          	sub	a3,a3,a4
   202e8:	00271713          	slli	a4,a4,0x2
   202ec:	00e887b3          	add	a5,a7,a4
   202f0:	00269713          	slli	a4,a3,0x2
   202f4:	00800693          	li	a3,8
   202f8:	00d76a63          	bltu	a4,a3,2030c <__subtf3+0x13cc>
   202fc:	0007a023          	sw	zero,0(a5)
   20300:	0007a223          	sw	zero,4(a5)
   20304:	ff870713          	addi	a4,a4,-8
   20308:	00878793          	addi	a5,a5,8
   2030c:	00400693          	li	a3,4
   20310:	00d76463          	bltu	a4,a3,20318 <__subtf3+0x13d8>
   20314:	0007a023          	sw	zero,0(a5)
   20318:	03012703          	lw	a4,48(sp)
   2031c:	006037b3          	snez	a5,t1
   20320:	00f767b3          	or	a5,a4,a5
   20324:	02f12823          	sw	a5,48(sp)
   20328:	c60ff06f          	j	1f788 <__subtf3+0x848>
   2032c:	0005a503          	lw	a0,0(a1)
   20330:	0045ae83          	lw	t4,4(a1)
   20334:	00c58f33          	add	t5,a1,a2
   20338:	00f55533          	srl	a0,a0,a5
   2033c:	010e9eb3          	sll	t4,t4,a6
   20340:	01d56533          	or	a0,a0,t4
   20344:	00af2023          	sw	a0,0(t5)
   20348:	001e0e13          	addi	t3,t3,1
   2034c:	00458593          	addi	a1,a1,4
   20350:	f6dff06f          	j	202bc <__subtf3+0x137c>
   20354:	03c12783          	lw	a5,60(sp)
   20358:	fff80737          	lui	a4,0xfff80
   2035c:	fff70713          	addi	a4,a4,-1 # fff7ffff <__BSS_END__+0xfff5d2cf>
   20360:	00e7f7b3          	and	a5,a5,a4
   20364:	40a484b3          	sub	s1,s1,a0
   20368:	02f12e23          	sw	a5,60(sp)
   2036c:	df1fe06f          	j	1f15c <__subtf3+0x21c>
   20370:	02012e23          	sw	zero,60(sp)
   20374:	02012c23          	sw	zero,56(sp)
   20378:	02012a23          	sw	zero,52(sp)
   2037c:	02012823          	sw	zero,48(sp)
   20380:	e5dfe06f          	j	1f1dc <__subtf3+0x29c>
   20384:	07400713          	li	a4,116
   20388:	00c74463          	blt	a4,a2,20390 <__subtf3+0x1450>
   2038c:	f51fe06f          	j	1f2dc <__subtf3+0x39c>
   20390:	02012623          	sw	zero,44(sp)
   20394:	02012423          	sw	zero,40(sp)
   20398:	02012223          	sw	zero,36(sp)
   2039c:	00100713          	li	a4,1
   203a0:	82cff06f          	j	1f3cc <__subtf3+0x48c>
   203a4:	07400793          	li	a5,116
   203a8:	a0b7de63          	bge	a5,a1,1f5c4 <__subtf3+0x684>
   203ac:	00012e23          	sw	zero,28(sp)
   203b0:	00012c23          	sw	zero,24(sp)
   203b4:	00012a23          	sw	zero,20(sp)
   203b8:	00100793          	li	a5,1
   203bc:	b00ff06f          	j	1f6bc <__subtf3+0x77c>
   203c0:	07400793          	li	a5,116
   203c4:	a717c4e3          	blt	a5,a7,1fe2c <__subtf3+0xeec>
   203c8:	00088313          	mv	t1,a7
   203cc:	8d5ff06f          	j	1fca0 <__subtf3+0xd60>

000203d0 <__unordtf2>:
   203d0:	00052703          	lw	a4,0(a0)
   203d4:	00452e83          	lw	t4,4(a0)
   203d8:	00852e03          	lw	t3,8(a0)
   203dc:	00c52503          	lw	a0,12(a0)
   203e0:	00c5a603          	lw	a2,12(a1)
   203e4:	000086b7          	lui	a3,0x8
   203e8:	fff68693          	addi	a3,a3,-1 # 7fff <exit-0x80b5>
   203ec:	01055813          	srli	a6,a0,0x10
   203f0:	0005a783          	lw	a5,0(a1)
   203f4:	0045a303          	lw	t1,4(a1)
   203f8:	0085a883          	lw	a7,8(a1)
   203fc:	00d87833          	and	a6,a6,a3
   20400:	01065593          	srli	a1,a2,0x10
   20404:	ff010113          	addi	sp,sp,-16
   20408:	00d5f5b3          	and	a1,a1,a3
   2040c:	02d81063          	bne	a6,a3,2042c <__unordtf2+0x5c>
   20410:	01d76733          	or	a4,a4,t4
   20414:	01051513          	slli	a0,a0,0x10
   20418:	01055513          	srli	a0,a0,0x10
   2041c:	01c76733          	or	a4,a4,t3
   20420:	00a76733          	or	a4,a4,a0
   20424:	00100513          	li	a0,1
   20428:	02071663          	bnez	a4,20454 <__unordtf2+0x84>
   2042c:	00008737          	lui	a4,0x8
   20430:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   20434:	00000513          	li	a0,0
   20438:	00e59e63          	bne	a1,a4,20454 <__unordtf2+0x84>
   2043c:	0067e533          	or	a0,a5,t1
   20440:	01061613          	slli	a2,a2,0x10
   20444:	01156533          	or	a0,a0,a7
   20448:	01065613          	srli	a2,a2,0x10
   2044c:	00c56533          	or	a0,a0,a2
   20450:	00a03533          	snez	a0,a0
   20454:	01010113          	addi	sp,sp,16
   20458:	00008067          	ret

0002045c <__fixtfsi>:
   2045c:	00052703          	lw	a4,0(a0)
   20460:	00452683          	lw	a3,4(a0)
   20464:	00c52783          	lw	a5,12(a0)
   20468:	00852583          	lw	a1,8(a0)
   2046c:	fe010113          	addi	sp,sp,-32
   20470:	00e12023          	sw	a4,0(sp)
   20474:	00d12223          	sw	a3,4(sp)
   20478:	00e12823          	sw	a4,16(sp)
   2047c:	00179693          	slli	a3,a5,0x1
   20480:	00004737          	lui	a4,0x4
   20484:	00b12423          	sw	a1,8(sp)
   20488:	00f12623          	sw	a5,12(sp)
   2048c:	00b12c23          	sw	a1,24(sp)
   20490:	0116d693          	srli	a3,a3,0x11
   20494:	ffe70713          	addi	a4,a4,-2 # 3ffe <exit-0xc0b6>
   20498:	00000513          	li	a0,0
   2049c:	02d75063          	bge	a4,a3,204bc <__fixtfsi+0x60>
   204a0:	00004737          	lui	a4,0x4
   204a4:	01d70713          	addi	a4,a4,29 # 401d <exit-0xc097>
   204a8:	01f7d813          	srli	a6,a5,0x1f
   204ac:	00d75c63          	bge	a4,a3,204c4 <__fixtfsi+0x68>
   204b0:	80000537          	lui	a0,0x80000
   204b4:	fff50513          	addi	a0,a0,-1 # 7fffffff <__BSS_END__+0x7ffdd2cf>
   204b8:	00a80533          	add	a0,a6,a0
   204bc:	02010113          	addi	sp,sp,32
   204c0:	00008067          	ret
   204c4:	01079793          	slli	a5,a5,0x10
   204c8:	00010737          	lui	a4,0x10
   204cc:	0107d793          	srli	a5,a5,0x10
   204d0:	00e7e7b3          	or	a5,a5,a4
   204d4:	00004737          	lui	a4,0x4
   204d8:	06f70713          	addi	a4,a4,111 # 406f <exit-0xc045>
   204dc:	40d70733          	sub	a4,a4,a3
   204e0:	40575613          	srai	a2,a4,0x5
   204e4:	00f12e23          	sw	a5,28(sp)
   204e8:	01f77713          	andi	a4,a4,31
   204ec:	02071463          	bnez	a4,20514 <__fixtfsi+0xb8>
   204f0:	00261613          	slli	a2,a2,0x2
   204f4:	02060793          	addi	a5,a2,32
   204f8:	00278633          	add	a2,a5,sp
   204fc:	ff062783          	lw	a5,-16(a2)
   20500:	00f12823          	sw	a5,16(sp)
   20504:	01012503          	lw	a0,16(sp)
   20508:	fa080ae3          	beqz	a6,204bc <__fixtfsi+0x60>
   2050c:	40a00533          	neg	a0,a0
   20510:	fadff06f          	j	204bc <__fixtfsi+0x60>
   20514:	00200513          	li	a0,2
   20518:	00000693          	li	a3,0
   2051c:	02a61063          	bne	a2,a0,2053c <__fixtfsi+0xe0>
   20520:	02000693          	li	a3,32
   20524:	40e686b3          	sub	a3,a3,a4
   20528:	00d796b3          	sll	a3,a5,a3
   2052c:	00e5d5b3          	srl	a1,a1,a4
   20530:	00b6e6b3          	or	a3,a3,a1
   20534:	00d12823          	sw	a3,16(sp)
   20538:	00100693          	li	a3,1
   2053c:	00269693          	slli	a3,a3,0x2
   20540:	02068693          	addi	a3,a3,32
   20544:	002686b3          	add	a3,a3,sp
   20548:	00e7d7b3          	srl	a5,a5,a4
   2054c:	fef6a823          	sw	a5,-16(a3)
   20550:	fb5ff06f          	j	20504 <__fixtfsi+0xa8>

00020554 <__floatsitf>:
   20554:	fd010113          	addi	sp,sp,-48
   20558:	02912223          	sw	s1,36(sp)
   2055c:	02112623          	sw	ra,44(sp)
   20560:	02812423          	sw	s0,40(sp)
   20564:	03212023          	sw	s2,32(sp)
   20568:	00050493          	mv	s1,a0
   2056c:	12058263          	beqz	a1,20690 <__floatsitf+0x13c>
   20570:	41f5d793          	srai	a5,a1,0x1f
   20574:	00b7c433          	xor	s0,a5,a1
   20578:	40f40433          	sub	s0,s0,a5
   2057c:	00040513          	mv	a0,s0
   20580:	01f5d913          	srli	s2,a1,0x1f
   20584:	330000ef          	jal	208b4 <__clzsi2>
   20588:	00004737          	lui	a4,0x4
   2058c:	01e70713          	addi	a4,a4,30 # 401e <exit-0xc096>
   20590:	05150793          	addi	a5,a0,81
   20594:	40a70633          	sub	a2,a4,a0
   20598:	00812823          	sw	s0,16(sp)
   2059c:	4057d713          	srai	a4,a5,0x5
   205a0:	00012a23          	sw	zero,20(sp)
   205a4:	00012c23          	sw	zero,24(sp)
   205a8:	00012e23          	sw	zero,28(sp)
   205ac:	01f7f793          	andi	a5,a5,31
   205b0:	02078c63          	beqz	a5,205e8 <__floatsitf+0x94>
   205b4:	00200693          	li	a3,2
   205b8:	0cd71863          	bne	a4,a3,20688 <__floatsitf+0x134>
   205bc:	02000693          	li	a3,32
   205c0:	40f686b3          	sub	a3,a3,a5
   205c4:	00d456b3          	srl	a3,s0,a3
   205c8:	00d12e23          	sw	a3,28(sp)
   205cc:	fff70693          	addi	a3,a4,-1
   205d0:	00271713          	slli	a4,a4,0x2
   205d4:	02070713          	addi	a4,a4,32
   205d8:	00270733          	add	a4,a4,sp
   205dc:	00f41433          	sll	s0,s0,a5
   205e0:	fe872823          	sw	s0,-16(a4)
   205e4:	0340006f          	j	20618 <__floatsitf+0xc4>
   205e8:	00300793          	li	a5,3
   205ec:	40e787b3          	sub	a5,a5,a4
   205f0:	00279793          	slli	a5,a5,0x2
   205f4:	02078793          	addi	a5,a5,32
   205f8:	002787b3          	add	a5,a5,sp
   205fc:	ff07a783          	lw	a5,-16(a5)
   20600:	00200693          	li	a3,2
   20604:	00f12e23          	sw	a5,28(sp)
   20608:	00200793          	li	a5,2
   2060c:	00f71663          	bne	a4,a5,20618 <__floatsitf+0xc4>
   20610:	00812c23          	sw	s0,24(sp)
   20614:	00100693          	li	a3,1
   20618:	00269693          	slli	a3,a3,0x2
   2061c:	00012823          	sw	zero,16(sp)
   20620:	00012a23          	sw	zero,20(sp)
   20624:	ffc68693          	addi	a3,a3,-4
   20628:	00400793          	li	a5,4
   2062c:	00f6e463          	bltu	a3,a5,20634 <__floatsitf+0xe0>
   20630:	00012c23          	sw	zero,24(sp)
   20634:	00090593          	mv	a1,s2
   20638:	01c12783          	lw	a5,28(sp)
   2063c:	00f59413          	slli	s0,a1,0xf
   20640:	00c46433          	or	s0,s0,a2
   20644:	00f11623          	sh	a5,12(sp)
   20648:	01012783          	lw	a5,16(sp)
   2064c:	00811723          	sh	s0,14(sp)
   20650:	02c12083          	lw	ra,44(sp)
   20654:	00f4a023          	sw	a5,0(s1)
   20658:	01412783          	lw	a5,20(sp)
   2065c:	02812403          	lw	s0,40(sp)
   20660:	02012903          	lw	s2,32(sp)
   20664:	00f4a223          	sw	a5,4(s1)
   20668:	01812783          	lw	a5,24(sp)
   2066c:	00048513          	mv	a0,s1
   20670:	00f4a423          	sw	a5,8(s1)
   20674:	00c12783          	lw	a5,12(sp)
   20678:	00f4a623          	sw	a5,12(s1)
   2067c:	02412483          	lw	s1,36(sp)
   20680:	03010113          	addi	sp,sp,48
   20684:	00008067          	ret
   20688:	00300713          	li	a4,3
   2068c:	f41ff06f          	j	205cc <__floatsitf+0x78>
   20690:	00012e23          	sw	zero,28(sp)
   20694:	00012c23          	sw	zero,24(sp)
   20698:	00012a23          	sw	zero,20(sp)
   2069c:	00012823          	sw	zero,16(sp)
   206a0:	00000613          	li	a2,0
   206a4:	f95ff06f          	j	20638 <__floatsitf+0xe4>

000206a8 <__extenddftf2>:
   206a8:	01465713          	srli	a4,a2,0x14
   206ac:	00c61793          	slli	a5,a2,0xc
   206b0:	7ff77713          	andi	a4,a4,2047
   206b4:	fd010113          	addi	sp,sp,-48
   206b8:	00c7d793          	srli	a5,a5,0xc
   206bc:	00170693          	addi	a3,a4,1
   206c0:	02812423          	sw	s0,40(sp)
   206c4:	02912223          	sw	s1,36(sp)
   206c8:	03212023          	sw	s2,32(sp)
   206cc:	02112623          	sw	ra,44(sp)
   206d0:	00b12823          	sw	a1,16(sp)
   206d4:	00f12a23          	sw	a5,20(sp)
   206d8:	00012e23          	sw	zero,28(sp)
   206dc:	00012c23          	sw	zero,24(sp)
   206e0:	7fe6f693          	andi	a3,a3,2046
   206e4:	00050913          	mv	s2,a0
   206e8:	00058413          	mv	s0,a1
   206ec:	01f65493          	srli	s1,a2,0x1f
   206f0:	08068263          	beqz	a3,20774 <__extenddftf2+0xcc>
   206f4:	000046b7          	lui	a3,0x4
   206f8:	c0068693          	addi	a3,a3,-1024 # 3c00 <exit-0xc4b4>
   206fc:	00d70733          	add	a4,a4,a3
   20700:	0047d693          	srli	a3,a5,0x4
   20704:	00d12e23          	sw	a3,28(sp)
   20708:	01c79793          	slli	a5,a5,0x1c
   2070c:	0045d693          	srli	a3,a1,0x4
   20710:	00d7e7b3          	or	a5,a5,a3
   20714:	01c59413          	slli	s0,a1,0x1c
   20718:	00f12c23          	sw	a5,24(sp)
   2071c:	00812a23          	sw	s0,20(sp)
   20720:	00012823          	sw	zero,16(sp)
   20724:	01c12783          	lw	a5,28(sp)
   20728:	00f49493          	slli	s1,s1,0xf
   2072c:	00e4e4b3          	or	s1,s1,a4
   20730:	00f11623          	sh	a5,12(sp)
   20734:	01012783          	lw	a5,16(sp)
   20738:	00911723          	sh	s1,14(sp)
   2073c:	02c12083          	lw	ra,44(sp)
   20740:	00f92023          	sw	a5,0(s2)
   20744:	01412783          	lw	a5,20(sp)
   20748:	02812403          	lw	s0,40(sp)
   2074c:	02412483          	lw	s1,36(sp)
   20750:	00f92223          	sw	a5,4(s2)
   20754:	01812783          	lw	a5,24(sp)
   20758:	00090513          	mv	a0,s2
   2075c:	00f92423          	sw	a5,8(s2)
   20760:	00c12783          	lw	a5,12(sp)
   20764:	00f92623          	sw	a5,12(s2)
   20768:	02012903          	lw	s2,32(sp)
   2076c:	03010113          	addi	sp,sp,48
   20770:	00008067          	ret
   20774:	00b7e533          	or	a0,a5,a1
   20778:	10071063          	bnez	a4,20878 <__extenddftf2+0x1d0>
   2077c:	fa0504e3          	beqz	a0,20724 <__extenddftf2+0x7c>
   20780:	04078e63          	beqz	a5,207dc <__extenddftf2+0x134>
   20784:	00078513          	mv	a0,a5
   20788:	12c000ef          	jal	208b4 <__clzsi2>
   2078c:	03150693          	addi	a3,a0,49
   20790:	4056d793          	srai	a5,a3,0x5
   20794:	01f6f693          	andi	a3,a3,31
   20798:	04068863          	beqz	a3,207e8 <__extenddftf2+0x140>
   2079c:	ffc00613          	li	a2,-4
   207a0:	02c78633          	mul	a2,a5,a2
   207a4:	02000813          	li	a6,32
   207a8:	01010313          	addi	t1,sp,16
   207ac:	40d80833          	sub	a6,a6,a3
   207b0:	00c60713          	addi	a4,a2,12
   207b4:	00e30733          	add	a4,t1,a4
   207b8:	40c00633          	neg	a2,a2
   207bc:	08e31c63          	bne	t1,a4,20854 <__extenddftf2+0x1ac>
   207c0:	fff78713          	addi	a4,a5,-1
   207c4:	00279793          	slli	a5,a5,0x2
   207c8:	02078793          	addi	a5,a5,32
   207cc:	002787b3          	add	a5,a5,sp
   207d0:	00d416b3          	sll	a3,s0,a3
   207d4:	fed7a823          	sw	a3,-16(a5)
   207d8:	03c0006f          	j	20814 <__extenddftf2+0x16c>
   207dc:	0d8000ef          	jal	208b4 <__clzsi2>
   207e0:	02050513          	addi	a0,a0,32
   207e4:	fa9ff06f          	j	2078c <__extenddftf2+0xe4>
   207e8:	ffc00613          	li	a2,-4
   207ec:	02c78633          	mul	a2,a5,a2
   207f0:	01c10713          	addi	a4,sp,28
   207f4:	00300693          	li	a3,3
   207f8:	00c705b3          	add	a1,a4,a2
   207fc:	0005a583          	lw	a1,0(a1)
   20800:	fff68693          	addi	a3,a3,-1
   20804:	ffc70713          	addi	a4,a4,-4
   20808:	00b72223          	sw	a1,4(a4)
   2080c:	fef6d6e3          	bge	a3,a5,207f8 <__extenddftf2+0x150>
   20810:	fff78713          	addi	a4,a5,-1
   20814:	00170793          	addi	a5,a4,1
   20818:	00279793          	slli	a5,a5,0x2
   2081c:	00800693          	li	a3,8
   20820:	01010713          	addi	a4,sp,16
   20824:	00d7ea63          	bltu	a5,a3,20838 <__extenddftf2+0x190>
   20828:	00012823          	sw	zero,16(sp)
   2082c:	00072223          	sw	zero,4(a4)
   20830:	ff878793          	addi	a5,a5,-8
   20834:	01810713          	addi	a4,sp,24
   20838:	00400693          	li	a3,4
   2083c:	00d7e463          	bltu	a5,a3,20844 <__extenddftf2+0x19c>
   20840:	00072023          	sw	zero,0(a4)
   20844:	00004737          	lui	a4,0x4
   20848:	c0c70713          	addi	a4,a4,-1012 # 3c0c <exit-0xc4a8>
   2084c:	40a70733          	sub	a4,a4,a0
   20850:	ed5ff06f          	j	20724 <__extenddftf2+0x7c>
   20854:	00072583          	lw	a1,0(a4)
   20858:	ffc72883          	lw	a7,-4(a4)
   2085c:	00c70e33          	add	t3,a4,a2
   20860:	00d595b3          	sll	a1,a1,a3
   20864:	0108d8b3          	srl	a7,a7,a6
   20868:	0115e5b3          	or	a1,a1,a7
   2086c:	00be2023          	sw	a1,0(t3)
   20870:	ffc70713          	addi	a4,a4,-4
   20874:	f49ff06f          	j	207bc <__extenddftf2+0x114>
   20878:	02050863          	beqz	a0,208a8 <__extenddftf2+0x200>
   2087c:	01c79713          	slli	a4,a5,0x1c
   20880:	0045d693          	srli	a3,a1,0x4
   20884:	00d76733          	or	a4,a4,a3
   20888:	00e12c23          	sw	a4,24(sp)
   2088c:	0047d793          	srli	a5,a5,0x4
   20890:	00008737          	lui	a4,0x8
   20894:	01c59413          	slli	s0,a1,0x1c
   20898:	00e7e7b3          	or	a5,a5,a4
   2089c:	00812a23          	sw	s0,20(sp)
   208a0:	00012823          	sw	zero,16(sp)
   208a4:	00f12e23          	sw	a5,28(sp)
   208a8:	00008737          	lui	a4,0x8
   208ac:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   208b0:	e75ff06f          	j	20724 <__extenddftf2+0x7c>

000208b4 <__clzsi2>:
   208b4:	000107b7          	lui	a5,0x10
   208b8:	02f57a63          	bgeu	a0,a5,208ec <__clzsi2+0x38>
   208bc:	10053793          	sltiu	a5,a0,256
   208c0:	0017b793          	seqz	a5,a5
   208c4:	00379793          	slli	a5,a5,0x3
   208c8:	02000713          	li	a4,32
   208cc:	40f70733          	sub	a4,a4,a5
   208d0:	00f55533          	srl	a0,a0,a5
   208d4:	00001797          	auipc	a5,0x1
   208d8:	a4478793          	addi	a5,a5,-1468 # 21318 <__clz_tab>
   208dc:	00a787b3          	add	a5,a5,a0
   208e0:	0007c503          	lbu	a0,0(a5)
   208e4:	40a70533          	sub	a0,a4,a0
   208e8:	00008067          	ret
   208ec:	01000737          	lui	a4,0x1000
   208f0:	01800793          	li	a5,24
   208f4:	fce57ae3          	bgeu	a0,a4,208c8 <__clzsi2+0x14>
   208f8:	01000793          	li	a5,16
   208fc:	fcdff06f          	j	208c8 <__clzsi2+0x14>

00020900 <_close>:
   20900:	05800793          	li	a5,88
   20904:	08f1ae23          	sw	a5,156(gp) # 2291c <errno>
   20908:	fff00513          	li	a0,-1
   2090c:	00008067          	ret

00020910 <_fstat>:
   20910:	05800793          	li	a5,88
   20914:	08f1ae23          	sw	a5,156(gp) # 2291c <errno>
   20918:	fff00513          	li	a0,-1
   2091c:	00008067          	ret

00020920 <_getpid>:
   20920:	05800793          	li	a5,88
   20924:	08f1ae23          	sw	a5,156(gp) # 2291c <errno>
   20928:	fff00513          	li	a0,-1
   2092c:	00008067          	ret

00020930 <_isatty>:
   20930:	05800793          	li	a5,88
   20934:	08f1ae23          	sw	a5,156(gp) # 2291c <errno>
   20938:	00000513          	li	a0,0
   2093c:	00008067          	ret

00020940 <_kill>:
   20940:	05800793          	li	a5,88
   20944:	08f1ae23          	sw	a5,156(gp) # 2291c <errno>
   20948:	fff00513          	li	a0,-1
   2094c:	00008067          	ret

00020950 <_lseek>:
   20950:	05800793          	li	a5,88
   20954:	08f1ae23          	sw	a5,156(gp) # 2291c <errno>
   20958:	fff00513          	li	a0,-1
   2095c:	00008067          	ret

00020960 <_read>:
   20960:	05800793          	li	a5,88
   20964:	08f1ae23          	sw	a5,156(gp) # 2291c <errno>
   20968:	fff00513          	li	a0,-1
   2096c:	00008067          	ret

00020970 <_sbrk>:
   20970:	0b418713          	addi	a4,gp,180 # 22934 <heap_end.0>
   20974:	00072783          	lw	a5,0(a4) # 1000000 <__BSS_END__+0xfdd2d0>
   20978:	00078a63          	beqz	a5,2098c <_sbrk+0x1c>
   2097c:	00a78533          	add	a0,a5,a0
   20980:	00a72023          	sw	a0,0(a4)
   20984:	00078513          	mv	a0,a5
   20988:	00008067          	ret
   2098c:	4b018793          	addi	a5,gp,1200 # 22d30 <__BSS_END__>
   20990:	00a78533          	add	a0,a5,a0
   20994:	00a72023          	sw	a0,0(a4)
   20998:	00078513          	mv	a0,a5
   2099c:	00008067          	ret

000209a0 <_write>:
   209a0:	05800793          	li	a5,88
   209a4:	08f1ae23          	sw	a5,156(gp) # 2291c <errno>
   209a8:	fff00513          	li	a0,-1
   209ac:	00008067          	ret

000209b0 <_exit>:
   209b0:	0000006f          	j	209b0 <_exit>

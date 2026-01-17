
/home/cloud/aps-mlir/examples/diff_match/v3ddist_vs/v3ddist_vs.out:     file format elf32-littleriscv


Disassembly of section .text:

000100b4 <exit>:
   100b4:	ff010113          	addi	sp,sp,-16
   100b8:	00000593          	li	a1,0
   100bc:	00812423          	sw	s0,8(sp)
   100c0:	00112623          	sw	ra,12(sp)
   100c4:	00050413          	mv	s0,a0
   100c8:	2ed000ef          	jal	10bb4 <__call_exitprocs>
   100cc:	e481a783          	lw	a5,-440(gp) # 126c8 <__stdio_exit_handler>
   100d0:	00078463          	beqz	a5,100d8 <exit+0x24>
   100d4:	000780e7          	jalr	a5
   100d8:	00040513          	mv	a0,s0
   100dc:	625010ef          	jal	11f00 <_exit>

000100e0 <register_fini>:
   100e0:	00000793          	li	a5,0
   100e4:	00078863          	beqz	a5,100f4 <register_fini+0x14>
   100e8:	00002517          	auipc	a0,0x2
   100ec:	cb450513          	addi	a0,a0,-844 # 11d9c <__libc_fini_array>
   100f0:	3fd0006f          	j	10cec <atexit>
   100f4:	00008067          	ret

000100f8 <_start>:
   100f8:	00002197          	auipc	gp,0x2
   100fc:	78818193          	addi	gp,gp,1928 # 12880 <__global_pointer$>
   10100:	e4818513          	addi	a0,gp,-440 # 126c8 <__stdio_exit_handler>
   10104:	23018613          	addi	a2,gp,560 # 12ab0 <__BSS_END__>
   10108:	40a60633          	sub	a2,a2,a0
   1010c:	00000593          	li	a1,0
   10110:	1c9000ef          	jal	10ad8 <memset>
   10114:	00001517          	auipc	a0,0x1
   10118:	bd850513          	addi	a0,a0,-1064 # 10cec <atexit>
   1011c:	00050863          	beqz	a0,1012c <_start+0x34>
   10120:	00002517          	auipc	a0,0x2
   10124:	c7c50513          	addi	a0,a0,-900 # 11d9c <__libc_fini_array>
   10128:	3c5000ef          	jal	10cec <atexit>
   1012c:	119000ef          	jal	10a44 <__libc_init_array>
   10130:	00012503          	lw	a0,0(sp)
   10134:	00410593          	addi	a1,sp,4
   10138:	00000613          	li	a2,0
   1013c:	12c000ef          	jal	10268 <main>
   10140:	f75ff06f          	j	100b4 <exit>

00010144 <__do_global_dtors_aux>:
   10144:	ff010113          	addi	sp,sp,-16
   10148:	00812423          	sw	s0,8(sp)
   1014c:	e8018413          	addi	s0,gp,-384 # 12700 <completed.1>
   10150:	00044783          	lbu	a5,0(s0)
   10154:	00112623          	sw	ra,12(sp)
   10158:	02079263          	bnez	a5,1017c <__do_global_dtors_aux+0x38>
   1015c:	00000793          	li	a5,0
   10160:	00078a63          	beqz	a5,10174 <__do_global_dtors_aux+0x30>
   10164:	00002517          	auipc	a0,0x2
   10168:	e9c50513          	addi	a0,a0,-356 # 12000 <__EH_FRAME_BEGIN__>
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
   10194:	e8418593          	addi	a1,gp,-380 # 12704 <object.0>
   10198:	00002517          	auipc	a0,0x2
   1019c:	e6850513          	addi	a0,a0,-408 # 12000 <__EH_FRAME_BEGIN__>
   101a0:	00000317          	auipc	t1,0x0
   101a4:	00000067          	jr	zero # 0 <exit-0x100b4>
   101a8:	00008067          	ret

000101ac <v3ddist_vs>:
   101ac:	00000513          	li	a0,0
   101b0:	00412603          	lw	a2,4(sp)
   101b4:	00000693          	li	a3,0
   101b8:	0100006f          	j	101c8 <v3ddist_vs+0x1c>
   101bc:	00d02733          	sgtz	a4,a3
   101c0:	00070a63          	beqz	a4,101d4 <v3ddist_vs+0x28>
   101c4:	06c0006f          	j	10230 <v3ddist_vs+0x84>
   101c8:	01053713          	sltiu	a4,a0,16
   101cc:	00174713          	xori	a4,a4,1
   101d0:	06071063          	bnez	a4,10230 <v3ddist_vs+0x84>
   101d4:	00251713          	slli	a4,a0,0x2
   101d8:	0c05a783          	lw	a5,192(a1)
   101dc:	00e58833          	add	a6,a1,a4
   101e0:	00082883          	lw	a7,0(a6)
   101e4:	04082283          	lw	t0,64(a6)
   101e8:	0c45a303          	lw	t1,196(a1)
   101ec:	0c85a383          	lw	t2,200(a1)
   101f0:	40f887b3          	sub	a5,a7,a5
   101f4:	08082803          	lw	a6,128(a6)
   101f8:	406288b3          	sub	a7,t0,t1
   101fc:	00150513          	addi	a0,a0,1
   10200:	00e60733          	add	a4,a2,a4
   10204:	40780833          	sub	a6,a6,t2
   10208:	00153293          	seqz	t0,a0
   1020c:	02f787b3          	mul	a5,a5,a5
   10210:	031888b3          	mul	a7,a7,a7
   10214:	03080833          	mul	a6,a6,a6
   10218:	01088833          	add	a6,a7,a6
   1021c:	010787b3          	add	a5,a5,a6
   10220:	00f72023          	sw	a5,0(a4)
   10224:	005686b3          	add	a3,a3,t0
   10228:	fa0680e3          	beqz	a3,101c8 <v3ddist_vs+0x1c>
   1022c:	f91ff06f          	j	101bc <v3ddist_vs+0x10>
   10230:	00000513          	li	a0,0
   10234:	00008067          	ret

00010238 <get_march>:
   10238:	fff50513          	addi	a0,a0,-1
   1023c:	00400593          	li	a1,4
   10240:	00a5ee63          	bltu	a1,a0,1025c <get_march+0x24>
   10244:	00251513          	slli	a0,a0,0x2
   10248:	00002597          	auipc	a1,0x2
   1024c:	cdc58593          	addi	a1,a1,-804 # 11f24 <_exit+0x24>
   10250:	00a58533          	add	a0,a1,a0
   10254:	00052503          	lw	a0,0(a0)
   10258:	00008067          	ret
   1025c:	00002517          	auipc	a0,0x2
   10260:	cbf50513          	addi	a0,a0,-833 # 11f1b <_exit+0x1b>
   10264:	00008067          	ret

00010268 <main>:
   10268:	ff010113          	addi	sp,sp,-16
   1026c:	00112623          	sw	ra,12(sp)
   10270:	f1202573          	.insn	4, 0xf1202573
   10274:	00012423          	sw	zero,8(sp)
   10278:	0ff0000f          	fence
   1027c:	00002517          	auipc	a0,0x2
   10280:	e0450513          	addi	a0,a0,-508 # 12080 <input_data>
   10284:	f0018593          	addi	a1,gp,-256 # 12780 <output_data>
   10288:	f25ff0ef          	jal	101ac <v3ddist_vs>
   1028c:	00a12423          	sw	a0,8(sp)
   10290:	0ff0000f          	fence
   10294:	00000513          	li	a0,0
   10298:	00c12083          	lw	ra,12(sp)
   1029c:	01010113          	addi	sp,sp,16
   102a0:	00008067          	ret

000102a4 <__fp_lock>:
   102a4:	00000513          	li	a0,0
   102a8:	00008067          	ret

000102ac <stdio_exit_handler>:
   102ac:	90018613          	addi	a2,gp,-1792 # 12180 <__sglue>
   102b0:	00001597          	auipc	a1,0x1
   102b4:	65c58593          	addi	a1,a1,1628 # 1190c <_fclose_r>
   102b8:	00002517          	auipc	a0,0x2
   102bc:	ed850513          	addi	a0,a0,-296 # 12190 <_impure_data>
   102c0:	33c0006f          	j	105fc <_fwalk_sglue>

000102c4 <cleanup_stdio>:
   102c4:	00452583          	lw	a1,4(a0)
   102c8:	ff010113          	addi	sp,sp,-16
   102cc:	00812423          	sw	s0,8(sp)
   102d0:	00112623          	sw	ra,12(sp)
   102d4:	f4018793          	addi	a5,gp,-192 # 127c0 <__sf>
   102d8:	00050413          	mv	s0,a0
   102dc:	00f58463          	beq	a1,a5,102e4 <cleanup_stdio+0x20>
   102e0:	62c010ef          	jal	1190c <_fclose_r>
   102e4:	00842583          	lw	a1,8(s0)
   102e8:	fa818793          	addi	a5,gp,-88 # 12828 <__sf+0x68>
   102ec:	00f58663          	beq	a1,a5,102f8 <cleanup_stdio+0x34>
   102f0:	00040513          	mv	a0,s0
   102f4:	618010ef          	jal	1190c <_fclose_r>
   102f8:	00c42583          	lw	a1,12(s0)
   102fc:	01018793          	addi	a5,gp,16 # 12890 <__global_pointer$+0x10>
   10300:	00f58c63          	beq	a1,a5,10318 <cleanup_stdio+0x54>
   10304:	00040513          	mv	a0,s0
   10308:	00812403          	lw	s0,8(sp)
   1030c:	00c12083          	lw	ra,12(sp)
   10310:	01010113          	addi	sp,sp,16
   10314:	5f80106f          	j	1190c <_fclose_r>
   10318:	00c12083          	lw	ra,12(sp)
   1031c:	00812403          	lw	s0,8(sp)
   10320:	01010113          	addi	sp,sp,16
   10324:	00008067          	ret

00010328 <__fp_unlock>:
   10328:	00000513          	li	a0,0
   1032c:	00008067          	ret

00010330 <global_stdio_init.part.0>:
   10330:	fe010113          	addi	sp,sp,-32
   10334:	00000797          	auipc	a5,0x0
   10338:	f7878793          	addi	a5,a5,-136 # 102ac <stdio_exit_handler>
   1033c:	00112e23          	sw	ra,28(sp)
   10340:	00812c23          	sw	s0,24(sp)
   10344:	00912a23          	sw	s1,20(sp)
   10348:	f4018413          	addi	s0,gp,-192 # 127c0 <__sf>
   1034c:	01212823          	sw	s2,16(sp)
   10350:	01312623          	sw	s3,12(sp)
   10354:	01412423          	sw	s4,8(sp)
   10358:	e4f1a423          	sw	a5,-440(gp) # 126c8 <__stdio_exit_handler>
   1035c:	00800613          	li	a2,8
   10360:	00400793          	li	a5,4
   10364:	00000593          	li	a1,0
   10368:	f9c18513          	addi	a0,gp,-100 # 1281c <__sf+0x5c>
   1036c:	00f42623          	sw	a5,12(s0)
   10370:	00042023          	sw	zero,0(s0)
   10374:	00042223          	sw	zero,4(s0)
   10378:	00042423          	sw	zero,8(s0)
   1037c:	06042223          	sw	zero,100(s0)
   10380:	00042823          	sw	zero,16(s0)
   10384:	00042a23          	sw	zero,20(s0)
   10388:	00042c23          	sw	zero,24(s0)
   1038c:	74c000ef          	jal	10ad8 <memset>
   10390:	000107b7          	lui	a5,0x10
   10394:	00000a17          	auipc	s4,0x0
   10398:	31ca0a13          	addi	s4,s4,796 # 106b0 <__sread>
   1039c:	00000997          	auipc	s3,0x0
   103a0:	37898993          	addi	s3,s3,888 # 10714 <__swrite>
   103a4:	00000917          	auipc	s2,0x0
   103a8:	3f890913          	addi	s2,s2,1016 # 1079c <__sseek>
   103ac:	00000497          	auipc	s1,0x0
   103b0:	46848493          	addi	s1,s1,1128 # 10814 <__sclose>
   103b4:	00978793          	addi	a5,a5,9 # 10009 <exit-0xab>
   103b8:	00800613          	li	a2,8
   103bc:	00000593          	li	a1,0
   103c0:	00418513          	addi	a0,gp,4 # 12884 <__global_pointer$+0x4>
   103c4:	03442023          	sw	s4,32(s0)
   103c8:	03342223          	sw	s3,36(s0)
   103cc:	03242423          	sw	s2,40(s0)
   103d0:	02942623          	sw	s1,44(s0)
   103d4:	06f42a23          	sw	a5,116(s0)
   103d8:	00842e23          	sw	s0,28(s0)
   103dc:	06042423          	sw	zero,104(s0)
   103e0:	06042623          	sw	zero,108(s0)
   103e4:	06042823          	sw	zero,112(s0)
   103e8:	0c042623          	sw	zero,204(s0)
   103ec:	06042c23          	sw	zero,120(s0)
   103f0:	06042e23          	sw	zero,124(s0)
   103f4:	08042023          	sw	zero,128(s0)
   103f8:	6e0000ef          	jal	10ad8 <memset>
   103fc:	000207b7          	lui	a5,0x20
   10400:	01278793          	addi	a5,a5,18 # 20012 <__BSS_END__+0xd562>
   10404:	fa818713          	addi	a4,gp,-88 # 12828 <__sf+0x68>
   10408:	00800613          	li	a2,8
   1040c:	00000593          	li	a1,0
   10410:	06c18513          	addi	a0,gp,108 # 128ec <__global_pointer$+0x6c>
   10414:	09442423          	sw	s4,136(s0)
   10418:	09342623          	sw	s3,140(s0)
   1041c:	09242823          	sw	s2,144(s0)
   10420:	08942a23          	sw	s1,148(s0)
   10424:	0cf42e23          	sw	a5,220(s0)
   10428:	08e42223          	sw	a4,132(s0)
   1042c:	0c042823          	sw	zero,208(s0)
   10430:	0c042a23          	sw	zero,212(s0)
   10434:	0c042c23          	sw	zero,216(s0)
   10438:	12042a23          	sw	zero,308(s0)
   1043c:	0e042023          	sw	zero,224(s0)
   10440:	0e042223          	sw	zero,228(s0)
   10444:	0e042423          	sw	zero,232(s0)
   10448:	690000ef          	jal	10ad8 <memset>
   1044c:	01018793          	addi	a5,gp,16 # 12890 <__global_pointer$+0x10>
   10450:	0f442823          	sw	s4,240(s0)
   10454:	0f342a23          	sw	s3,244(s0)
   10458:	0f242c23          	sw	s2,248(s0)
   1045c:	0e942e23          	sw	s1,252(s0)
   10460:	01c12083          	lw	ra,28(sp)
   10464:	0ef42623          	sw	a5,236(s0)
   10468:	01812403          	lw	s0,24(sp)
   1046c:	01412483          	lw	s1,20(sp)
   10470:	01012903          	lw	s2,16(sp)
   10474:	00c12983          	lw	s3,12(sp)
   10478:	00812a03          	lw	s4,8(sp)
   1047c:	02010113          	addi	sp,sp,32
   10480:	00008067          	ret

00010484 <__sfp>:
   10484:	fe010113          	addi	sp,sp,-32
   10488:	01312623          	sw	s3,12(sp)
   1048c:	00112e23          	sw	ra,28(sp)
   10490:	00812c23          	sw	s0,24(sp)
   10494:	00912a23          	sw	s1,20(sp)
   10498:	01212823          	sw	s2,16(sp)
   1049c:	e481a783          	lw	a5,-440(gp) # 126c8 <__stdio_exit_handler>
   104a0:	00050993          	mv	s3,a0
   104a4:	0e078663          	beqz	a5,10590 <__sfp+0x10c>
   104a8:	90018913          	addi	s2,gp,-1792 # 12180 <__sglue>
   104ac:	fff00493          	li	s1,-1
   104b0:	00492783          	lw	a5,4(s2)
   104b4:	00892403          	lw	s0,8(s2)
   104b8:	fff78793          	addi	a5,a5,-1
   104bc:	0007d863          	bgez	a5,104cc <__sfp+0x48>
   104c0:	0800006f          	j	10540 <__sfp+0xbc>
   104c4:	06840413          	addi	s0,s0,104
   104c8:	06978c63          	beq	a5,s1,10540 <__sfp+0xbc>
   104cc:	00c41703          	lh	a4,12(s0)
   104d0:	fff78793          	addi	a5,a5,-1
   104d4:	fe0718e3          	bnez	a4,104c4 <__sfp+0x40>
   104d8:	ffff07b7          	lui	a5,0xffff0
   104dc:	00178793          	addi	a5,a5,1 # ffff0001 <__BSS_END__+0xfffdd551>
   104e0:	00f42623          	sw	a5,12(s0)
   104e4:	06042223          	sw	zero,100(s0)
   104e8:	00042023          	sw	zero,0(s0)
   104ec:	00042423          	sw	zero,8(s0)
   104f0:	00042223          	sw	zero,4(s0)
   104f4:	00042823          	sw	zero,16(s0)
   104f8:	00042a23          	sw	zero,20(s0)
   104fc:	00042c23          	sw	zero,24(s0)
   10500:	00800613          	li	a2,8
   10504:	00000593          	li	a1,0
   10508:	05c40513          	addi	a0,s0,92
   1050c:	5cc000ef          	jal	10ad8 <memset>
   10510:	02042823          	sw	zero,48(s0)
   10514:	02042a23          	sw	zero,52(s0)
   10518:	04042223          	sw	zero,68(s0)
   1051c:	04042423          	sw	zero,72(s0)
   10520:	01c12083          	lw	ra,28(sp)
   10524:	00040513          	mv	a0,s0
   10528:	01812403          	lw	s0,24(sp)
   1052c:	01412483          	lw	s1,20(sp)
   10530:	01012903          	lw	s2,16(sp)
   10534:	00c12983          	lw	s3,12(sp)
   10538:	02010113          	addi	sp,sp,32
   1053c:	00008067          	ret
   10540:	00092403          	lw	s0,0(s2)
   10544:	00040663          	beqz	s0,10550 <__sfp+0xcc>
   10548:	00040913          	mv	s2,s0
   1054c:	f65ff06f          	j	104b0 <__sfp+0x2c>
   10550:	1ac00593          	li	a1,428
   10554:	00098513          	mv	a0,s3
   10558:	3e5000ef          	jal	1113c <_malloc_r>
   1055c:	00050413          	mv	s0,a0
   10560:	02050c63          	beqz	a0,10598 <__sfp+0x114>
   10564:	00c50513          	addi	a0,a0,12
   10568:	00400793          	li	a5,4
   1056c:	00042023          	sw	zero,0(s0)
   10570:	00f42223          	sw	a5,4(s0)
   10574:	00a42423          	sw	a0,8(s0)
   10578:	1a000613          	li	a2,416
   1057c:	00000593          	li	a1,0
   10580:	558000ef          	jal	10ad8 <memset>
   10584:	00892023          	sw	s0,0(s2)
   10588:	00040913          	mv	s2,s0
   1058c:	f25ff06f          	j	104b0 <__sfp+0x2c>
   10590:	da1ff0ef          	jal	10330 <global_stdio_init.part.0>
   10594:	f15ff06f          	j	104a8 <__sfp+0x24>
   10598:	00092023          	sw	zero,0(s2)
   1059c:	00c00793          	li	a5,12
   105a0:	00f9a023          	sw	a5,0(s3)
   105a4:	f7dff06f          	j	10520 <__sfp+0x9c>

000105a8 <__sinit>:
   105a8:	03452783          	lw	a5,52(a0)
   105ac:	00078463          	beqz	a5,105b4 <__sinit+0xc>
   105b0:	00008067          	ret
   105b4:	00000797          	auipc	a5,0x0
   105b8:	d1078793          	addi	a5,a5,-752 # 102c4 <cleanup_stdio>
   105bc:	02f52a23          	sw	a5,52(a0)
   105c0:	e481a783          	lw	a5,-440(gp) # 126c8 <__stdio_exit_handler>
   105c4:	fe0796e3          	bnez	a5,105b0 <__sinit+0x8>
   105c8:	d69ff06f          	j	10330 <global_stdio_init.part.0>

000105cc <__sfp_lock_acquire>:
   105cc:	00008067          	ret

000105d0 <__sfp_lock_release>:
   105d0:	00008067          	ret

000105d4 <__fp_lock_all>:
   105d4:	90018613          	addi	a2,gp,-1792 # 12180 <__sglue>
   105d8:	00000597          	auipc	a1,0x0
   105dc:	ccc58593          	addi	a1,a1,-820 # 102a4 <__fp_lock>
   105e0:	00000513          	li	a0,0
   105e4:	0180006f          	j	105fc <_fwalk_sglue>

000105e8 <__fp_unlock_all>:
   105e8:	90018613          	addi	a2,gp,-1792 # 12180 <__sglue>
   105ec:	00000597          	auipc	a1,0x0
   105f0:	d3c58593          	addi	a1,a1,-708 # 10328 <__fp_unlock>
   105f4:	00000513          	li	a0,0
   105f8:	0040006f          	j	105fc <_fwalk_sglue>

000105fc <_fwalk_sglue>:
   105fc:	fd010113          	addi	sp,sp,-48
   10600:	03212023          	sw	s2,32(sp)
   10604:	01312e23          	sw	s3,28(sp)
   10608:	01412c23          	sw	s4,24(sp)
   1060c:	01512a23          	sw	s5,20(sp)
   10610:	01612823          	sw	s6,16(sp)
   10614:	01712623          	sw	s7,12(sp)
   10618:	02112623          	sw	ra,44(sp)
   1061c:	02812423          	sw	s0,40(sp)
   10620:	02912223          	sw	s1,36(sp)
   10624:	00050b13          	mv	s6,a0
   10628:	00058b93          	mv	s7,a1
   1062c:	00060a93          	mv	s5,a2
   10630:	00000a13          	li	s4,0
   10634:	00100993          	li	s3,1
   10638:	fff00913          	li	s2,-1
   1063c:	004aa483          	lw	s1,4(s5)
   10640:	008aa403          	lw	s0,8(s5)
   10644:	fff48493          	addi	s1,s1,-1
   10648:	0204c863          	bltz	s1,10678 <_fwalk_sglue+0x7c>
   1064c:	00c45783          	lhu	a5,12(s0)
   10650:	00f9fe63          	bgeu	s3,a5,1066c <_fwalk_sglue+0x70>
   10654:	00e41783          	lh	a5,14(s0)
   10658:	00040593          	mv	a1,s0
   1065c:	000b0513          	mv	a0,s6
   10660:	01278663          	beq	a5,s2,1066c <_fwalk_sglue+0x70>
   10664:	000b80e7          	jalr	s7
   10668:	00aa6a33          	or	s4,s4,a0
   1066c:	fff48493          	addi	s1,s1,-1
   10670:	06840413          	addi	s0,s0,104
   10674:	fd249ce3          	bne	s1,s2,1064c <_fwalk_sglue+0x50>
   10678:	000aaa83          	lw	s5,0(s5)
   1067c:	fc0a90e3          	bnez	s5,1063c <_fwalk_sglue+0x40>
   10680:	02c12083          	lw	ra,44(sp)
   10684:	02812403          	lw	s0,40(sp)
   10688:	02412483          	lw	s1,36(sp)
   1068c:	02012903          	lw	s2,32(sp)
   10690:	01c12983          	lw	s3,28(sp)
   10694:	01412a83          	lw	s5,20(sp)
   10698:	01012b03          	lw	s6,16(sp)
   1069c:	00c12b83          	lw	s7,12(sp)
   106a0:	000a0513          	mv	a0,s4
   106a4:	01812a03          	lw	s4,24(sp)
   106a8:	03010113          	addi	sp,sp,48
   106ac:	00008067          	ret

000106b0 <__sread>:
   106b0:	ff010113          	addi	sp,sp,-16
   106b4:	00812423          	sw	s0,8(sp)
   106b8:	00058413          	mv	s0,a1
   106bc:	00e59583          	lh	a1,14(a1)
   106c0:	00112623          	sw	ra,12(sp)
   106c4:	2c8000ef          	jal	1098c <_read_r>
   106c8:	02054063          	bltz	a0,106e8 <__sread+0x38>
   106cc:	05042783          	lw	a5,80(s0)
   106d0:	00c12083          	lw	ra,12(sp)
   106d4:	00a787b3          	add	a5,a5,a0
   106d8:	04f42823          	sw	a5,80(s0)
   106dc:	00812403          	lw	s0,8(sp)
   106e0:	01010113          	addi	sp,sp,16
   106e4:	00008067          	ret
   106e8:	00c45783          	lhu	a5,12(s0)
   106ec:	fffff737          	lui	a4,0xfffff
   106f0:	fff70713          	addi	a4,a4,-1 # ffffefff <__BSS_END__+0xfffec54f>
   106f4:	00e7f7b3          	and	a5,a5,a4
   106f8:	00c12083          	lw	ra,12(sp)
   106fc:	00f41623          	sh	a5,12(s0)
   10700:	00812403          	lw	s0,8(sp)
   10704:	01010113          	addi	sp,sp,16
   10708:	00008067          	ret

0001070c <__seofread>:
   1070c:	00000513          	li	a0,0
   10710:	00008067          	ret

00010714 <__swrite>:
   10714:	00c59783          	lh	a5,12(a1)
   10718:	fe010113          	addi	sp,sp,-32
   1071c:	00812c23          	sw	s0,24(sp)
   10720:	00912a23          	sw	s1,20(sp)
   10724:	01212823          	sw	s2,16(sp)
   10728:	01312623          	sw	s3,12(sp)
   1072c:	00112e23          	sw	ra,28(sp)
   10730:	1007f713          	andi	a4,a5,256
   10734:	00058413          	mv	s0,a1
   10738:	00050493          	mv	s1,a0
   1073c:	00060913          	mv	s2,a2
   10740:	00068993          	mv	s3,a3
   10744:	04071063          	bnez	a4,10784 <__swrite+0x70>
   10748:	fffff737          	lui	a4,0xfffff
   1074c:	fff70713          	addi	a4,a4,-1 # ffffefff <__BSS_END__+0xfffec54f>
   10750:	00e7f7b3          	and	a5,a5,a4
   10754:	00e41583          	lh	a1,14(s0)
   10758:	00f41623          	sh	a5,12(s0)
   1075c:	01812403          	lw	s0,24(sp)
   10760:	01c12083          	lw	ra,28(sp)
   10764:	00098693          	mv	a3,s3
   10768:	00090613          	mv	a2,s2
   1076c:	00c12983          	lw	s3,12(sp)
   10770:	01012903          	lw	s2,16(sp)
   10774:	00048513          	mv	a0,s1
   10778:	01412483          	lw	s1,20(sp)
   1077c:	02010113          	addi	sp,sp,32
   10780:	2680006f          	j	109e8 <_write_r>
   10784:	00e59583          	lh	a1,14(a1)
   10788:	00200693          	li	a3,2
   1078c:	00000613          	li	a2,0
   10790:	1a0000ef          	jal	10930 <_lseek_r>
   10794:	00c41783          	lh	a5,12(s0)
   10798:	fb1ff06f          	j	10748 <__swrite+0x34>

0001079c <__sseek>:
   1079c:	ff010113          	addi	sp,sp,-16
   107a0:	00812423          	sw	s0,8(sp)
   107a4:	00058413          	mv	s0,a1
   107a8:	00e59583          	lh	a1,14(a1)
   107ac:	00112623          	sw	ra,12(sp)
   107b0:	180000ef          	jal	10930 <_lseek_r>
   107b4:	fff00793          	li	a5,-1
   107b8:	02f50863          	beq	a0,a5,107e8 <__sseek+0x4c>
   107bc:	00c45783          	lhu	a5,12(s0)
   107c0:	00001737          	lui	a4,0x1
   107c4:	00c12083          	lw	ra,12(sp)
   107c8:	00e7e7b3          	or	a5,a5,a4
   107cc:	01079793          	slli	a5,a5,0x10
   107d0:	4107d793          	srai	a5,a5,0x10
   107d4:	04a42823          	sw	a0,80(s0)
   107d8:	00f41623          	sh	a5,12(s0)
   107dc:	00812403          	lw	s0,8(sp)
   107e0:	01010113          	addi	sp,sp,16
   107e4:	00008067          	ret
   107e8:	00c45783          	lhu	a5,12(s0)
   107ec:	fffff737          	lui	a4,0xfffff
   107f0:	fff70713          	addi	a4,a4,-1 # ffffefff <__BSS_END__+0xfffec54f>
   107f4:	00e7f7b3          	and	a5,a5,a4
   107f8:	01079793          	slli	a5,a5,0x10
   107fc:	4107d793          	srai	a5,a5,0x10
   10800:	00c12083          	lw	ra,12(sp)
   10804:	00f41623          	sh	a5,12(s0)
   10808:	00812403          	lw	s0,8(sp)
   1080c:	01010113          	addi	sp,sp,16
   10810:	00008067          	ret

00010814 <__sclose>:
   10814:	00e59583          	lh	a1,14(a1)
   10818:	0040006f          	j	1081c <_close_r>

0001081c <_close_r>:
   1081c:	ff010113          	addi	sp,sp,-16
   10820:	00812423          	sw	s0,8(sp)
   10824:	00050413          	mv	s0,a0
   10828:	00058513          	mv	a0,a1
   1082c:	e401a623          	sw	zero,-436(gp) # 126cc <errno>
   10830:	00112623          	sw	ra,12(sp)
   10834:	65c010ef          	jal	11e90 <_close>
   10838:	fff00793          	li	a5,-1
   1083c:	00f50a63          	beq	a0,a5,10850 <_close_r+0x34>
   10840:	00c12083          	lw	ra,12(sp)
   10844:	00812403          	lw	s0,8(sp)
   10848:	01010113          	addi	sp,sp,16
   1084c:	00008067          	ret
   10850:	e4c1a783          	lw	a5,-436(gp) # 126cc <errno>
   10854:	fe0786e3          	beqz	a5,10840 <_close_r+0x24>
   10858:	00c12083          	lw	ra,12(sp)
   1085c:	00f42023          	sw	a5,0(s0)
   10860:	00812403          	lw	s0,8(sp)
   10864:	01010113          	addi	sp,sp,16
   10868:	00008067          	ret

0001086c <_reclaim_reent>:
   1086c:	e3c1a783          	lw	a5,-452(gp) # 126bc <_impure_ptr>
   10870:	0aa78e63          	beq	a5,a0,1092c <_reclaim_reent+0xc0>
   10874:	04452583          	lw	a1,68(a0)
   10878:	fe010113          	addi	sp,sp,-32
   1087c:	00912a23          	sw	s1,20(sp)
   10880:	00112e23          	sw	ra,28(sp)
   10884:	00050493          	mv	s1,a0
   10888:	04058c63          	beqz	a1,108e0 <_reclaim_reent+0x74>
   1088c:	01212823          	sw	s2,16(sp)
   10890:	01312623          	sw	s3,12(sp)
   10894:	00812c23          	sw	s0,24(sp)
   10898:	00000913          	li	s2,0
   1089c:	08000993          	li	s3,128
   108a0:	012587b3          	add	a5,a1,s2
   108a4:	0007a403          	lw	s0,0(a5)
   108a8:	00040e63          	beqz	s0,108c4 <_reclaim_reent+0x58>
   108ac:	00040593          	mv	a1,s0
   108b0:	00042403          	lw	s0,0(s0)
   108b4:	00048513          	mv	a0,s1
   108b8:	580000ef          	jal	10e38 <_free_r>
   108bc:	fe0418e3          	bnez	s0,108ac <_reclaim_reent+0x40>
   108c0:	0444a583          	lw	a1,68(s1)
   108c4:	00490913          	addi	s2,s2,4
   108c8:	fd391ce3          	bne	s2,s3,108a0 <_reclaim_reent+0x34>
   108cc:	00048513          	mv	a0,s1
   108d0:	568000ef          	jal	10e38 <_free_r>
   108d4:	01812403          	lw	s0,24(sp)
   108d8:	01012903          	lw	s2,16(sp)
   108dc:	00c12983          	lw	s3,12(sp)
   108e0:	0384a583          	lw	a1,56(s1)
   108e4:	00058663          	beqz	a1,108f0 <_reclaim_reent+0x84>
   108e8:	00048513          	mv	a0,s1
   108ec:	54c000ef          	jal	10e38 <_free_r>
   108f0:	04c4a583          	lw	a1,76(s1)
   108f4:	00058663          	beqz	a1,10900 <_reclaim_reent+0x94>
   108f8:	00048513          	mv	a0,s1
   108fc:	53c000ef          	jal	10e38 <_free_r>
   10900:	0344a783          	lw	a5,52(s1)
   10904:	00078c63          	beqz	a5,1091c <_reclaim_reent+0xb0>
   10908:	01c12083          	lw	ra,28(sp)
   1090c:	00048513          	mv	a0,s1
   10910:	01412483          	lw	s1,20(sp)
   10914:	02010113          	addi	sp,sp,32
   10918:	00078067          	jr	a5
   1091c:	01c12083          	lw	ra,28(sp)
   10920:	01412483          	lw	s1,20(sp)
   10924:	02010113          	addi	sp,sp,32
   10928:	00008067          	ret
   1092c:	00008067          	ret

00010930 <_lseek_r>:
   10930:	ff010113          	addi	sp,sp,-16
   10934:	00058713          	mv	a4,a1
   10938:	00812423          	sw	s0,8(sp)
   1093c:	00060593          	mv	a1,a2
   10940:	00050413          	mv	s0,a0
   10944:	00068613          	mv	a2,a3
   10948:	00070513          	mv	a0,a4
   1094c:	e401a623          	sw	zero,-436(gp) # 126cc <errno>
   10950:	00112623          	sw	ra,12(sp)
   10954:	54c010ef          	jal	11ea0 <_lseek>
   10958:	fff00793          	li	a5,-1
   1095c:	00f50a63          	beq	a0,a5,10970 <_lseek_r+0x40>
   10960:	00c12083          	lw	ra,12(sp)
   10964:	00812403          	lw	s0,8(sp)
   10968:	01010113          	addi	sp,sp,16
   1096c:	00008067          	ret
   10970:	e4c1a783          	lw	a5,-436(gp) # 126cc <errno>
   10974:	fe0786e3          	beqz	a5,10960 <_lseek_r+0x30>
   10978:	00c12083          	lw	ra,12(sp)
   1097c:	00f42023          	sw	a5,0(s0)
   10980:	00812403          	lw	s0,8(sp)
   10984:	01010113          	addi	sp,sp,16
   10988:	00008067          	ret

0001098c <_read_r>:
   1098c:	ff010113          	addi	sp,sp,-16
   10990:	00058713          	mv	a4,a1
   10994:	00812423          	sw	s0,8(sp)
   10998:	00060593          	mv	a1,a2
   1099c:	00050413          	mv	s0,a0
   109a0:	00068613          	mv	a2,a3
   109a4:	00070513          	mv	a0,a4
   109a8:	e401a623          	sw	zero,-436(gp) # 126cc <errno>
   109ac:	00112623          	sw	ra,12(sp)
   109b0:	500010ef          	jal	11eb0 <_read>
   109b4:	fff00793          	li	a5,-1
   109b8:	00f50a63          	beq	a0,a5,109cc <_read_r+0x40>
   109bc:	00c12083          	lw	ra,12(sp)
   109c0:	00812403          	lw	s0,8(sp)
   109c4:	01010113          	addi	sp,sp,16
   109c8:	00008067          	ret
   109cc:	e4c1a783          	lw	a5,-436(gp) # 126cc <errno>
   109d0:	fe0786e3          	beqz	a5,109bc <_read_r+0x30>
   109d4:	00c12083          	lw	ra,12(sp)
   109d8:	00f42023          	sw	a5,0(s0)
   109dc:	00812403          	lw	s0,8(sp)
   109e0:	01010113          	addi	sp,sp,16
   109e4:	00008067          	ret

000109e8 <_write_r>:
   109e8:	ff010113          	addi	sp,sp,-16
   109ec:	00058713          	mv	a4,a1
   109f0:	00812423          	sw	s0,8(sp)
   109f4:	00060593          	mv	a1,a2
   109f8:	00050413          	mv	s0,a0
   109fc:	00068613          	mv	a2,a3
   10a00:	00070513          	mv	a0,a4
   10a04:	e401a623          	sw	zero,-436(gp) # 126cc <errno>
   10a08:	00112623          	sw	ra,12(sp)
   10a0c:	4e4010ef          	jal	11ef0 <_write>
   10a10:	fff00793          	li	a5,-1
   10a14:	00f50a63          	beq	a0,a5,10a28 <_write_r+0x40>
   10a18:	00c12083          	lw	ra,12(sp)
   10a1c:	00812403          	lw	s0,8(sp)
   10a20:	01010113          	addi	sp,sp,16
   10a24:	00008067          	ret
   10a28:	e4c1a783          	lw	a5,-436(gp) # 126cc <errno>
   10a2c:	fe0786e3          	beqz	a5,10a18 <_write_r+0x30>
   10a30:	00c12083          	lw	ra,12(sp)
   10a34:	00f42023          	sw	a5,0(s0)
   10a38:	00812403          	lw	s0,8(sp)
   10a3c:	01010113          	addi	sp,sp,16
   10a40:	00008067          	ret

00010a44 <__libc_init_array>:
   10a44:	ff010113          	addi	sp,sp,-16
   10a48:	00812423          	sw	s0,8(sp)
   10a4c:	01212023          	sw	s2,0(sp)
   10a50:	00001797          	auipc	a5,0x1
   10a54:	60c78793          	addi	a5,a5,1548 # 1205c <__init_array_start>
   10a58:	00001417          	auipc	s0,0x1
   10a5c:	60440413          	addi	s0,s0,1540 # 1205c <__init_array_start>
   10a60:	00112623          	sw	ra,12(sp)
   10a64:	00912223          	sw	s1,4(sp)
   10a68:	40878933          	sub	s2,a5,s0
   10a6c:	02878063          	beq	a5,s0,10a8c <__libc_init_array+0x48>
   10a70:	40295913          	srai	s2,s2,0x2
   10a74:	00000493          	li	s1,0
   10a78:	00042783          	lw	a5,0(s0)
   10a7c:	00148493          	addi	s1,s1,1
   10a80:	00440413          	addi	s0,s0,4
   10a84:	000780e7          	jalr	a5
   10a88:	ff24e8e3          	bltu	s1,s2,10a78 <__libc_init_array+0x34>
   10a8c:	00001797          	auipc	a5,0x1
   10a90:	5d878793          	addi	a5,a5,1496 # 12064 <__do_global_dtors_aux_fini_array_entry>
   10a94:	00001417          	auipc	s0,0x1
   10a98:	5c840413          	addi	s0,s0,1480 # 1205c <__init_array_start>
   10a9c:	40878933          	sub	s2,a5,s0
   10aa0:	40295913          	srai	s2,s2,0x2
   10aa4:	00878e63          	beq	a5,s0,10ac0 <__libc_init_array+0x7c>
   10aa8:	00000493          	li	s1,0
   10aac:	00042783          	lw	a5,0(s0)
   10ab0:	00148493          	addi	s1,s1,1
   10ab4:	00440413          	addi	s0,s0,4
   10ab8:	000780e7          	jalr	a5
   10abc:	ff24e8e3          	bltu	s1,s2,10aac <__libc_init_array+0x68>
   10ac0:	00c12083          	lw	ra,12(sp)
   10ac4:	00812403          	lw	s0,8(sp)
   10ac8:	00412483          	lw	s1,4(sp)
   10acc:	00012903          	lw	s2,0(sp)
   10ad0:	01010113          	addi	sp,sp,16
   10ad4:	00008067          	ret

00010ad8 <memset>:
   10ad8:	00f00313          	li	t1,15
   10adc:	00050713          	mv	a4,a0
   10ae0:	02c37e63          	bgeu	t1,a2,10b1c <memset+0x44>
   10ae4:	00f77793          	andi	a5,a4,15
   10ae8:	0a079063          	bnez	a5,10b88 <memset+0xb0>
   10aec:	08059263          	bnez	a1,10b70 <memset+0x98>
   10af0:	ff067693          	andi	a3,a2,-16
   10af4:	00f67613          	andi	a2,a2,15
   10af8:	00e686b3          	add	a3,a3,a4
   10afc:	00b72023          	sw	a1,0(a4)
   10b00:	00b72223          	sw	a1,4(a4)
   10b04:	00b72423          	sw	a1,8(a4)
   10b08:	00b72623          	sw	a1,12(a4)
   10b0c:	01070713          	addi	a4,a4,16
   10b10:	fed766e3          	bltu	a4,a3,10afc <memset+0x24>
   10b14:	00061463          	bnez	a2,10b1c <memset+0x44>
   10b18:	00008067          	ret
   10b1c:	40c306b3          	sub	a3,t1,a2
   10b20:	00269693          	slli	a3,a3,0x2
   10b24:	00000297          	auipc	t0,0x0
   10b28:	005686b3          	add	a3,a3,t0
   10b2c:	00c68067          	jr	12(a3)
   10b30:	00b70723          	sb	a1,14(a4)
   10b34:	00b706a3          	sb	a1,13(a4)
   10b38:	00b70623          	sb	a1,12(a4)
   10b3c:	00b705a3          	sb	a1,11(a4)
   10b40:	00b70523          	sb	a1,10(a4)
   10b44:	00b704a3          	sb	a1,9(a4)
   10b48:	00b70423          	sb	a1,8(a4)
   10b4c:	00b703a3          	sb	a1,7(a4)
   10b50:	00b70323          	sb	a1,6(a4)
   10b54:	00b702a3          	sb	a1,5(a4)
   10b58:	00b70223          	sb	a1,4(a4)
   10b5c:	00b701a3          	sb	a1,3(a4)
   10b60:	00b70123          	sb	a1,2(a4)
   10b64:	00b700a3          	sb	a1,1(a4)
   10b68:	00b70023          	sb	a1,0(a4)
   10b6c:	00008067          	ret
   10b70:	0ff5f593          	zext.b	a1,a1
   10b74:	00859693          	slli	a3,a1,0x8
   10b78:	00d5e5b3          	or	a1,a1,a3
   10b7c:	01059693          	slli	a3,a1,0x10
   10b80:	00d5e5b3          	or	a1,a1,a3
   10b84:	f6dff06f          	j	10af0 <memset+0x18>
   10b88:	00279693          	slli	a3,a5,0x2
   10b8c:	00000297          	auipc	t0,0x0
   10b90:	005686b3          	add	a3,a3,t0
   10b94:	00008293          	mv	t0,ra
   10b98:	fa0680e7          	jalr	-96(a3)
   10b9c:	00028093          	mv	ra,t0
   10ba0:	ff078793          	addi	a5,a5,-16
   10ba4:	40f70733          	sub	a4,a4,a5
   10ba8:	00f60633          	add	a2,a2,a5
   10bac:	f6c378e3          	bgeu	t1,a2,10b1c <memset+0x44>
   10bb0:	f3dff06f          	j	10aec <memset+0x14>

00010bb4 <__call_exitprocs>:
   10bb4:	fd010113          	addi	sp,sp,-48
   10bb8:	01412c23          	sw	s4,24(sp)
   10bbc:	e5018a13          	addi	s4,gp,-432 # 126d0 <__atexit>
   10bc0:	03212023          	sw	s2,32(sp)
   10bc4:	000a2903          	lw	s2,0(s4)
   10bc8:	02112623          	sw	ra,44(sp)
   10bcc:	0a090863          	beqz	s2,10c7c <__call_exitprocs+0xc8>
   10bd0:	01312e23          	sw	s3,28(sp)
   10bd4:	01512a23          	sw	s5,20(sp)
   10bd8:	01612823          	sw	s6,16(sp)
   10bdc:	01712623          	sw	s7,12(sp)
   10be0:	02812423          	sw	s0,40(sp)
   10be4:	02912223          	sw	s1,36(sp)
   10be8:	01812423          	sw	s8,8(sp)
   10bec:	00050b13          	mv	s6,a0
   10bf0:	00058b93          	mv	s7,a1
   10bf4:	fff00993          	li	s3,-1
   10bf8:	00100a93          	li	s5,1
   10bfc:	00492483          	lw	s1,4(s2)
   10c00:	fff48413          	addi	s0,s1,-1
   10c04:	04044e63          	bltz	s0,10c60 <__call_exitprocs+0xac>
   10c08:	00249493          	slli	s1,s1,0x2
   10c0c:	009904b3          	add	s1,s2,s1
   10c10:	080b9063          	bnez	s7,10c90 <__call_exitprocs+0xdc>
   10c14:	00492783          	lw	a5,4(s2)
   10c18:	0044a683          	lw	a3,4(s1)
   10c1c:	fff78793          	addi	a5,a5,-1
   10c20:	0a878c63          	beq	a5,s0,10cd8 <__call_exitprocs+0x124>
   10c24:	0004a223          	sw	zero,4(s1)
   10c28:	02068663          	beqz	a3,10c54 <__call_exitprocs+0xa0>
   10c2c:	18892783          	lw	a5,392(s2)
   10c30:	008a9733          	sll	a4,s5,s0
   10c34:	00492c03          	lw	s8,4(s2)
   10c38:	00f777b3          	and	a5,a4,a5
   10c3c:	06079663          	bnez	a5,10ca8 <__call_exitprocs+0xf4>
   10c40:	000680e7          	jalr	a3
   10c44:	00492703          	lw	a4,4(s2)
   10c48:	000a2783          	lw	a5,0(s4)
   10c4c:	09871063          	bne	a4,s8,10ccc <__call_exitprocs+0x118>
   10c50:	07279e63          	bne	a5,s2,10ccc <__call_exitprocs+0x118>
   10c54:	fff40413          	addi	s0,s0,-1
   10c58:	ffc48493          	addi	s1,s1,-4
   10c5c:	fb341ae3          	bne	s0,s3,10c10 <__call_exitprocs+0x5c>
   10c60:	02812403          	lw	s0,40(sp)
   10c64:	02412483          	lw	s1,36(sp)
   10c68:	01c12983          	lw	s3,28(sp)
   10c6c:	01412a83          	lw	s5,20(sp)
   10c70:	01012b03          	lw	s6,16(sp)
   10c74:	00c12b83          	lw	s7,12(sp)
   10c78:	00812c03          	lw	s8,8(sp)
   10c7c:	02c12083          	lw	ra,44(sp)
   10c80:	02012903          	lw	s2,32(sp)
   10c84:	01812a03          	lw	s4,24(sp)
   10c88:	03010113          	addi	sp,sp,48
   10c8c:	00008067          	ret
   10c90:	1044a783          	lw	a5,260(s1)
   10c94:	f97780e3          	beq	a5,s7,10c14 <__call_exitprocs+0x60>
   10c98:	fff40413          	addi	s0,s0,-1
   10c9c:	ffc48493          	addi	s1,s1,-4
   10ca0:	ff3418e3          	bne	s0,s3,10c90 <__call_exitprocs+0xdc>
   10ca4:	fbdff06f          	j	10c60 <__call_exitprocs+0xac>
   10ca8:	18c92783          	lw	a5,396(s2)
   10cac:	0844a583          	lw	a1,132(s1)
   10cb0:	00f77733          	and	a4,a4,a5
   10cb4:	02071663          	bnez	a4,10ce0 <__call_exitprocs+0x12c>
   10cb8:	000b0513          	mv	a0,s6
   10cbc:	000680e7          	jalr	a3
   10cc0:	00492703          	lw	a4,4(s2)
   10cc4:	000a2783          	lw	a5,0(s4)
   10cc8:	f98704e3          	beq	a4,s8,10c50 <__call_exitprocs+0x9c>
   10ccc:	f8078ae3          	beqz	a5,10c60 <__call_exitprocs+0xac>
   10cd0:	00078913          	mv	s2,a5
   10cd4:	f29ff06f          	j	10bfc <__call_exitprocs+0x48>
   10cd8:	00892223          	sw	s0,4(s2)
   10cdc:	f4dff06f          	j	10c28 <__call_exitprocs+0x74>
   10ce0:	00058513          	mv	a0,a1
   10ce4:	000680e7          	jalr	a3
   10ce8:	f5dff06f          	j	10c44 <__call_exitprocs+0x90>

00010cec <atexit>:
   10cec:	00050593          	mv	a1,a0
   10cf0:	00000693          	li	a3,0
   10cf4:	00000613          	li	a2,0
   10cf8:	00000513          	li	a0,0
   10cfc:	0fc0106f          	j	11df8 <__register_exitproc>

00010d00 <_malloc_trim_r>:
   10d00:	fe010113          	addi	sp,sp,-32
   10d04:	00812c23          	sw	s0,24(sp)
   10d08:	00912a23          	sw	s1,20(sp)
   10d0c:	01212823          	sw	s2,16(sp)
   10d10:	01312623          	sw	s3,12(sp)
   10d14:	01412423          	sw	s4,8(sp)
   10d18:	00058993          	mv	s3,a1
   10d1c:	00112e23          	sw	ra,28(sp)
   10d20:	00050913          	mv	s2,a0
   10d24:	00001a17          	auipc	s4,0x1
   10d28:	58ca0a13          	addi	s4,s4,1420 # 122b0 <__malloc_av_>
   10d2c:	3d9000ef          	jal	11904 <__malloc_lock>
   10d30:	008a2703          	lw	a4,8(s4)
   10d34:	000017b7          	lui	a5,0x1
   10d38:	fef78793          	addi	a5,a5,-17 # fef <exit-0xf0c5>
   10d3c:	00472483          	lw	s1,4(a4)
   10d40:	00001737          	lui	a4,0x1
   10d44:	ffc4f493          	andi	s1,s1,-4
   10d48:	00f48433          	add	s0,s1,a5
   10d4c:	41340433          	sub	s0,s0,s3
   10d50:	00c45413          	srli	s0,s0,0xc
   10d54:	fff40413          	addi	s0,s0,-1
   10d58:	00c41413          	slli	s0,s0,0xc
   10d5c:	00e44e63          	blt	s0,a4,10d78 <_malloc_trim_r+0x78>
   10d60:	00000593          	li	a1,0
   10d64:	00090513          	mv	a0,s2
   10d68:	7e5000ef          	jal	11d4c <_sbrk_r>
   10d6c:	008a2783          	lw	a5,8(s4)
   10d70:	009787b3          	add	a5,a5,s1
   10d74:	02f50863          	beq	a0,a5,10da4 <_malloc_trim_r+0xa4>
   10d78:	00090513          	mv	a0,s2
   10d7c:	38d000ef          	jal	11908 <__malloc_unlock>
   10d80:	01c12083          	lw	ra,28(sp)
   10d84:	01812403          	lw	s0,24(sp)
   10d88:	01412483          	lw	s1,20(sp)
   10d8c:	01012903          	lw	s2,16(sp)
   10d90:	00c12983          	lw	s3,12(sp)
   10d94:	00812a03          	lw	s4,8(sp)
   10d98:	00000513          	li	a0,0
   10d9c:	02010113          	addi	sp,sp,32
   10da0:	00008067          	ret
   10da4:	408005b3          	neg	a1,s0
   10da8:	00090513          	mv	a0,s2
   10dac:	7a1000ef          	jal	11d4c <_sbrk_r>
   10db0:	fff00793          	li	a5,-1
   10db4:	04f50863          	beq	a0,a5,10e04 <_malloc_trim_r+0x104>
   10db8:	07818713          	addi	a4,gp,120 # 128f8 <__malloc_current_mallinfo>
   10dbc:	00072783          	lw	a5,0(a4) # 1000 <exit-0xf0b4>
   10dc0:	008a2683          	lw	a3,8(s4)
   10dc4:	408484b3          	sub	s1,s1,s0
   10dc8:	0014e493          	ori	s1,s1,1
   10dcc:	408787b3          	sub	a5,a5,s0
   10dd0:	00090513          	mv	a0,s2
   10dd4:	0096a223          	sw	s1,4(a3)
   10dd8:	00f72023          	sw	a5,0(a4)
   10ddc:	32d000ef          	jal	11908 <__malloc_unlock>
   10de0:	01c12083          	lw	ra,28(sp)
   10de4:	01812403          	lw	s0,24(sp)
   10de8:	01412483          	lw	s1,20(sp)
   10dec:	01012903          	lw	s2,16(sp)
   10df0:	00c12983          	lw	s3,12(sp)
   10df4:	00812a03          	lw	s4,8(sp)
   10df8:	00100513          	li	a0,1
   10dfc:	02010113          	addi	sp,sp,32
   10e00:	00008067          	ret
   10e04:	00000593          	li	a1,0
   10e08:	00090513          	mv	a0,s2
   10e0c:	741000ef          	jal	11d4c <_sbrk_r>
   10e10:	008a2703          	lw	a4,8(s4)
   10e14:	00f00693          	li	a3,15
   10e18:	40e507b3          	sub	a5,a0,a4
   10e1c:	f4f6dee3          	bge	a3,a5,10d78 <_malloc_trim_r+0x78>
   10e20:	e401a683          	lw	a3,-448(gp) # 126c0 <__malloc_sbrk_base>
   10e24:	40d50533          	sub	a0,a0,a3
   10e28:	0017e793          	ori	a5,a5,1
   10e2c:	06a1ac23          	sw	a0,120(gp) # 128f8 <__malloc_current_mallinfo>
   10e30:	00f72223          	sw	a5,4(a4)
   10e34:	f45ff06f          	j	10d78 <_malloc_trim_r+0x78>

00010e38 <_free_r>:
   10e38:	18058263          	beqz	a1,10fbc <_free_r+0x184>
   10e3c:	ff010113          	addi	sp,sp,-16
   10e40:	00812423          	sw	s0,8(sp)
   10e44:	00912223          	sw	s1,4(sp)
   10e48:	00058413          	mv	s0,a1
   10e4c:	00050493          	mv	s1,a0
   10e50:	00112623          	sw	ra,12(sp)
   10e54:	2b1000ef          	jal	11904 <__malloc_lock>
   10e58:	ffc42583          	lw	a1,-4(s0)
   10e5c:	ff840713          	addi	a4,s0,-8
   10e60:	00001517          	auipc	a0,0x1
   10e64:	45050513          	addi	a0,a0,1104 # 122b0 <__malloc_av_>
   10e68:	ffe5f793          	andi	a5,a1,-2
   10e6c:	00f70633          	add	a2,a4,a5
   10e70:	00462683          	lw	a3,4(a2)
   10e74:	00852803          	lw	a6,8(a0)
   10e78:	ffc6f693          	andi	a3,a3,-4
   10e7c:	1ac80263          	beq	a6,a2,11020 <_free_r+0x1e8>
   10e80:	00d62223          	sw	a3,4(a2)
   10e84:	0015f593          	andi	a1,a1,1
   10e88:	00d60833          	add	a6,a2,a3
   10e8c:	0a059063          	bnez	a1,10f2c <_free_r+0xf4>
   10e90:	ff842303          	lw	t1,-8(s0)
   10e94:	00482583          	lw	a1,4(a6)
   10e98:	00001897          	auipc	a7,0x1
   10e9c:	42088893          	addi	a7,a7,1056 # 122b8 <__malloc_av_+0x8>
   10ea0:	40670733          	sub	a4,a4,t1
   10ea4:	00872803          	lw	a6,8(a4)
   10ea8:	006787b3          	add	a5,a5,t1
   10eac:	0015f593          	andi	a1,a1,1
   10eb0:	15180263          	beq	a6,a7,10ff4 <_free_r+0x1bc>
   10eb4:	00c72303          	lw	t1,12(a4)
   10eb8:	00682623          	sw	t1,12(a6)
   10ebc:	01032423          	sw	a6,8(t1) # 101a8 <frame_dummy+0x1c>
   10ec0:	1a058663          	beqz	a1,1106c <_free_r+0x234>
   10ec4:	0017e693          	ori	a3,a5,1
   10ec8:	00d72223          	sw	a3,4(a4)
   10ecc:	00f62023          	sw	a5,0(a2)
   10ed0:	1ff00693          	li	a3,511
   10ed4:	06f6ec63          	bltu	a3,a5,10f4c <_free_r+0x114>
   10ed8:	ff87f693          	andi	a3,a5,-8
   10edc:	00868693          	addi	a3,a3,8
   10ee0:	00452583          	lw	a1,4(a0)
   10ee4:	00d506b3          	add	a3,a0,a3
   10ee8:	0006a603          	lw	a2,0(a3)
   10eec:	0057d813          	srli	a6,a5,0x5
   10ef0:	00100793          	li	a5,1
   10ef4:	010797b3          	sll	a5,a5,a6
   10ef8:	00b7e7b3          	or	a5,a5,a1
   10efc:	ff868593          	addi	a1,a3,-8
   10f00:	00b72623          	sw	a1,12(a4)
   10f04:	00c72423          	sw	a2,8(a4)
   10f08:	00f52223          	sw	a5,4(a0)
   10f0c:	00e6a023          	sw	a4,0(a3)
   10f10:	00e62623          	sw	a4,12(a2)
   10f14:	00812403          	lw	s0,8(sp)
   10f18:	00c12083          	lw	ra,12(sp)
   10f1c:	00048513          	mv	a0,s1
   10f20:	00412483          	lw	s1,4(sp)
   10f24:	01010113          	addi	sp,sp,16
   10f28:	1e10006f          	j	11908 <__malloc_unlock>
   10f2c:	00482583          	lw	a1,4(a6)
   10f30:	0015f593          	andi	a1,a1,1
   10f34:	08058663          	beqz	a1,10fc0 <_free_r+0x188>
   10f38:	0017e693          	ori	a3,a5,1
   10f3c:	fed42e23          	sw	a3,-4(s0)
   10f40:	00f62023          	sw	a5,0(a2)
   10f44:	1ff00693          	li	a3,511
   10f48:	f8f6f8e3          	bgeu	a3,a5,10ed8 <_free_r+0xa0>
   10f4c:	0097d693          	srli	a3,a5,0x9
   10f50:	00400613          	li	a2,4
   10f54:	12d66063          	bltu	a2,a3,11074 <_free_r+0x23c>
   10f58:	0067d693          	srli	a3,a5,0x6
   10f5c:	03968593          	addi	a1,a3,57
   10f60:	03868613          	addi	a2,a3,56
   10f64:	00359593          	slli	a1,a1,0x3
   10f68:	00b505b3          	add	a1,a0,a1
   10f6c:	0005a683          	lw	a3,0(a1)
   10f70:	ff858593          	addi	a1,a1,-8
   10f74:	00d59863          	bne	a1,a3,10f84 <_free_r+0x14c>
   10f78:	1540006f          	j	110cc <_free_r+0x294>
   10f7c:	0086a683          	lw	a3,8(a3)
   10f80:	00d58863          	beq	a1,a3,10f90 <_free_r+0x158>
   10f84:	0046a603          	lw	a2,4(a3)
   10f88:	ffc67613          	andi	a2,a2,-4
   10f8c:	fec7e8e3          	bltu	a5,a2,10f7c <_free_r+0x144>
   10f90:	00c6a583          	lw	a1,12(a3)
   10f94:	00b72623          	sw	a1,12(a4)
   10f98:	00d72423          	sw	a3,8(a4)
   10f9c:	00812403          	lw	s0,8(sp)
   10fa0:	00c12083          	lw	ra,12(sp)
   10fa4:	00e5a423          	sw	a4,8(a1)
   10fa8:	00048513          	mv	a0,s1
   10fac:	00412483          	lw	s1,4(sp)
   10fb0:	00e6a623          	sw	a4,12(a3)
   10fb4:	01010113          	addi	sp,sp,16
   10fb8:	1510006f          	j	11908 <__malloc_unlock>
   10fbc:	00008067          	ret
   10fc0:	00d787b3          	add	a5,a5,a3
   10fc4:	00001897          	auipc	a7,0x1
   10fc8:	2f488893          	addi	a7,a7,756 # 122b8 <__malloc_av_+0x8>
   10fcc:	00862683          	lw	a3,8(a2)
   10fd0:	0d168c63          	beq	a3,a7,110a8 <_free_r+0x270>
   10fd4:	00c62803          	lw	a6,12(a2)
   10fd8:	0017e593          	ori	a1,a5,1
   10fdc:	00f70633          	add	a2,a4,a5
   10fe0:	0106a623          	sw	a6,12(a3)
   10fe4:	00d82423          	sw	a3,8(a6)
   10fe8:	00b72223          	sw	a1,4(a4)
   10fec:	00f62023          	sw	a5,0(a2)
   10ff0:	ee1ff06f          	j	10ed0 <_free_r+0x98>
   10ff4:	12059c63          	bnez	a1,1112c <_free_r+0x2f4>
   10ff8:	00862583          	lw	a1,8(a2)
   10ffc:	00c62603          	lw	a2,12(a2)
   11000:	00f686b3          	add	a3,a3,a5
   11004:	0016e793          	ori	a5,a3,1
   11008:	00c5a623          	sw	a2,12(a1)
   1100c:	00b62423          	sw	a1,8(a2)
   11010:	00f72223          	sw	a5,4(a4)
   11014:	00d70733          	add	a4,a4,a3
   11018:	00d72023          	sw	a3,0(a4)
   1101c:	ef9ff06f          	j	10f14 <_free_r+0xdc>
   11020:	0015f593          	andi	a1,a1,1
   11024:	00d786b3          	add	a3,a5,a3
   11028:	02059063          	bnez	a1,11048 <_free_r+0x210>
   1102c:	ff842583          	lw	a1,-8(s0)
   11030:	40b70733          	sub	a4,a4,a1
   11034:	00c72783          	lw	a5,12(a4)
   11038:	00872603          	lw	a2,8(a4)
   1103c:	00b686b3          	add	a3,a3,a1
   11040:	00f62623          	sw	a5,12(a2)
   11044:	00c7a423          	sw	a2,8(a5)
   11048:	0016e793          	ori	a5,a3,1
   1104c:	00f72223          	sw	a5,4(a4)
   11050:	00e52423          	sw	a4,8(a0)
   11054:	e441a783          	lw	a5,-444(gp) # 126c4 <__malloc_trim_threshold>
   11058:	eaf6eee3          	bltu	a3,a5,10f14 <_free_r+0xdc>
   1105c:	e5c1a583          	lw	a1,-420(gp) # 126dc <__malloc_top_pad>
   11060:	00048513          	mv	a0,s1
   11064:	c9dff0ef          	jal	10d00 <_malloc_trim_r>
   11068:	eadff06f          	j	10f14 <_free_r+0xdc>
   1106c:	00d787b3          	add	a5,a5,a3
   11070:	f5dff06f          	j	10fcc <_free_r+0x194>
   11074:	01400613          	li	a2,20
   11078:	02d67063          	bgeu	a2,a3,11098 <_free_r+0x260>
   1107c:	05400613          	li	a2,84
   11080:	06d66463          	bltu	a2,a3,110e8 <_free_r+0x2b0>
   11084:	00c7d693          	srli	a3,a5,0xc
   11088:	06f68593          	addi	a1,a3,111
   1108c:	06e68613          	addi	a2,a3,110
   11090:	00359593          	slli	a1,a1,0x3
   11094:	ed5ff06f          	j	10f68 <_free_r+0x130>
   11098:	05c68593          	addi	a1,a3,92
   1109c:	05b68613          	addi	a2,a3,91
   110a0:	00359593          	slli	a1,a1,0x3
   110a4:	ec5ff06f          	j	10f68 <_free_r+0x130>
   110a8:	00e52a23          	sw	a4,20(a0)
   110ac:	00e52823          	sw	a4,16(a0)
   110b0:	0017e693          	ori	a3,a5,1
   110b4:	01172623          	sw	a7,12(a4)
   110b8:	01172423          	sw	a7,8(a4)
   110bc:	00d72223          	sw	a3,4(a4)
   110c0:	00f70733          	add	a4,a4,a5
   110c4:	00f72023          	sw	a5,0(a4)
   110c8:	e4dff06f          	j	10f14 <_free_r+0xdc>
   110cc:	00452803          	lw	a6,4(a0)
   110d0:	40265613          	srai	a2,a2,0x2
   110d4:	00100793          	li	a5,1
   110d8:	00c797b3          	sll	a5,a5,a2
   110dc:	0107e7b3          	or	a5,a5,a6
   110e0:	00f52223          	sw	a5,4(a0)
   110e4:	eb1ff06f          	j	10f94 <_free_r+0x15c>
   110e8:	15400613          	li	a2,340
   110ec:	00d66c63          	bltu	a2,a3,11104 <_free_r+0x2cc>
   110f0:	00f7d693          	srli	a3,a5,0xf
   110f4:	07868593          	addi	a1,a3,120
   110f8:	07768613          	addi	a2,a3,119
   110fc:	00359593          	slli	a1,a1,0x3
   11100:	e69ff06f          	j	10f68 <_free_r+0x130>
   11104:	55400613          	li	a2,1364
   11108:	00d66c63          	bltu	a2,a3,11120 <_free_r+0x2e8>
   1110c:	0127d693          	srli	a3,a5,0x12
   11110:	07d68593          	addi	a1,a3,125
   11114:	07c68613          	addi	a2,a3,124
   11118:	00359593          	slli	a1,a1,0x3
   1111c:	e4dff06f          	j	10f68 <_free_r+0x130>
   11120:	3f800593          	li	a1,1016
   11124:	07e00613          	li	a2,126
   11128:	e41ff06f          	j	10f68 <_free_r+0x130>
   1112c:	0017e693          	ori	a3,a5,1
   11130:	00d72223          	sw	a3,4(a4)
   11134:	00f62023          	sw	a5,0(a2)
   11138:	dddff06f          	j	10f14 <_free_r+0xdc>

0001113c <_malloc_r>:
   1113c:	fd010113          	addi	sp,sp,-48
   11140:	03212023          	sw	s2,32(sp)
   11144:	02112623          	sw	ra,44(sp)
   11148:	02812423          	sw	s0,40(sp)
   1114c:	02912223          	sw	s1,36(sp)
   11150:	01312e23          	sw	s3,28(sp)
   11154:	00b58793          	addi	a5,a1,11
   11158:	01600713          	li	a4,22
   1115c:	00050913          	mv	s2,a0
   11160:	08f76263          	bltu	a4,a5,111e4 <_malloc_r+0xa8>
   11164:	01000793          	li	a5,16
   11168:	20b7e663          	bltu	a5,a1,11374 <_malloc_r+0x238>
   1116c:	798000ef          	jal	11904 <__malloc_lock>
   11170:	01800793          	li	a5,24
   11174:	00200593          	li	a1,2
   11178:	01000493          	li	s1,16
   1117c:	00001997          	auipc	s3,0x1
   11180:	13498993          	addi	s3,s3,308 # 122b0 <__malloc_av_>
   11184:	00f987b3          	add	a5,s3,a5
   11188:	0047a403          	lw	s0,4(a5)
   1118c:	ff878713          	addi	a4,a5,-8
   11190:	34e40a63          	beq	s0,a4,114e4 <_malloc_r+0x3a8>
   11194:	00442783          	lw	a5,4(s0)
   11198:	00c42683          	lw	a3,12(s0)
   1119c:	00842603          	lw	a2,8(s0)
   111a0:	ffc7f793          	andi	a5,a5,-4
   111a4:	00f407b3          	add	a5,s0,a5
   111a8:	0047a703          	lw	a4,4(a5)
   111ac:	00d62623          	sw	a3,12(a2)
   111b0:	00c6a423          	sw	a2,8(a3)
   111b4:	00176713          	ori	a4,a4,1
   111b8:	00090513          	mv	a0,s2
   111bc:	00e7a223          	sw	a4,4(a5)
   111c0:	748000ef          	jal	11908 <__malloc_unlock>
   111c4:	00840513          	addi	a0,s0,8
   111c8:	02c12083          	lw	ra,44(sp)
   111cc:	02812403          	lw	s0,40(sp)
   111d0:	02412483          	lw	s1,36(sp)
   111d4:	02012903          	lw	s2,32(sp)
   111d8:	01c12983          	lw	s3,28(sp)
   111dc:	03010113          	addi	sp,sp,48
   111e0:	00008067          	ret
   111e4:	ff87f493          	andi	s1,a5,-8
   111e8:	1807c663          	bltz	a5,11374 <_malloc_r+0x238>
   111ec:	18b4e463          	bltu	s1,a1,11374 <_malloc_r+0x238>
   111f0:	714000ef          	jal	11904 <__malloc_lock>
   111f4:	1f700793          	li	a5,503
   111f8:	4097f063          	bgeu	a5,s1,115f8 <_malloc_r+0x4bc>
   111fc:	0094d793          	srli	a5,s1,0x9
   11200:	18078263          	beqz	a5,11384 <_malloc_r+0x248>
   11204:	00400713          	li	a4,4
   11208:	34f76663          	bltu	a4,a5,11554 <_malloc_r+0x418>
   1120c:	0064d793          	srli	a5,s1,0x6
   11210:	03978593          	addi	a1,a5,57
   11214:	03878813          	addi	a6,a5,56
   11218:	00359613          	slli	a2,a1,0x3
   1121c:	00001997          	auipc	s3,0x1
   11220:	09498993          	addi	s3,s3,148 # 122b0 <__malloc_av_>
   11224:	00c98633          	add	a2,s3,a2
   11228:	00462403          	lw	s0,4(a2)
   1122c:	ff860613          	addi	a2,a2,-8
   11230:	02860863          	beq	a2,s0,11260 <_malloc_r+0x124>
   11234:	00f00513          	li	a0,15
   11238:	0140006f          	j	1124c <_malloc_r+0x110>
   1123c:	00c42683          	lw	a3,12(s0)
   11240:	28075e63          	bgez	a4,114dc <_malloc_r+0x3a0>
   11244:	00d60e63          	beq	a2,a3,11260 <_malloc_r+0x124>
   11248:	00068413          	mv	s0,a3
   1124c:	00442783          	lw	a5,4(s0)
   11250:	ffc7f793          	andi	a5,a5,-4
   11254:	40978733          	sub	a4,a5,s1
   11258:	fee552e3          	bge	a0,a4,1123c <_malloc_r+0x100>
   1125c:	00080593          	mv	a1,a6
   11260:	0109a403          	lw	s0,16(s3)
   11264:	00001897          	auipc	a7,0x1
   11268:	05488893          	addi	a7,a7,84 # 122b8 <__malloc_av_+0x8>
   1126c:	27140463          	beq	s0,a7,114d4 <_malloc_r+0x398>
   11270:	00442783          	lw	a5,4(s0)
   11274:	00f00693          	li	a3,15
   11278:	ffc7f793          	andi	a5,a5,-4
   1127c:	40978733          	sub	a4,a5,s1
   11280:	38e6c263          	blt	a3,a4,11604 <_malloc_r+0x4c8>
   11284:	0119aa23          	sw	a7,20(s3)
   11288:	0119a823          	sw	a7,16(s3)
   1128c:	34075663          	bgez	a4,115d8 <_malloc_r+0x49c>
   11290:	1ff00713          	li	a4,511
   11294:	0049a503          	lw	a0,4(s3)
   11298:	24f76e63          	bltu	a4,a5,114f4 <_malloc_r+0x3b8>
   1129c:	ff87f713          	andi	a4,a5,-8
   112a0:	00870713          	addi	a4,a4,8
   112a4:	00e98733          	add	a4,s3,a4
   112a8:	00072683          	lw	a3,0(a4)
   112ac:	0057d613          	srli	a2,a5,0x5
   112b0:	00100793          	li	a5,1
   112b4:	00c797b3          	sll	a5,a5,a2
   112b8:	00f56533          	or	a0,a0,a5
   112bc:	ff870793          	addi	a5,a4,-8
   112c0:	00f42623          	sw	a5,12(s0)
   112c4:	00d42423          	sw	a3,8(s0)
   112c8:	00a9a223          	sw	a0,4(s3)
   112cc:	00872023          	sw	s0,0(a4)
   112d0:	0086a623          	sw	s0,12(a3)
   112d4:	4025d793          	srai	a5,a1,0x2
   112d8:	00100613          	li	a2,1
   112dc:	00f61633          	sll	a2,a2,a5
   112e0:	0ac56a63          	bltu	a0,a2,11394 <_malloc_r+0x258>
   112e4:	00a677b3          	and	a5,a2,a0
   112e8:	02079463          	bnez	a5,11310 <_malloc_r+0x1d4>
   112ec:	00161613          	slli	a2,a2,0x1
   112f0:	ffc5f593          	andi	a1,a1,-4
   112f4:	00a677b3          	and	a5,a2,a0
   112f8:	00458593          	addi	a1,a1,4
   112fc:	00079a63          	bnez	a5,11310 <_malloc_r+0x1d4>
   11300:	00161613          	slli	a2,a2,0x1
   11304:	00a677b3          	and	a5,a2,a0
   11308:	00458593          	addi	a1,a1,4
   1130c:	fe078ae3          	beqz	a5,11300 <_malloc_r+0x1c4>
   11310:	00f00813          	li	a6,15
   11314:	00359313          	slli	t1,a1,0x3
   11318:	00698333          	add	t1,s3,t1
   1131c:	00030513          	mv	a0,t1
   11320:	00c52783          	lw	a5,12(a0)
   11324:	00058e13          	mv	t3,a1
   11328:	24f50863          	beq	a0,a5,11578 <_malloc_r+0x43c>
   1132c:	0047a703          	lw	a4,4(a5)
   11330:	00078413          	mv	s0,a5
   11334:	00c7a783          	lw	a5,12(a5)
   11338:	ffc77713          	andi	a4,a4,-4
   1133c:	409706b3          	sub	a3,a4,s1
   11340:	24d84863          	blt	a6,a3,11590 <_malloc_r+0x454>
   11344:	fe06c2e3          	bltz	a3,11328 <_malloc_r+0x1ec>
   11348:	00e40733          	add	a4,s0,a4
   1134c:	00472683          	lw	a3,4(a4)
   11350:	00842603          	lw	a2,8(s0)
   11354:	00090513          	mv	a0,s2
   11358:	0016e693          	ori	a3,a3,1
   1135c:	00d72223          	sw	a3,4(a4)
   11360:	00f62623          	sw	a5,12(a2)
   11364:	00c7a423          	sw	a2,8(a5)
   11368:	5a0000ef          	jal	11908 <__malloc_unlock>
   1136c:	00840513          	addi	a0,s0,8
   11370:	e59ff06f          	j	111c8 <_malloc_r+0x8c>
   11374:	00c00793          	li	a5,12
   11378:	00f92023          	sw	a5,0(s2)
   1137c:	00000513          	li	a0,0
   11380:	e49ff06f          	j	111c8 <_malloc_r+0x8c>
   11384:	20000613          	li	a2,512
   11388:	04000593          	li	a1,64
   1138c:	03f00813          	li	a6,63
   11390:	e8dff06f          	j	1121c <_malloc_r+0xe0>
   11394:	0089a403          	lw	s0,8(s3)
   11398:	01612823          	sw	s6,16(sp)
   1139c:	00442783          	lw	a5,4(s0)
   113a0:	ffc7fb13          	andi	s6,a5,-4
   113a4:	009b6863          	bltu	s6,s1,113b4 <_malloc_r+0x278>
   113a8:	409b0733          	sub	a4,s6,s1
   113ac:	00f00793          	li	a5,15
   113b0:	0ee7c063          	blt	a5,a4,11490 <_malloc_r+0x354>
   113b4:	01912223          	sw	s9,4(sp)
   113b8:	e4018c93          	addi	s9,gp,-448 # 126c0 <__malloc_sbrk_base>
   113bc:	000ca703          	lw	a4,0(s9)
   113c0:	01412c23          	sw	s4,24(sp)
   113c4:	01512a23          	sw	s5,20(sp)
   113c8:	01712623          	sw	s7,12(sp)
   113cc:	e5c1aa83          	lw	s5,-420(gp) # 126dc <__malloc_top_pad>
   113d0:	fff00793          	li	a5,-1
   113d4:	01640a33          	add	s4,s0,s6
   113d8:	01548ab3          	add	s5,s1,s5
   113dc:	3cf70a63          	beq	a4,a5,117b0 <_malloc_r+0x674>
   113e0:	000017b7          	lui	a5,0x1
   113e4:	00f78793          	addi	a5,a5,15 # 100f <exit-0xf0a5>
   113e8:	00fa8ab3          	add	s5,s5,a5
   113ec:	fffff7b7          	lui	a5,0xfffff
   113f0:	00fafab3          	and	s5,s5,a5
   113f4:	000a8593          	mv	a1,s5
   113f8:	00090513          	mv	a0,s2
   113fc:	151000ef          	jal	11d4c <_sbrk_r>
   11400:	fff00793          	li	a5,-1
   11404:	00050b93          	mv	s7,a0
   11408:	44f50e63          	beq	a0,a5,11864 <_malloc_r+0x728>
   1140c:	01812423          	sw	s8,8(sp)
   11410:	25456263          	bltu	a0,s4,11654 <_malloc_r+0x518>
   11414:	07818c13          	addi	s8,gp,120 # 128f8 <__malloc_current_mallinfo>
   11418:	000c2583          	lw	a1,0(s8)
   1141c:	00ba85b3          	add	a1,s5,a1
   11420:	00bc2023          	sw	a1,0(s8)
   11424:	00058713          	mv	a4,a1
   11428:	2aaa1a63          	bne	s4,a0,116dc <_malloc_r+0x5a0>
   1142c:	01451793          	slli	a5,a0,0x14
   11430:	2a079663          	bnez	a5,116dc <_malloc_r+0x5a0>
   11434:	0089ab83          	lw	s7,8(s3)
   11438:	015b07b3          	add	a5,s6,s5
   1143c:	0017e793          	ori	a5,a5,1
   11440:	00fba223          	sw	a5,4(s7)
   11444:	e5818713          	addi	a4,gp,-424 # 126d8 <__malloc_max_sbrked_mem>
   11448:	00072683          	lw	a3,0(a4)
   1144c:	00b6f463          	bgeu	a3,a1,11454 <_malloc_r+0x318>
   11450:	00b72023          	sw	a1,0(a4)
   11454:	e5418713          	addi	a4,gp,-428 # 126d4 <__malloc_max_total_mem>
   11458:	00072683          	lw	a3,0(a4)
   1145c:	00b6f463          	bgeu	a3,a1,11464 <_malloc_r+0x328>
   11460:	00b72023          	sw	a1,0(a4)
   11464:	00812c03          	lw	s8,8(sp)
   11468:	000b8413          	mv	s0,s7
   1146c:	ffc7f793          	andi	a5,a5,-4
   11470:	40978733          	sub	a4,a5,s1
   11474:	3897ea63          	bltu	a5,s1,11808 <_malloc_r+0x6cc>
   11478:	00f00793          	li	a5,15
   1147c:	38e7d663          	bge	a5,a4,11808 <_malloc_r+0x6cc>
   11480:	01812a03          	lw	s4,24(sp)
   11484:	01412a83          	lw	s5,20(sp)
   11488:	00c12b83          	lw	s7,12(sp)
   1148c:	00412c83          	lw	s9,4(sp)
   11490:	0014e793          	ori	a5,s1,1
   11494:	00f42223          	sw	a5,4(s0)
   11498:	009404b3          	add	s1,s0,s1
   1149c:	0099a423          	sw	s1,8(s3)
   114a0:	00176713          	ori	a4,a4,1
   114a4:	00090513          	mv	a0,s2
   114a8:	00e4a223          	sw	a4,4(s1)
   114ac:	45c000ef          	jal	11908 <__malloc_unlock>
   114b0:	02c12083          	lw	ra,44(sp)
   114b4:	00840513          	addi	a0,s0,8
   114b8:	02812403          	lw	s0,40(sp)
   114bc:	01012b03          	lw	s6,16(sp)
   114c0:	02412483          	lw	s1,36(sp)
   114c4:	02012903          	lw	s2,32(sp)
   114c8:	01c12983          	lw	s3,28(sp)
   114cc:	03010113          	addi	sp,sp,48
   114d0:	00008067          	ret
   114d4:	0049a503          	lw	a0,4(s3)
   114d8:	dfdff06f          	j	112d4 <_malloc_r+0x198>
   114dc:	00842603          	lw	a2,8(s0)
   114e0:	cc5ff06f          	j	111a4 <_malloc_r+0x68>
   114e4:	00c7a403          	lw	s0,12(a5) # fffff00c <__BSS_END__+0xfffec55c>
   114e8:	00258593          	addi	a1,a1,2
   114ec:	d6878ae3          	beq	a5,s0,11260 <_malloc_r+0x124>
   114f0:	ca5ff06f          	j	11194 <_malloc_r+0x58>
   114f4:	0097d713          	srli	a4,a5,0x9
   114f8:	00400693          	li	a3,4
   114fc:	14e6f263          	bgeu	a3,a4,11640 <_malloc_r+0x504>
   11500:	01400693          	li	a3,20
   11504:	32e6e463          	bltu	a3,a4,1182c <_malloc_r+0x6f0>
   11508:	05c70613          	addi	a2,a4,92
   1150c:	05b70693          	addi	a3,a4,91
   11510:	00361613          	slli	a2,a2,0x3
   11514:	00c98633          	add	a2,s3,a2
   11518:	00062703          	lw	a4,0(a2)
   1151c:	ff860613          	addi	a2,a2,-8
   11520:	00e61863          	bne	a2,a4,11530 <_malloc_r+0x3f4>
   11524:	2940006f          	j	117b8 <_malloc_r+0x67c>
   11528:	00872703          	lw	a4,8(a4)
   1152c:	00e60863          	beq	a2,a4,1153c <_malloc_r+0x400>
   11530:	00472683          	lw	a3,4(a4)
   11534:	ffc6f693          	andi	a3,a3,-4
   11538:	fed7e8e3          	bltu	a5,a3,11528 <_malloc_r+0x3ec>
   1153c:	00c72603          	lw	a2,12(a4)
   11540:	00c42623          	sw	a2,12(s0)
   11544:	00e42423          	sw	a4,8(s0)
   11548:	00862423          	sw	s0,8(a2)
   1154c:	00872623          	sw	s0,12(a4)
   11550:	d85ff06f          	j	112d4 <_malloc_r+0x198>
   11554:	01400713          	li	a4,20
   11558:	10f77863          	bgeu	a4,a5,11668 <_malloc_r+0x52c>
   1155c:	05400713          	li	a4,84
   11560:	2ef76463          	bltu	a4,a5,11848 <_malloc_r+0x70c>
   11564:	00c4d793          	srli	a5,s1,0xc
   11568:	06f78593          	addi	a1,a5,111
   1156c:	06e78813          	addi	a6,a5,110
   11570:	00359613          	slli	a2,a1,0x3
   11574:	ca9ff06f          	j	1121c <_malloc_r+0xe0>
   11578:	001e0e13          	addi	t3,t3,1
   1157c:	003e7793          	andi	a5,t3,3
   11580:	00850513          	addi	a0,a0,8
   11584:	10078063          	beqz	a5,11684 <_malloc_r+0x548>
   11588:	00c52783          	lw	a5,12(a0)
   1158c:	d9dff06f          	j	11328 <_malloc_r+0x1ec>
   11590:	00842603          	lw	a2,8(s0)
   11594:	0014e593          	ori	a1,s1,1
   11598:	00b42223          	sw	a1,4(s0)
   1159c:	00f62623          	sw	a5,12(a2)
   115a0:	00c7a423          	sw	a2,8(a5)
   115a4:	009404b3          	add	s1,s0,s1
   115a8:	0099aa23          	sw	s1,20(s3)
   115ac:	0099a823          	sw	s1,16(s3)
   115b0:	0016e793          	ori	a5,a3,1
   115b4:	0114a623          	sw	a7,12(s1)
   115b8:	0114a423          	sw	a7,8(s1)
   115bc:	00f4a223          	sw	a5,4(s1)
   115c0:	00e40733          	add	a4,s0,a4
   115c4:	00090513          	mv	a0,s2
   115c8:	00d72023          	sw	a3,0(a4)
   115cc:	33c000ef          	jal	11908 <__malloc_unlock>
   115d0:	00840513          	addi	a0,s0,8
   115d4:	bf5ff06f          	j	111c8 <_malloc_r+0x8c>
   115d8:	00f407b3          	add	a5,s0,a5
   115dc:	0047a703          	lw	a4,4(a5)
   115e0:	00090513          	mv	a0,s2
   115e4:	00176713          	ori	a4,a4,1
   115e8:	00e7a223          	sw	a4,4(a5)
   115ec:	31c000ef          	jal	11908 <__malloc_unlock>
   115f0:	00840513          	addi	a0,s0,8
   115f4:	bd5ff06f          	j	111c8 <_malloc_r+0x8c>
   115f8:	0034d593          	srli	a1,s1,0x3
   115fc:	00848793          	addi	a5,s1,8
   11600:	b7dff06f          	j	1117c <_malloc_r+0x40>
   11604:	0014e693          	ori	a3,s1,1
   11608:	00d42223          	sw	a3,4(s0)
   1160c:	009404b3          	add	s1,s0,s1
   11610:	0099aa23          	sw	s1,20(s3)
   11614:	0099a823          	sw	s1,16(s3)
   11618:	00176693          	ori	a3,a4,1
   1161c:	0114a623          	sw	a7,12(s1)
   11620:	0114a423          	sw	a7,8(s1)
   11624:	00d4a223          	sw	a3,4(s1)
   11628:	00f407b3          	add	a5,s0,a5
   1162c:	00090513          	mv	a0,s2
   11630:	00e7a023          	sw	a4,0(a5)
   11634:	2d4000ef          	jal	11908 <__malloc_unlock>
   11638:	00840513          	addi	a0,s0,8
   1163c:	b8dff06f          	j	111c8 <_malloc_r+0x8c>
   11640:	0067d713          	srli	a4,a5,0x6
   11644:	03970613          	addi	a2,a4,57
   11648:	03870693          	addi	a3,a4,56
   1164c:	00361613          	slli	a2,a2,0x3
   11650:	ec5ff06f          	j	11514 <_malloc_r+0x3d8>
   11654:	07340c63          	beq	s0,s3,116cc <_malloc_r+0x590>
   11658:	0089a403          	lw	s0,8(s3)
   1165c:	00812c03          	lw	s8,8(sp)
   11660:	00442783          	lw	a5,4(s0)
   11664:	e09ff06f          	j	1146c <_malloc_r+0x330>
   11668:	05c78593          	addi	a1,a5,92
   1166c:	05b78813          	addi	a6,a5,91
   11670:	00359613          	slli	a2,a1,0x3
   11674:	ba9ff06f          	j	1121c <_malloc_r+0xe0>
   11678:	00832783          	lw	a5,8(t1)
   1167c:	fff58593          	addi	a1,a1,-1
   11680:	26679e63          	bne	a5,t1,118fc <_malloc_r+0x7c0>
   11684:	0035f793          	andi	a5,a1,3
   11688:	ff830313          	addi	t1,t1,-8
   1168c:	fe0796e3          	bnez	a5,11678 <_malloc_r+0x53c>
   11690:	0049a703          	lw	a4,4(s3)
   11694:	fff64793          	not	a5,a2
   11698:	00e7f7b3          	and	a5,a5,a4
   1169c:	00f9a223          	sw	a5,4(s3)
   116a0:	00161613          	slli	a2,a2,0x1
   116a4:	cec7e8e3          	bltu	a5,a2,11394 <_malloc_r+0x258>
   116a8:	ce0606e3          	beqz	a2,11394 <_malloc_r+0x258>
   116ac:	00f67733          	and	a4,a2,a5
   116b0:	00071a63          	bnez	a4,116c4 <_malloc_r+0x588>
   116b4:	00161613          	slli	a2,a2,0x1
   116b8:	00f67733          	and	a4,a2,a5
   116bc:	004e0e13          	addi	t3,t3,4
   116c0:	fe070ae3          	beqz	a4,116b4 <_malloc_r+0x578>
   116c4:	000e0593          	mv	a1,t3
   116c8:	c4dff06f          	j	11314 <_malloc_r+0x1d8>
   116cc:	07818c13          	addi	s8,gp,120 # 128f8 <__malloc_current_mallinfo>
   116d0:	000c2703          	lw	a4,0(s8)
   116d4:	00ea8733          	add	a4,s5,a4
   116d8:	00ec2023          	sw	a4,0(s8)
   116dc:	000ca683          	lw	a3,0(s9)
   116e0:	fff00793          	li	a5,-1
   116e4:	18f68663          	beq	a3,a5,11870 <_malloc_r+0x734>
   116e8:	414b87b3          	sub	a5,s7,s4
   116ec:	00e787b3          	add	a5,a5,a4
   116f0:	00fc2023          	sw	a5,0(s8)
   116f4:	007bfc93          	andi	s9,s7,7
   116f8:	0c0c8c63          	beqz	s9,117d0 <_malloc_r+0x694>
   116fc:	419b8bb3          	sub	s7,s7,s9
   11700:	000017b7          	lui	a5,0x1
   11704:	00878793          	addi	a5,a5,8 # 1008 <exit-0xf0ac>
   11708:	008b8b93          	addi	s7,s7,8
   1170c:	419785b3          	sub	a1,a5,s9
   11710:	015b8ab3          	add	s5,s7,s5
   11714:	415585b3          	sub	a1,a1,s5
   11718:	01459593          	slli	a1,a1,0x14
   1171c:	0145da13          	srli	s4,a1,0x14
   11720:	000a0593          	mv	a1,s4
   11724:	00090513          	mv	a0,s2
   11728:	624000ef          	jal	11d4c <_sbrk_r>
   1172c:	fff00793          	li	a5,-1
   11730:	18f50063          	beq	a0,a5,118b0 <_malloc_r+0x774>
   11734:	41750533          	sub	a0,a0,s7
   11738:	01450ab3          	add	s5,a0,s4
   1173c:	000c2703          	lw	a4,0(s8)
   11740:	0179a423          	sw	s7,8(s3)
   11744:	001ae793          	ori	a5,s5,1
   11748:	00ea05b3          	add	a1,s4,a4
   1174c:	00bc2023          	sw	a1,0(s8)
   11750:	00fba223          	sw	a5,4(s7)
   11754:	cf3408e3          	beq	s0,s3,11444 <_malloc_r+0x308>
   11758:	00f00693          	li	a3,15
   1175c:	0b66f063          	bgeu	a3,s6,117fc <_malloc_r+0x6c0>
   11760:	00442703          	lw	a4,4(s0)
   11764:	ff4b0793          	addi	a5,s6,-12
   11768:	ff87f793          	andi	a5,a5,-8
   1176c:	00177713          	andi	a4,a4,1
   11770:	00f76733          	or	a4,a4,a5
   11774:	00e42223          	sw	a4,4(s0)
   11778:	00500613          	li	a2,5
   1177c:	00f40733          	add	a4,s0,a5
   11780:	00c72223          	sw	a2,4(a4)
   11784:	00c72423          	sw	a2,8(a4)
   11788:	00f6e663          	bltu	a3,a5,11794 <_malloc_r+0x658>
   1178c:	004ba783          	lw	a5,4(s7)
   11790:	cb5ff06f          	j	11444 <_malloc_r+0x308>
   11794:	00840593          	addi	a1,s0,8
   11798:	00090513          	mv	a0,s2
   1179c:	e9cff0ef          	jal	10e38 <_free_r>
   117a0:	0089ab83          	lw	s7,8(s3)
   117a4:	000c2583          	lw	a1,0(s8)
   117a8:	004ba783          	lw	a5,4(s7)
   117ac:	c99ff06f          	j	11444 <_malloc_r+0x308>
   117b0:	010a8a93          	addi	s5,s5,16
   117b4:	c41ff06f          	j	113f4 <_malloc_r+0x2b8>
   117b8:	4026d693          	srai	a3,a3,0x2
   117bc:	00100793          	li	a5,1
   117c0:	00d797b3          	sll	a5,a5,a3
   117c4:	00f56533          	or	a0,a0,a5
   117c8:	00a9a223          	sw	a0,4(s3)
   117cc:	d75ff06f          	j	11540 <_malloc_r+0x404>
   117d0:	015b85b3          	add	a1,s7,s5
   117d4:	40b005b3          	neg	a1,a1
   117d8:	01459593          	slli	a1,a1,0x14
   117dc:	0145da13          	srli	s4,a1,0x14
   117e0:	000a0593          	mv	a1,s4
   117e4:	00090513          	mv	a0,s2
   117e8:	564000ef          	jal	11d4c <_sbrk_r>
   117ec:	fff00793          	li	a5,-1
   117f0:	f4f512e3          	bne	a0,a5,11734 <_malloc_r+0x5f8>
   117f4:	00000a13          	li	s4,0
   117f8:	f45ff06f          	j	1173c <_malloc_r+0x600>
   117fc:	00812c03          	lw	s8,8(sp)
   11800:	00100793          	li	a5,1
   11804:	00fba223          	sw	a5,4(s7)
   11808:	00090513          	mv	a0,s2
   1180c:	0fc000ef          	jal	11908 <__malloc_unlock>
   11810:	00000513          	li	a0,0
   11814:	01812a03          	lw	s4,24(sp)
   11818:	01412a83          	lw	s5,20(sp)
   1181c:	01012b03          	lw	s6,16(sp)
   11820:	00c12b83          	lw	s7,12(sp)
   11824:	00412c83          	lw	s9,4(sp)
   11828:	9a1ff06f          	j	111c8 <_malloc_r+0x8c>
   1182c:	05400693          	li	a3,84
   11830:	04e6e463          	bltu	a3,a4,11878 <_malloc_r+0x73c>
   11834:	00c7d713          	srli	a4,a5,0xc
   11838:	06f70613          	addi	a2,a4,111
   1183c:	06e70693          	addi	a3,a4,110
   11840:	00361613          	slli	a2,a2,0x3
   11844:	cd1ff06f          	j	11514 <_malloc_r+0x3d8>
   11848:	15400713          	li	a4,340
   1184c:	04f76463          	bltu	a4,a5,11894 <_malloc_r+0x758>
   11850:	00f4d793          	srli	a5,s1,0xf
   11854:	07878593          	addi	a1,a5,120
   11858:	07778813          	addi	a6,a5,119
   1185c:	00359613          	slli	a2,a1,0x3
   11860:	9bdff06f          	j	1121c <_malloc_r+0xe0>
   11864:	0089a403          	lw	s0,8(s3)
   11868:	00442783          	lw	a5,4(s0)
   1186c:	c01ff06f          	j	1146c <_malloc_r+0x330>
   11870:	017ca023          	sw	s7,0(s9)
   11874:	e81ff06f          	j	116f4 <_malloc_r+0x5b8>
   11878:	15400693          	li	a3,340
   1187c:	04e6e463          	bltu	a3,a4,118c4 <_malloc_r+0x788>
   11880:	00f7d713          	srli	a4,a5,0xf
   11884:	07870613          	addi	a2,a4,120
   11888:	07770693          	addi	a3,a4,119
   1188c:	00361613          	slli	a2,a2,0x3
   11890:	c85ff06f          	j	11514 <_malloc_r+0x3d8>
   11894:	55400713          	li	a4,1364
   11898:	04f76463          	bltu	a4,a5,118e0 <_malloc_r+0x7a4>
   1189c:	0124d793          	srli	a5,s1,0x12
   118a0:	07d78593          	addi	a1,a5,125
   118a4:	07c78813          	addi	a6,a5,124
   118a8:	00359613          	slli	a2,a1,0x3
   118ac:	971ff06f          	j	1121c <_malloc_r+0xe0>
   118b0:	ff8c8c93          	addi	s9,s9,-8
   118b4:	019a8ab3          	add	s5,s5,s9
   118b8:	417a8ab3          	sub	s5,s5,s7
   118bc:	00000a13          	li	s4,0
   118c0:	e7dff06f          	j	1173c <_malloc_r+0x600>
   118c4:	55400693          	li	a3,1364
   118c8:	02e6e463          	bltu	a3,a4,118f0 <_malloc_r+0x7b4>
   118cc:	0127d713          	srli	a4,a5,0x12
   118d0:	07d70613          	addi	a2,a4,125
   118d4:	07c70693          	addi	a3,a4,124
   118d8:	00361613          	slli	a2,a2,0x3
   118dc:	c39ff06f          	j	11514 <_malloc_r+0x3d8>
   118e0:	3f800613          	li	a2,1016
   118e4:	07f00593          	li	a1,127
   118e8:	07e00813          	li	a6,126
   118ec:	931ff06f          	j	1121c <_malloc_r+0xe0>
   118f0:	3f800613          	li	a2,1016
   118f4:	07e00693          	li	a3,126
   118f8:	c1dff06f          	j	11514 <_malloc_r+0x3d8>
   118fc:	0049a783          	lw	a5,4(s3)
   11900:	da1ff06f          	j	116a0 <_malloc_r+0x564>

00011904 <__malloc_lock>:
   11904:	00008067          	ret

00011908 <__malloc_unlock>:
   11908:	00008067          	ret

0001190c <_fclose_r>:
   1190c:	ff010113          	addi	sp,sp,-16
   11910:	00112623          	sw	ra,12(sp)
   11914:	01212023          	sw	s2,0(sp)
   11918:	02058863          	beqz	a1,11948 <_fclose_r+0x3c>
   1191c:	00812423          	sw	s0,8(sp)
   11920:	00912223          	sw	s1,4(sp)
   11924:	00058413          	mv	s0,a1
   11928:	00050493          	mv	s1,a0
   1192c:	00050663          	beqz	a0,11938 <_fclose_r+0x2c>
   11930:	03452783          	lw	a5,52(a0)
   11934:	0c078c63          	beqz	a5,11a0c <_fclose_r+0x100>
   11938:	00c41783          	lh	a5,12(s0)
   1193c:	02079263          	bnez	a5,11960 <_fclose_r+0x54>
   11940:	00812403          	lw	s0,8(sp)
   11944:	00412483          	lw	s1,4(sp)
   11948:	00c12083          	lw	ra,12(sp)
   1194c:	00000913          	li	s2,0
   11950:	00090513          	mv	a0,s2
   11954:	00012903          	lw	s2,0(sp)
   11958:	01010113          	addi	sp,sp,16
   1195c:	00008067          	ret
   11960:	00040593          	mv	a1,s0
   11964:	00048513          	mv	a0,s1
   11968:	0b8000ef          	jal	11a20 <__sflush_r>
   1196c:	02c42783          	lw	a5,44(s0)
   11970:	00050913          	mv	s2,a0
   11974:	00078a63          	beqz	a5,11988 <_fclose_r+0x7c>
   11978:	01c42583          	lw	a1,28(s0)
   1197c:	00048513          	mv	a0,s1
   11980:	000780e7          	jalr	a5
   11984:	06054463          	bltz	a0,119ec <_fclose_r+0xe0>
   11988:	00c45783          	lhu	a5,12(s0)
   1198c:	0807f793          	andi	a5,a5,128
   11990:	06079663          	bnez	a5,119fc <_fclose_r+0xf0>
   11994:	03042583          	lw	a1,48(s0)
   11998:	00058c63          	beqz	a1,119b0 <_fclose_r+0xa4>
   1199c:	04040793          	addi	a5,s0,64
   119a0:	00f58663          	beq	a1,a5,119ac <_fclose_r+0xa0>
   119a4:	00048513          	mv	a0,s1
   119a8:	c90ff0ef          	jal	10e38 <_free_r>
   119ac:	02042823          	sw	zero,48(s0)
   119b0:	04442583          	lw	a1,68(s0)
   119b4:	00058863          	beqz	a1,119c4 <_fclose_r+0xb8>
   119b8:	00048513          	mv	a0,s1
   119bc:	c7cff0ef          	jal	10e38 <_free_r>
   119c0:	04042223          	sw	zero,68(s0)
   119c4:	c09fe0ef          	jal	105cc <__sfp_lock_acquire>
   119c8:	00041623          	sh	zero,12(s0)
   119cc:	c05fe0ef          	jal	105d0 <__sfp_lock_release>
   119d0:	00c12083          	lw	ra,12(sp)
   119d4:	00812403          	lw	s0,8(sp)
   119d8:	00412483          	lw	s1,4(sp)
   119dc:	00090513          	mv	a0,s2
   119e0:	00012903          	lw	s2,0(sp)
   119e4:	01010113          	addi	sp,sp,16
   119e8:	00008067          	ret
   119ec:	00c45783          	lhu	a5,12(s0)
   119f0:	fff00913          	li	s2,-1
   119f4:	0807f793          	andi	a5,a5,128
   119f8:	f8078ee3          	beqz	a5,11994 <_fclose_r+0x88>
   119fc:	01042583          	lw	a1,16(s0)
   11a00:	00048513          	mv	a0,s1
   11a04:	c34ff0ef          	jal	10e38 <_free_r>
   11a08:	f8dff06f          	j	11994 <_fclose_r+0x88>
   11a0c:	b9dfe0ef          	jal	105a8 <__sinit>
   11a10:	f29ff06f          	j	11938 <_fclose_r+0x2c>

00011a14 <fclose>:
   11a14:	00050593          	mv	a1,a0
   11a18:	e3c1a503          	lw	a0,-452(gp) # 126bc <_impure_ptr>
   11a1c:	ef1ff06f          	j	1190c <_fclose_r>

00011a20 <__sflush_r>:
   11a20:	00c59703          	lh	a4,12(a1)
   11a24:	fe010113          	addi	sp,sp,-32
   11a28:	00812c23          	sw	s0,24(sp)
   11a2c:	01312623          	sw	s3,12(sp)
   11a30:	00112e23          	sw	ra,28(sp)
   11a34:	00877793          	andi	a5,a4,8
   11a38:	00058413          	mv	s0,a1
   11a3c:	00050993          	mv	s3,a0
   11a40:	12079063          	bnez	a5,11b60 <__sflush_r+0x140>
   11a44:	000017b7          	lui	a5,0x1
   11a48:	80078793          	addi	a5,a5,-2048 # 800 <exit-0xf8b4>
   11a4c:	0045a683          	lw	a3,4(a1)
   11a50:	00f767b3          	or	a5,a4,a5
   11a54:	00f59623          	sh	a5,12(a1)
   11a58:	18d05263          	blez	a3,11bdc <__sflush_r+0x1bc>
   11a5c:	02842803          	lw	a6,40(s0)
   11a60:	0e080463          	beqz	a6,11b48 <__sflush_r+0x128>
   11a64:	00912a23          	sw	s1,20(sp)
   11a68:	01371693          	slli	a3,a4,0x13
   11a6c:	0009a483          	lw	s1,0(s3)
   11a70:	0009a023          	sw	zero,0(s3)
   11a74:	01c42583          	lw	a1,28(s0)
   11a78:	1606ce63          	bltz	a3,11bf4 <__sflush_r+0x1d4>
   11a7c:	00000613          	li	a2,0
   11a80:	00100693          	li	a3,1
   11a84:	00098513          	mv	a0,s3
   11a88:	000800e7          	jalr	a6
   11a8c:	fff00793          	li	a5,-1
   11a90:	00050613          	mv	a2,a0
   11a94:	1af50463          	beq	a0,a5,11c3c <__sflush_r+0x21c>
   11a98:	00c41783          	lh	a5,12(s0)
   11a9c:	02842803          	lw	a6,40(s0)
   11aa0:	01c42583          	lw	a1,28(s0)
   11aa4:	0047f793          	andi	a5,a5,4
   11aa8:	00078e63          	beqz	a5,11ac4 <__sflush_r+0xa4>
   11aac:	00442703          	lw	a4,4(s0)
   11ab0:	03042783          	lw	a5,48(s0)
   11ab4:	40e60633          	sub	a2,a2,a4
   11ab8:	00078663          	beqz	a5,11ac4 <__sflush_r+0xa4>
   11abc:	03c42783          	lw	a5,60(s0)
   11ac0:	40f60633          	sub	a2,a2,a5
   11ac4:	00000693          	li	a3,0
   11ac8:	00098513          	mv	a0,s3
   11acc:	000800e7          	jalr	a6
   11ad0:	fff00793          	li	a5,-1
   11ad4:	12f51463          	bne	a0,a5,11bfc <__sflush_r+0x1dc>
   11ad8:	0009a683          	lw	a3,0(s3)
   11adc:	01d00793          	li	a5,29
   11ae0:	00c41703          	lh	a4,12(s0)
   11ae4:	16d7ea63          	bltu	a5,a3,11c58 <__sflush_r+0x238>
   11ae8:	204007b7          	lui	a5,0x20400
   11aec:	00178793          	addi	a5,a5,1 # 20400001 <__BSS_END__+0x203ed551>
   11af0:	00d7d7b3          	srl	a5,a5,a3
   11af4:	0017f793          	andi	a5,a5,1
   11af8:	16078063          	beqz	a5,11c58 <__sflush_r+0x238>
   11afc:	01042603          	lw	a2,16(s0)
   11b00:	fffff7b7          	lui	a5,0xfffff
   11b04:	7ff78793          	addi	a5,a5,2047 # fffff7ff <__BSS_END__+0xfffecd4f>
   11b08:	00f777b3          	and	a5,a4,a5
   11b0c:	00f41623          	sh	a5,12(s0)
   11b10:	00042223          	sw	zero,4(s0)
   11b14:	00c42023          	sw	a2,0(s0)
   11b18:	01371793          	slli	a5,a4,0x13
   11b1c:	0007d463          	bgez	a5,11b24 <__sflush_r+0x104>
   11b20:	10068263          	beqz	a3,11c24 <__sflush_r+0x204>
   11b24:	03042583          	lw	a1,48(s0)
   11b28:	0099a023          	sw	s1,0(s3)
   11b2c:	10058463          	beqz	a1,11c34 <__sflush_r+0x214>
   11b30:	04040793          	addi	a5,s0,64
   11b34:	00f58663          	beq	a1,a5,11b40 <__sflush_r+0x120>
   11b38:	00098513          	mv	a0,s3
   11b3c:	afcff0ef          	jal	10e38 <_free_r>
   11b40:	01412483          	lw	s1,20(sp)
   11b44:	02042823          	sw	zero,48(s0)
   11b48:	00000513          	li	a0,0
   11b4c:	01c12083          	lw	ra,28(sp)
   11b50:	01812403          	lw	s0,24(sp)
   11b54:	00c12983          	lw	s3,12(sp)
   11b58:	02010113          	addi	sp,sp,32
   11b5c:	00008067          	ret
   11b60:	01212823          	sw	s2,16(sp)
   11b64:	0105a903          	lw	s2,16(a1)
   11b68:	08090263          	beqz	s2,11bec <__sflush_r+0x1cc>
   11b6c:	00912a23          	sw	s1,20(sp)
   11b70:	0005a483          	lw	s1,0(a1)
   11b74:	00377713          	andi	a4,a4,3
   11b78:	0125a023          	sw	s2,0(a1)
   11b7c:	412484b3          	sub	s1,s1,s2
   11b80:	00000793          	li	a5,0
   11b84:	00071463          	bnez	a4,11b8c <__sflush_r+0x16c>
   11b88:	0145a783          	lw	a5,20(a1)
   11b8c:	00f42423          	sw	a5,8(s0)
   11b90:	00904863          	bgtz	s1,11ba0 <__sflush_r+0x180>
   11b94:	0540006f          	j	11be8 <__sflush_r+0x1c8>
   11b98:	00a90933          	add	s2,s2,a0
   11b9c:	04905663          	blez	s1,11be8 <__sflush_r+0x1c8>
   11ba0:	02442783          	lw	a5,36(s0)
   11ba4:	01c42583          	lw	a1,28(s0)
   11ba8:	00048693          	mv	a3,s1
   11bac:	00090613          	mv	a2,s2
   11bb0:	00098513          	mv	a0,s3
   11bb4:	000780e7          	jalr	a5
   11bb8:	40a484b3          	sub	s1,s1,a0
   11bbc:	fca04ee3          	bgtz	a0,11b98 <__sflush_r+0x178>
   11bc0:	00c41703          	lh	a4,12(s0)
   11bc4:	01012903          	lw	s2,16(sp)
   11bc8:	04076713          	ori	a4,a4,64
   11bcc:	01412483          	lw	s1,20(sp)
   11bd0:	00e41623          	sh	a4,12(s0)
   11bd4:	fff00513          	li	a0,-1
   11bd8:	f75ff06f          	j	11b4c <__sflush_r+0x12c>
   11bdc:	03c5a683          	lw	a3,60(a1)
   11be0:	e6d04ee3          	bgtz	a3,11a5c <__sflush_r+0x3c>
   11be4:	f65ff06f          	j	11b48 <__sflush_r+0x128>
   11be8:	01412483          	lw	s1,20(sp)
   11bec:	01012903          	lw	s2,16(sp)
   11bf0:	f59ff06f          	j	11b48 <__sflush_r+0x128>
   11bf4:	05042603          	lw	a2,80(s0)
   11bf8:	eadff06f          	j	11aa4 <__sflush_r+0x84>
   11bfc:	00c41703          	lh	a4,12(s0)
   11c00:	01042683          	lw	a3,16(s0)
   11c04:	fffff7b7          	lui	a5,0xfffff
   11c08:	7ff78793          	addi	a5,a5,2047 # fffff7ff <__BSS_END__+0xfffecd4f>
   11c0c:	00f777b3          	and	a5,a4,a5
   11c10:	00f41623          	sh	a5,12(s0)
   11c14:	00042223          	sw	zero,4(s0)
   11c18:	00d42023          	sw	a3,0(s0)
   11c1c:	01371793          	slli	a5,a4,0x13
   11c20:	f007d2e3          	bgez	a5,11b24 <__sflush_r+0x104>
   11c24:	03042583          	lw	a1,48(s0)
   11c28:	04a42823          	sw	a0,80(s0)
   11c2c:	0099a023          	sw	s1,0(s3)
   11c30:	f00590e3          	bnez	a1,11b30 <__sflush_r+0x110>
   11c34:	01412483          	lw	s1,20(sp)
   11c38:	f11ff06f          	j	11b48 <__sflush_r+0x128>
   11c3c:	0009a783          	lw	a5,0(s3)
   11c40:	e4078ce3          	beqz	a5,11a98 <__sflush_r+0x78>
   11c44:	01d00713          	li	a4,29
   11c48:	00e78c63          	beq	a5,a4,11c60 <__sflush_r+0x240>
   11c4c:	01600713          	li	a4,22
   11c50:	00e78863          	beq	a5,a4,11c60 <__sflush_r+0x240>
   11c54:	00c41703          	lh	a4,12(s0)
   11c58:	04076713          	ori	a4,a4,64
   11c5c:	f71ff06f          	j	11bcc <__sflush_r+0x1ac>
   11c60:	0099a023          	sw	s1,0(s3)
   11c64:	01412483          	lw	s1,20(sp)
   11c68:	ee1ff06f          	j	11b48 <__sflush_r+0x128>

00011c6c <_fflush_r>:
   11c6c:	fe010113          	addi	sp,sp,-32
   11c70:	00812c23          	sw	s0,24(sp)
   11c74:	00112e23          	sw	ra,28(sp)
   11c78:	00050413          	mv	s0,a0
   11c7c:	00050663          	beqz	a0,11c88 <_fflush_r+0x1c>
   11c80:	03452783          	lw	a5,52(a0)
   11c84:	02078a63          	beqz	a5,11cb8 <_fflush_r+0x4c>
   11c88:	00c59783          	lh	a5,12(a1)
   11c8c:	00079c63          	bnez	a5,11ca4 <_fflush_r+0x38>
   11c90:	01c12083          	lw	ra,28(sp)
   11c94:	01812403          	lw	s0,24(sp)
   11c98:	00000513          	li	a0,0
   11c9c:	02010113          	addi	sp,sp,32
   11ca0:	00008067          	ret
   11ca4:	00040513          	mv	a0,s0
   11ca8:	01812403          	lw	s0,24(sp)
   11cac:	01c12083          	lw	ra,28(sp)
   11cb0:	02010113          	addi	sp,sp,32
   11cb4:	d6dff06f          	j	11a20 <__sflush_r>
   11cb8:	00b12623          	sw	a1,12(sp)
   11cbc:	8edfe0ef          	jal	105a8 <__sinit>
   11cc0:	00c12583          	lw	a1,12(sp)
   11cc4:	fc5ff06f          	j	11c88 <_fflush_r+0x1c>

00011cc8 <fflush>:
   11cc8:	06050063          	beqz	a0,11d28 <fflush+0x60>
   11ccc:	00050593          	mv	a1,a0
   11cd0:	e3c1a503          	lw	a0,-452(gp) # 126bc <_impure_ptr>
   11cd4:	00050663          	beqz	a0,11ce0 <fflush+0x18>
   11cd8:	03452783          	lw	a5,52(a0)
   11cdc:	00078c63          	beqz	a5,11cf4 <fflush+0x2c>
   11ce0:	00c59783          	lh	a5,12(a1)
   11ce4:	00079663          	bnez	a5,11cf0 <fflush+0x28>
   11ce8:	00000513          	li	a0,0
   11cec:	00008067          	ret
   11cf0:	d31ff06f          	j	11a20 <__sflush_r>
   11cf4:	fe010113          	addi	sp,sp,-32
   11cf8:	00b12623          	sw	a1,12(sp)
   11cfc:	00a12423          	sw	a0,8(sp)
   11d00:	00112e23          	sw	ra,28(sp)
   11d04:	8a5fe0ef          	jal	105a8 <__sinit>
   11d08:	00c12583          	lw	a1,12(sp)
   11d0c:	00812503          	lw	a0,8(sp)
   11d10:	00c59783          	lh	a5,12(a1)
   11d14:	02079663          	bnez	a5,11d40 <fflush+0x78>
   11d18:	01c12083          	lw	ra,28(sp)
   11d1c:	00000513          	li	a0,0
   11d20:	02010113          	addi	sp,sp,32
   11d24:	00008067          	ret
   11d28:	90018613          	addi	a2,gp,-1792 # 12180 <__sglue>
   11d2c:	00000597          	auipc	a1,0x0
   11d30:	f4058593          	addi	a1,a1,-192 # 11c6c <_fflush_r>
   11d34:	00000517          	auipc	a0,0x0
   11d38:	45c50513          	addi	a0,a0,1116 # 12190 <_impure_data>
   11d3c:	8c1fe06f          	j	105fc <_fwalk_sglue>
   11d40:	01c12083          	lw	ra,28(sp)
   11d44:	02010113          	addi	sp,sp,32
   11d48:	cd9ff06f          	j	11a20 <__sflush_r>

00011d4c <_sbrk_r>:
   11d4c:	ff010113          	addi	sp,sp,-16
   11d50:	00812423          	sw	s0,8(sp)
   11d54:	00050413          	mv	s0,a0
   11d58:	00058513          	mv	a0,a1
   11d5c:	e401a623          	sw	zero,-436(gp) # 126cc <errno>
   11d60:	00112623          	sw	ra,12(sp)
   11d64:	15c000ef          	jal	11ec0 <_sbrk>
   11d68:	fff00793          	li	a5,-1
   11d6c:	00f50a63          	beq	a0,a5,11d80 <_sbrk_r+0x34>
   11d70:	00c12083          	lw	ra,12(sp)
   11d74:	00812403          	lw	s0,8(sp)
   11d78:	01010113          	addi	sp,sp,16
   11d7c:	00008067          	ret
   11d80:	e4c1a783          	lw	a5,-436(gp) # 126cc <errno>
   11d84:	fe0786e3          	beqz	a5,11d70 <_sbrk_r+0x24>
   11d88:	00c12083          	lw	ra,12(sp)
   11d8c:	00f42023          	sw	a5,0(s0)
   11d90:	00812403          	lw	s0,8(sp)
   11d94:	01010113          	addi	sp,sp,16
   11d98:	00008067          	ret

00011d9c <__libc_fini_array>:
   11d9c:	ff010113          	addi	sp,sp,-16
   11da0:	00812423          	sw	s0,8(sp)
   11da4:	00000797          	auipc	a5,0x0
   11da8:	2c078793          	addi	a5,a5,704 # 12064 <__do_global_dtors_aux_fini_array_entry>
   11dac:	00000417          	auipc	s0,0x0
   11db0:	2bc40413          	addi	s0,s0,700 # 12068 <__fini_array_end>
   11db4:	40f40433          	sub	s0,s0,a5
   11db8:	00912223          	sw	s1,4(sp)
   11dbc:	00112623          	sw	ra,12(sp)
   11dc0:	40245493          	srai	s1,s0,0x2
   11dc4:	02048063          	beqz	s1,11de4 <__libc_fini_array+0x48>
   11dc8:	ffc40413          	addi	s0,s0,-4
   11dcc:	00f40433          	add	s0,s0,a5
   11dd0:	00042783          	lw	a5,0(s0)
   11dd4:	fff48493          	addi	s1,s1,-1
   11dd8:	ffc40413          	addi	s0,s0,-4
   11ddc:	000780e7          	jalr	a5
   11de0:	fe0498e3          	bnez	s1,11dd0 <__libc_fini_array+0x34>
   11de4:	00c12083          	lw	ra,12(sp)
   11de8:	00812403          	lw	s0,8(sp)
   11dec:	00412483          	lw	s1,4(sp)
   11df0:	01010113          	addi	sp,sp,16
   11df4:	00008067          	ret

00011df8 <__register_exitproc>:
   11df8:	e5018713          	addi	a4,gp,-432 # 126d0 <__atexit>
   11dfc:	00072783          	lw	a5,0(a4)
   11e00:	04078c63          	beqz	a5,11e58 <__register_exitproc+0x60>
   11e04:	0047a703          	lw	a4,4(a5)
   11e08:	01f00813          	li	a6,31
   11e0c:	06e84e63          	blt	a6,a4,11e88 <__register_exitproc+0x90>
   11e10:	00271813          	slli	a6,a4,0x2
   11e14:	02050663          	beqz	a0,11e40 <__register_exitproc+0x48>
   11e18:	01078333          	add	t1,a5,a6
   11e1c:	08c32423          	sw	a2,136(t1)
   11e20:	1887a883          	lw	a7,392(a5)
   11e24:	00100613          	li	a2,1
   11e28:	00e61633          	sll	a2,a2,a4
   11e2c:	00c8e8b3          	or	a7,a7,a2
   11e30:	1917a423          	sw	a7,392(a5)
   11e34:	10d32423          	sw	a3,264(t1)
   11e38:	00200693          	li	a3,2
   11e3c:	02d50463          	beq	a0,a3,11e64 <__register_exitproc+0x6c>
   11e40:	00170713          	addi	a4,a4,1
   11e44:	00e7a223          	sw	a4,4(a5)
   11e48:	010787b3          	add	a5,a5,a6
   11e4c:	00b7a423          	sw	a1,8(a5)
   11e50:	00000513          	li	a0,0
   11e54:	00008067          	ret
   11e58:	0a018793          	addi	a5,gp,160 # 12920 <__atexit0>
   11e5c:	00f72023          	sw	a5,0(a4)
   11e60:	fa5ff06f          	j	11e04 <__register_exitproc+0xc>
   11e64:	18c7a683          	lw	a3,396(a5)
   11e68:	00170713          	addi	a4,a4,1
   11e6c:	00e7a223          	sw	a4,4(a5)
   11e70:	00c6e6b3          	or	a3,a3,a2
   11e74:	18d7a623          	sw	a3,396(a5)
   11e78:	010787b3          	add	a5,a5,a6
   11e7c:	00b7a423          	sw	a1,8(a5)
   11e80:	00000513          	li	a0,0
   11e84:	00008067          	ret
   11e88:	fff00513          	li	a0,-1
   11e8c:	00008067          	ret

00011e90 <_close>:
   11e90:	05800793          	li	a5,88
   11e94:	e4f1a623          	sw	a5,-436(gp) # 126cc <errno>
   11e98:	fff00513          	li	a0,-1
   11e9c:	00008067          	ret

00011ea0 <_lseek>:
   11ea0:	05800793          	li	a5,88
   11ea4:	e4f1a623          	sw	a5,-436(gp) # 126cc <errno>
   11ea8:	fff00513          	li	a0,-1
   11eac:	00008067          	ret

00011eb0 <_read>:
   11eb0:	05800793          	li	a5,88
   11eb4:	e4f1a623          	sw	a5,-436(gp) # 126cc <errno>
   11eb8:	fff00513          	li	a0,-1
   11ebc:	00008067          	ret

00011ec0 <_sbrk>:
   11ec0:	e6018713          	addi	a4,gp,-416 # 126e0 <heap_end.0>
   11ec4:	00072783          	lw	a5,0(a4)
   11ec8:	00078a63          	beqz	a5,11edc <_sbrk+0x1c>
   11ecc:	00a78533          	add	a0,a5,a0
   11ed0:	00a72023          	sw	a0,0(a4)
   11ed4:	00078513          	mv	a0,a5
   11ed8:	00008067          	ret
   11edc:	23018793          	addi	a5,gp,560 # 12ab0 <__BSS_END__>
   11ee0:	00a78533          	add	a0,a5,a0
   11ee4:	00a72023          	sw	a0,0(a4)
   11ee8:	00078513          	mv	a0,a5
   11eec:	00008067          	ret

00011ef0 <_write>:
   11ef0:	05800793          	li	a5,88
   11ef4:	e4f1a623          	sw	a5,-436(gp) # 126cc <errno>
   11ef8:	fff00513          	li	a0,-1
   11efc:	00008067          	ret

00011f00 <_exit>:
   11f00:	0000006f          	j	11f00 <_exit>

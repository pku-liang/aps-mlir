
/home/cloud/aps-mlir/examples/diff_match/avg_r/avg_r.out:     file format elf32-littleriscv


Disassembly of section .text:

000100b4 <exit>:
   100b4:	ff010113          	addi	sp,sp,-16
   100b8:	00000593          	li	a1,0
   100bc:	00812423          	sw	s0,8(sp)
   100c0:	00112623          	sw	ra,12(sp)
   100c4:	00050413          	mv	s0,a0
   100c8:	3f1000ef          	jal	10cb8 <__call_exitprocs>
   100cc:	e181a783          	lw	a5,-488(gp) # 13718 <__stdio_exit_handler>
   100d0:	00078463          	beqz	a5,100d8 <exit+0x24>
   100d4:	000780e7          	jalr	a5
   100d8:	00040513          	mv	a0,s0
   100dc:	729010ef          	jal	12004 <_exit>

000100e0 <register_fini>:
   100e0:	00000793          	li	a5,0
   100e4:	00078863          	beqz	a5,100f4 <register_fini+0x14>
   100e8:	00002517          	auipc	a0,0x2
   100ec:	db850513          	addi	a0,a0,-584 # 11ea0 <__libc_fini_array>
   100f0:	5010006f          	j	10df0 <atexit>
   100f4:	00008067          	ret

000100f8 <_start>:
   100f8:	00004197          	auipc	gp,0x4
   100fc:	80818193          	addi	gp,gp,-2040 # 13900 <__global_pointer$>
   10100:	e1818513          	addi	a0,gp,-488 # 13718 <__stdio_exit_handler>
   10104:	14018613          	addi	a2,gp,320 # 13a40 <__BSS_END__>
   10108:	40a60633          	sub	a2,a2,a0
   1010c:	00000593          	li	a1,0
   10110:	2cd000ef          	jal	10bdc <memset>
   10114:	00001517          	auipc	a0,0x1
   10118:	cdc50513          	addi	a0,a0,-804 # 10df0 <atexit>
   1011c:	00050863          	beqz	a0,1012c <_start+0x34>
   10120:	00002517          	auipc	a0,0x2
   10124:	d8050513          	addi	a0,a0,-640 # 11ea0 <__libc_fini_array>
   10128:	4c9000ef          	jal	10df0 <atexit>
   1012c:	21d000ef          	jal	10b48 <__libc_init_array>
   10130:	00012503          	lw	a0,0(sp)
   10134:	00410593          	addi	a1,sp,4
   10138:	00000613          	li	a2,0
   1013c:	0a8000ef          	jal	101e4 <main>
   10140:	f75ff06f          	j	100b4 <exit>

00010144 <__do_global_dtors_aux>:
   10144:	ff010113          	addi	sp,sp,-16
   10148:	00812423          	sw	s0,8(sp)
   1014c:	e3418413          	addi	s0,gp,-460 # 13734 <completed.1>
   10150:	00044783          	lbu	a5,0(s0)
   10154:	00112623          	sw	ra,12(sp)
   10158:	02079263          	bnez	a5,1017c <__do_global_dtors_aux+0x38>
   1015c:	00000793          	li	a5,0
   10160:	00078a63          	beqz	a5,10174 <__do_global_dtors_aux+0x30>
   10164:	00003517          	auipc	a0,0x3
   10168:	ed850513          	addi	a0,a0,-296 # 1303c <__EH_FRAME_BEGIN__>
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
   10194:	e3818593          	addi	a1,gp,-456 # 13738 <object.0>
   10198:	00003517          	auipc	a0,0x3
   1019c:	ea450513          	addi	a0,a0,-348 # 1303c <__EH_FRAME_BEGIN__>
   101a0:	00000317          	auipc	t1,0x0
   101a4:	00000067          	jr	zero # 0 <exit-0x100b4>
   101a8:	00008067          	ret

000101ac <avg_r>:
   101ac:	52b5752b          	.insn	4, 0x52b5752b
   101b0:	00008067          	ret

000101b4 <get_march>:
   101b4:	fff50513          	addi	a0,a0,-1
   101b8:	00400593          	li	a1,4
   101bc:	00a5ee63          	bltu	a1,a0,101d8 <get_march+0x24>
   101c0:	00251513          	slli	a0,a0,0x2
   101c4:	00002597          	auipc	a1,0x2
   101c8:	e6458593          	addi	a1,a1,-412 # 12028 <_exit+0x24>
   101cc:	00a58533          	add	a0,a1,a0
   101d0:	00052503          	lw	a0,0(a0)
   101d4:	00008067          	ret
   101d8:	00002517          	auipc	a0,0x2
   101dc:	e4750513          	addi	a0,a0,-441 # 1201f <_exit+0x1b>
   101e0:	00008067          	ret

000101e4 <main>:
   101e4:	ff010113          	addi	sp,sp,-16
   101e8:	00112623          	sw	ra,12(sp)
   101ec:	00812423          	sw	s0,8(sp)
   101f0:	00912223          	sw	s1,4(sp)
   101f4:	01212023          	sw	s2,0(sp)
   101f8:	0ff0000f          	fence
   101fc:	00003417          	auipc	s0,0x3
   10200:	f0440413          	addi	s0,s0,-252 # 13100 <expected_results>
   10204:	08042503          	lw	a0,128(s0)
   10208:	08442583          	lw	a1,132(s0)
   1020c:	fa1ff0ef          	jal	101ac <avg_r>
   10210:	0ff0000f          	fence
   10214:	00042683          	lw	a3,0(s0)
   10218:	0ff0000f          	fence
   1021c:	08842603          	lw	a2,136(s0)
   10220:	08c42583          	lw	a1,140(s0)
   10224:	00d54533          	xor	a0,a0,a3
   10228:	00153493          	seqz	s1,a0
   1022c:	00060513          	mv	a0,a2
   10230:	f7dff0ef          	jal	101ac <avg_r>
   10234:	0ff0000f          	fence
   10238:	00442683          	lw	a3,4(s0)
   1023c:	0ff0000f          	fence
   10240:	09042603          	lw	a2,144(s0)
   10244:	09442583          	lw	a1,148(s0)
   10248:	00d54533          	xor	a0,a0,a3
   1024c:	00153513          	seqz	a0,a0
   10250:	00a484b3          	add	s1,s1,a0
   10254:	00060513          	mv	a0,a2
   10258:	f55ff0ef          	jal	101ac <avg_r>
   1025c:	0ff0000f          	fence
   10260:	00842683          	lw	a3,8(s0)
   10264:	0ff0000f          	fence
   10268:	09842603          	lw	a2,152(s0)
   1026c:	09c42583          	lw	a1,156(s0)
   10270:	00d54533          	xor	a0,a0,a3
   10274:	00153913          	seqz	s2,a0
   10278:	00060513          	mv	a0,a2
   1027c:	f31ff0ef          	jal	101ac <avg_r>
   10280:	0ff0000f          	fence
   10284:	00c42683          	lw	a3,12(s0)
   10288:	0ff0000f          	fence
   1028c:	0a042603          	lw	a2,160(s0)
   10290:	0a442583          	lw	a1,164(s0)
   10294:	00d54533          	xor	a0,a0,a3
   10298:	00153513          	seqz	a0,a0
   1029c:	00a90533          	add	a0,s2,a0
   102a0:	00a484b3          	add	s1,s1,a0
   102a4:	00060513          	mv	a0,a2
   102a8:	f05ff0ef          	jal	101ac <avg_r>
   102ac:	0ff0000f          	fence
   102b0:	01042683          	lw	a3,16(s0)
   102b4:	0ff0000f          	fence
   102b8:	0a842603          	lw	a2,168(s0)
   102bc:	0ac42583          	lw	a1,172(s0)
   102c0:	00d54533          	xor	a0,a0,a3
   102c4:	00153913          	seqz	s2,a0
   102c8:	00060513          	mv	a0,a2
   102cc:	ee1ff0ef          	jal	101ac <avg_r>
   102d0:	0ff0000f          	fence
   102d4:	01442683          	lw	a3,20(s0)
   102d8:	0ff0000f          	fence
   102dc:	0b042603          	lw	a2,176(s0)
   102e0:	0b442583          	lw	a1,180(s0)
   102e4:	00d54533          	xor	a0,a0,a3
   102e8:	00153513          	seqz	a0,a0
   102ec:	00a90933          	add	s2,s2,a0
   102f0:	00060513          	mv	a0,a2
   102f4:	eb9ff0ef          	jal	101ac <avg_r>
   102f8:	0ff0000f          	fence
   102fc:	01842683          	lw	a3,24(s0)
   10300:	0ff0000f          	fence
   10304:	0b842603          	lw	a2,184(s0)
   10308:	0bc42583          	lw	a1,188(s0)
   1030c:	00d54533          	xor	a0,a0,a3
   10310:	00153513          	seqz	a0,a0
   10314:	00a90533          	add	a0,s2,a0
   10318:	00a484b3          	add	s1,s1,a0
   1031c:	00060513          	mv	a0,a2
   10320:	e8dff0ef          	jal	101ac <avg_r>
   10324:	0ff0000f          	fence
   10328:	01c42683          	lw	a3,28(s0)
   1032c:	0ff0000f          	fence
   10330:	0c042603          	lw	a2,192(s0)
   10334:	0c442583          	lw	a1,196(s0)
   10338:	00d54533          	xor	a0,a0,a3
   1033c:	00153913          	seqz	s2,a0
   10340:	00060513          	mv	a0,a2
   10344:	e69ff0ef          	jal	101ac <avg_r>
   10348:	0ff0000f          	fence
   1034c:	02042683          	lw	a3,32(s0)
   10350:	0ff0000f          	fence
   10354:	0c842603          	lw	a2,200(s0)
   10358:	0cc42583          	lw	a1,204(s0)
   1035c:	00d54533          	xor	a0,a0,a3
   10360:	00153513          	seqz	a0,a0
   10364:	00a90933          	add	s2,s2,a0
   10368:	00060513          	mv	a0,a2
   1036c:	e41ff0ef          	jal	101ac <avg_r>
   10370:	0ff0000f          	fence
   10374:	02442583          	lw	a1,36(s0)
   10378:	00b54533          	xor	a0,a0,a1
   1037c:	00153513          	seqz	a0,a0
   10380:	00a90533          	add	a0,s2,a0
   10384:	00a48533          	add	a0,s1,a0
   10388:	ff650513          	addi	a0,a0,-10
   1038c:	00a03533          	snez	a0,a0
   10390:	00c12083          	lw	ra,12(sp)
   10394:	00812403          	lw	s0,8(sp)
   10398:	00412483          	lw	s1,4(sp)
   1039c:	00012903          	lw	s2,0(sp)
   103a0:	01010113          	addi	sp,sp,16
   103a4:	00008067          	ret

000103a8 <__fp_lock>:
   103a8:	00000513          	li	a0,0
   103ac:	00008067          	ret

000103b0 <stdio_exit_handler>:
   103b0:	8d018613          	addi	a2,gp,-1840 # 131d0 <__sglue>
   103b4:	00001597          	auipc	a1,0x1
   103b8:	65c58593          	addi	a1,a1,1628 # 11a10 <_fclose_r>
   103bc:	00003517          	auipc	a0,0x3
   103c0:	e2450513          	addi	a0,a0,-476 # 131e0 <_impure_data>
   103c4:	33c0006f          	j	10700 <_fwalk_sglue>

000103c8 <cleanup_stdio>:
   103c8:	00452583          	lw	a1,4(a0)
   103cc:	ff010113          	addi	sp,sp,-16
   103d0:	00812423          	sw	s0,8(sp)
   103d4:	00112623          	sw	ra,12(sp)
   103d8:	e5018793          	addi	a5,gp,-432 # 13750 <__sf>
   103dc:	00050413          	mv	s0,a0
   103e0:	00f58463          	beq	a1,a5,103e8 <cleanup_stdio+0x20>
   103e4:	62c010ef          	jal	11a10 <_fclose_r>
   103e8:	00842583          	lw	a1,8(s0)
   103ec:	eb818793          	addi	a5,gp,-328 # 137b8 <__sf+0x68>
   103f0:	00f58663          	beq	a1,a5,103fc <cleanup_stdio+0x34>
   103f4:	00040513          	mv	a0,s0
   103f8:	618010ef          	jal	11a10 <_fclose_r>
   103fc:	00c42583          	lw	a1,12(s0)
   10400:	f2018793          	addi	a5,gp,-224 # 13820 <__sf+0xd0>
   10404:	00f58c63          	beq	a1,a5,1041c <cleanup_stdio+0x54>
   10408:	00040513          	mv	a0,s0
   1040c:	00812403          	lw	s0,8(sp)
   10410:	00c12083          	lw	ra,12(sp)
   10414:	01010113          	addi	sp,sp,16
   10418:	5f80106f          	j	11a10 <_fclose_r>
   1041c:	00c12083          	lw	ra,12(sp)
   10420:	00812403          	lw	s0,8(sp)
   10424:	01010113          	addi	sp,sp,16
   10428:	00008067          	ret

0001042c <__fp_unlock>:
   1042c:	00000513          	li	a0,0
   10430:	00008067          	ret

00010434 <global_stdio_init.part.0>:
   10434:	fe010113          	addi	sp,sp,-32
   10438:	00000797          	auipc	a5,0x0
   1043c:	f7878793          	addi	a5,a5,-136 # 103b0 <stdio_exit_handler>
   10440:	00112e23          	sw	ra,28(sp)
   10444:	00812c23          	sw	s0,24(sp)
   10448:	00912a23          	sw	s1,20(sp)
   1044c:	e5018413          	addi	s0,gp,-432 # 13750 <__sf>
   10450:	01212823          	sw	s2,16(sp)
   10454:	01312623          	sw	s3,12(sp)
   10458:	01412423          	sw	s4,8(sp)
   1045c:	e0f1ac23          	sw	a5,-488(gp) # 13718 <__stdio_exit_handler>
   10460:	00800613          	li	a2,8
   10464:	00400793          	li	a5,4
   10468:	00000593          	li	a1,0
   1046c:	eac18513          	addi	a0,gp,-340 # 137ac <__sf+0x5c>
   10470:	00f42623          	sw	a5,12(s0)
   10474:	00042023          	sw	zero,0(s0)
   10478:	00042223          	sw	zero,4(s0)
   1047c:	00042423          	sw	zero,8(s0)
   10480:	06042223          	sw	zero,100(s0)
   10484:	00042823          	sw	zero,16(s0)
   10488:	00042a23          	sw	zero,20(s0)
   1048c:	00042c23          	sw	zero,24(s0)
   10490:	74c000ef          	jal	10bdc <memset>
   10494:	000107b7          	lui	a5,0x10
   10498:	00000a17          	auipc	s4,0x0
   1049c:	31ca0a13          	addi	s4,s4,796 # 107b4 <__sread>
   104a0:	00000997          	auipc	s3,0x0
   104a4:	37898993          	addi	s3,s3,888 # 10818 <__swrite>
   104a8:	00000917          	auipc	s2,0x0
   104ac:	3f890913          	addi	s2,s2,1016 # 108a0 <__sseek>
   104b0:	00000497          	auipc	s1,0x0
   104b4:	46848493          	addi	s1,s1,1128 # 10918 <__sclose>
   104b8:	00978793          	addi	a5,a5,9 # 10009 <exit-0xab>
   104bc:	00800613          	li	a2,8
   104c0:	00000593          	li	a1,0
   104c4:	f1418513          	addi	a0,gp,-236 # 13814 <__sf+0xc4>
   104c8:	03442023          	sw	s4,32(s0)
   104cc:	03342223          	sw	s3,36(s0)
   104d0:	03242423          	sw	s2,40(s0)
   104d4:	02942623          	sw	s1,44(s0)
   104d8:	06f42a23          	sw	a5,116(s0)
   104dc:	00842e23          	sw	s0,28(s0)
   104e0:	06042423          	sw	zero,104(s0)
   104e4:	06042623          	sw	zero,108(s0)
   104e8:	06042823          	sw	zero,112(s0)
   104ec:	0c042623          	sw	zero,204(s0)
   104f0:	06042c23          	sw	zero,120(s0)
   104f4:	06042e23          	sw	zero,124(s0)
   104f8:	08042023          	sw	zero,128(s0)
   104fc:	6e0000ef          	jal	10bdc <memset>
   10500:	000207b7          	lui	a5,0x20
   10504:	01278793          	addi	a5,a5,18 # 20012 <__BSS_END__+0xc5d2>
   10508:	eb818713          	addi	a4,gp,-328 # 137b8 <__sf+0x68>
   1050c:	00800613          	li	a2,8
   10510:	00000593          	li	a1,0
   10514:	f7c18513          	addi	a0,gp,-132 # 1387c <__sf+0x12c>
   10518:	09442423          	sw	s4,136(s0)
   1051c:	09342623          	sw	s3,140(s0)
   10520:	09242823          	sw	s2,144(s0)
   10524:	08942a23          	sw	s1,148(s0)
   10528:	0cf42e23          	sw	a5,220(s0)
   1052c:	08e42223          	sw	a4,132(s0)
   10530:	0c042823          	sw	zero,208(s0)
   10534:	0c042a23          	sw	zero,212(s0)
   10538:	0c042c23          	sw	zero,216(s0)
   1053c:	12042a23          	sw	zero,308(s0)
   10540:	0e042023          	sw	zero,224(s0)
   10544:	0e042223          	sw	zero,228(s0)
   10548:	0e042423          	sw	zero,232(s0)
   1054c:	690000ef          	jal	10bdc <memset>
   10550:	f2018793          	addi	a5,gp,-224 # 13820 <__sf+0xd0>
   10554:	0f442823          	sw	s4,240(s0)
   10558:	0f342a23          	sw	s3,244(s0)
   1055c:	0f242c23          	sw	s2,248(s0)
   10560:	0e942e23          	sw	s1,252(s0)
   10564:	01c12083          	lw	ra,28(sp)
   10568:	0ef42623          	sw	a5,236(s0)
   1056c:	01812403          	lw	s0,24(sp)
   10570:	01412483          	lw	s1,20(sp)
   10574:	01012903          	lw	s2,16(sp)
   10578:	00c12983          	lw	s3,12(sp)
   1057c:	00812a03          	lw	s4,8(sp)
   10580:	02010113          	addi	sp,sp,32
   10584:	00008067          	ret

00010588 <__sfp>:
   10588:	fe010113          	addi	sp,sp,-32
   1058c:	01312623          	sw	s3,12(sp)
   10590:	00112e23          	sw	ra,28(sp)
   10594:	00812c23          	sw	s0,24(sp)
   10598:	00912a23          	sw	s1,20(sp)
   1059c:	01212823          	sw	s2,16(sp)
   105a0:	e181a783          	lw	a5,-488(gp) # 13718 <__stdio_exit_handler>
   105a4:	00050993          	mv	s3,a0
   105a8:	0e078663          	beqz	a5,10694 <__sfp+0x10c>
   105ac:	8d018913          	addi	s2,gp,-1840 # 131d0 <__sglue>
   105b0:	fff00493          	li	s1,-1
   105b4:	00492783          	lw	a5,4(s2)
   105b8:	00892403          	lw	s0,8(s2)
   105bc:	fff78793          	addi	a5,a5,-1
   105c0:	0007d863          	bgez	a5,105d0 <__sfp+0x48>
   105c4:	0800006f          	j	10644 <__sfp+0xbc>
   105c8:	06840413          	addi	s0,s0,104
   105cc:	06978c63          	beq	a5,s1,10644 <__sfp+0xbc>
   105d0:	00c41703          	lh	a4,12(s0)
   105d4:	fff78793          	addi	a5,a5,-1
   105d8:	fe0718e3          	bnez	a4,105c8 <__sfp+0x40>
   105dc:	ffff07b7          	lui	a5,0xffff0
   105e0:	00178793          	addi	a5,a5,1 # ffff0001 <__BSS_END__+0xfffdc5c1>
   105e4:	00f42623          	sw	a5,12(s0)
   105e8:	06042223          	sw	zero,100(s0)
   105ec:	00042023          	sw	zero,0(s0)
   105f0:	00042423          	sw	zero,8(s0)
   105f4:	00042223          	sw	zero,4(s0)
   105f8:	00042823          	sw	zero,16(s0)
   105fc:	00042a23          	sw	zero,20(s0)
   10600:	00042c23          	sw	zero,24(s0)
   10604:	00800613          	li	a2,8
   10608:	00000593          	li	a1,0
   1060c:	05c40513          	addi	a0,s0,92
   10610:	5cc000ef          	jal	10bdc <memset>
   10614:	02042823          	sw	zero,48(s0)
   10618:	02042a23          	sw	zero,52(s0)
   1061c:	04042223          	sw	zero,68(s0)
   10620:	04042423          	sw	zero,72(s0)
   10624:	01c12083          	lw	ra,28(sp)
   10628:	00040513          	mv	a0,s0
   1062c:	01812403          	lw	s0,24(sp)
   10630:	01412483          	lw	s1,20(sp)
   10634:	01012903          	lw	s2,16(sp)
   10638:	00c12983          	lw	s3,12(sp)
   1063c:	02010113          	addi	sp,sp,32
   10640:	00008067          	ret
   10644:	00092403          	lw	s0,0(s2)
   10648:	00040663          	beqz	s0,10654 <__sfp+0xcc>
   1064c:	00040913          	mv	s2,s0
   10650:	f65ff06f          	j	105b4 <__sfp+0x2c>
   10654:	1ac00593          	li	a1,428
   10658:	00098513          	mv	a0,s3
   1065c:	3e5000ef          	jal	11240 <_malloc_r>
   10660:	00050413          	mv	s0,a0
   10664:	02050c63          	beqz	a0,1069c <__sfp+0x114>
   10668:	00c50513          	addi	a0,a0,12
   1066c:	00400793          	li	a5,4
   10670:	00042023          	sw	zero,0(s0)
   10674:	00f42223          	sw	a5,4(s0)
   10678:	00a42423          	sw	a0,8(s0)
   1067c:	1a000613          	li	a2,416
   10680:	00000593          	li	a1,0
   10684:	558000ef          	jal	10bdc <memset>
   10688:	00892023          	sw	s0,0(s2)
   1068c:	00040913          	mv	s2,s0
   10690:	f25ff06f          	j	105b4 <__sfp+0x2c>
   10694:	da1ff0ef          	jal	10434 <global_stdio_init.part.0>
   10698:	f15ff06f          	j	105ac <__sfp+0x24>
   1069c:	00092023          	sw	zero,0(s2)
   106a0:	00c00793          	li	a5,12
   106a4:	00f9a023          	sw	a5,0(s3)
   106a8:	f7dff06f          	j	10624 <__sfp+0x9c>

000106ac <__sinit>:
   106ac:	03452783          	lw	a5,52(a0)
   106b0:	00078463          	beqz	a5,106b8 <__sinit+0xc>
   106b4:	00008067          	ret
   106b8:	00000797          	auipc	a5,0x0
   106bc:	d1078793          	addi	a5,a5,-752 # 103c8 <cleanup_stdio>
   106c0:	02f52a23          	sw	a5,52(a0)
   106c4:	e181a783          	lw	a5,-488(gp) # 13718 <__stdio_exit_handler>
   106c8:	fe0796e3          	bnez	a5,106b4 <__sinit+0x8>
   106cc:	d69ff06f          	j	10434 <global_stdio_init.part.0>

000106d0 <__sfp_lock_acquire>:
   106d0:	00008067          	ret

000106d4 <__sfp_lock_release>:
   106d4:	00008067          	ret

000106d8 <__fp_lock_all>:
   106d8:	8d018613          	addi	a2,gp,-1840 # 131d0 <__sglue>
   106dc:	00000597          	auipc	a1,0x0
   106e0:	ccc58593          	addi	a1,a1,-820 # 103a8 <__fp_lock>
   106e4:	00000513          	li	a0,0
   106e8:	0180006f          	j	10700 <_fwalk_sglue>

000106ec <__fp_unlock_all>:
   106ec:	8d018613          	addi	a2,gp,-1840 # 131d0 <__sglue>
   106f0:	00000597          	auipc	a1,0x0
   106f4:	d3c58593          	addi	a1,a1,-708 # 1042c <__fp_unlock>
   106f8:	00000513          	li	a0,0
   106fc:	0040006f          	j	10700 <_fwalk_sglue>

00010700 <_fwalk_sglue>:
   10700:	fd010113          	addi	sp,sp,-48
   10704:	03212023          	sw	s2,32(sp)
   10708:	01312e23          	sw	s3,28(sp)
   1070c:	01412c23          	sw	s4,24(sp)
   10710:	01512a23          	sw	s5,20(sp)
   10714:	01612823          	sw	s6,16(sp)
   10718:	01712623          	sw	s7,12(sp)
   1071c:	02112623          	sw	ra,44(sp)
   10720:	02812423          	sw	s0,40(sp)
   10724:	02912223          	sw	s1,36(sp)
   10728:	00050b13          	mv	s6,a0
   1072c:	00058b93          	mv	s7,a1
   10730:	00060a93          	mv	s5,a2
   10734:	00000a13          	li	s4,0
   10738:	00100993          	li	s3,1
   1073c:	fff00913          	li	s2,-1
   10740:	004aa483          	lw	s1,4(s5)
   10744:	008aa403          	lw	s0,8(s5)
   10748:	fff48493          	addi	s1,s1,-1
   1074c:	0204c863          	bltz	s1,1077c <_fwalk_sglue+0x7c>
   10750:	00c45783          	lhu	a5,12(s0)
   10754:	00f9fe63          	bgeu	s3,a5,10770 <_fwalk_sglue+0x70>
   10758:	00e41783          	lh	a5,14(s0)
   1075c:	00040593          	mv	a1,s0
   10760:	000b0513          	mv	a0,s6
   10764:	01278663          	beq	a5,s2,10770 <_fwalk_sglue+0x70>
   10768:	000b80e7          	jalr	s7
   1076c:	00aa6a33          	or	s4,s4,a0
   10770:	fff48493          	addi	s1,s1,-1
   10774:	06840413          	addi	s0,s0,104
   10778:	fd249ce3          	bne	s1,s2,10750 <_fwalk_sglue+0x50>
   1077c:	000aaa83          	lw	s5,0(s5)
   10780:	fc0a90e3          	bnez	s5,10740 <_fwalk_sglue+0x40>
   10784:	02c12083          	lw	ra,44(sp)
   10788:	02812403          	lw	s0,40(sp)
   1078c:	02412483          	lw	s1,36(sp)
   10790:	02012903          	lw	s2,32(sp)
   10794:	01c12983          	lw	s3,28(sp)
   10798:	01412a83          	lw	s5,20(sp)
   1079c:	01012b03          	lw	s6,16(sp)
   107a0:	00c12b83          	lw	s7,12(sp)
   107a4:	000a0513          	mv	a0,s4
   107a8:	01812a03          	lw	s4,24(sp)
   107ac:	03010113          	addi	sp,sp,48
   107b0:	00008067          	ret

000107b4 <__sread>:
   107b4:	ff010113          	addi	sp,sp,-16
   107b8:	00812423          	sw	s0,8(sp)
   107bc:	00058413          	mv	s0,a1
   107c0:	00e59583          	lh	a1,14(a1)
   107c4:	00112623          	sw	ra,12(sp)
   107c8:	2c8000ef          	jal	10a90 <_read_r>
   107cc:	02054063          	bltz	a0,107ec <__sread+0x38>
   107d0:	05042783          	lw	a5,80(s0)
   107d4:	00c12083          	lw	ra,12(sp)
   107d8:	00a787b3          	add	a5,a5,a0
   107dc:	04f42823          	sw	a5,80(s0)
   107e0:	00812403          	lw	s0,8(sp)
   107e4:	01010113          	addi	sp,sp,16
   107e8:	00008067          	ret
   107ec:	00c45783          	lhu	a5,12(s0)
   107f0:	fffff737          	lui	a4,0xfffff
   107f4:	fff70713          	addi	a4,a4,-1 # ffffefff <__BSS_END__+0xfffeb5bf>
   107f8:	00e7f7b3          	and	a5,a5,a4
   107fc:	00c12083          	lw	ra,12(sp)
   10800:	00f41623          	sh	a5,12(s0)
   10804:	00812403          	lw	s0,8(sp)
   10808:	01010113          	addi	sp,sp,16
   1080c:	00008067          	ret

00010810 <__seofread>:
   10810:	00000513          	li	a0,0
   10814:	00008067          	ret

00010818 <__swrite>:
   10818:	00c59783          	lh	a5,12(a1)
   1081c:	fe010113          	addi	sp,sp,-32
   10820:	00812c23          	sw	s0,24(sp)
   10824:	00912a23          	sw	s1,20(sp)
   10828:	01212823          	sw	s2,16(sp)
   1082c:	01312623          	sw	s3,12(sp)
   10830:	00112e23          	sw	ra,28(sp)
   10834:	1007f713          	andi	a4,a5,256
   10838:	00058413          	mv	s0,a1
   1083c:	00050493          	mv	s1,a0
   10840:	00060913          	mv	s2,a2
   10844:	00068993          	mv	s3,a3
   10848:	04071063          	bnez	a4,10888 <__swrite+0x70>
   1084c:	fffff737          	lui	a4,0xfffff
   10850:	fff70713          	addi	a4,a4,-1 # ffffefff <__BSS_END__+0xfffeb5bf>
   10854:	00e7f7b3          	and	a5,a5,a4
   10858:	00e41583          	lh	a1,14(s0)
   1085c:	00f41623          	sh	a5,12(s0)
   10860:	01812403          	lw	s0,24(sp)
   10864:	01c12083          	lw	ra,28(sp)
   10868:	00098693          	mv	a3,s3
   1086c:	00090613          	mv	a2,s2
   10870:	00c12983          	lw	s3,12(sp)
   10874:	01012903          	lw	s2,16(sp)
   10878:	00048513          	mv	a0,s1
   1087c:	01412483          	lw	s1,20(sp)
   10880:	02010113          	addi	sp,sp,32
   10884:	2680006f          	j	10aec <_write_r>
   10888:	00e59583          	lh	a1,14(a1)
   1088c:	00200693          	li	a3,2
   10890:	00000613          	li	a2,0
   10894:	1a0000ef          	jal	10a34 <_lseek_r>
   10898:	00c41783          	lh	a5,12(s0)
   1089c:	fb1ff06f          	j	1084c <__swrite+0x34>

000108a0 <__sseek>:
   108a0:	ff010113          	addi	sp,sp,-16
   108a4:	00812423          	sw	s0,8(sp)
   108a8:	00058413          	mv	s0,a1
   108ac:	00e59583          	lh	a1,14(a1)
   108b0:	00112623          	sw	ra,12(sp)
   108b4:	180000ef          	jal	10a34 <_lseek_r>
   108b8:	fff00793          	li	a5,-1
   108bc:	02f50863          	beq	a0,a5,108ec <__sseek+0x4c>
   108c0:	00c45783          	lhu	a5,12(s0)
   108c4:	00001737          	lui	a4,0x1
   108c8:	00c12083          	lw	ra,12(sp)
   108cc:	00e7e7b3          	or	a5,a5,a4
   108d0:	01079793          	slli	a5,a5,0x10
   108d4:	4107d793          	srai	a5,a5,0x10
   108d8:	04a42823          	sw	a0,80(s0)
   108dc:	00f41623          	sh	a5,12(s0)
   108e0:	00812403          	lw	s0,8(sp)
   108e4:	01010113          	addi	sp,sp,16
   108e8:	00008067          	ret
   108ec:	00c45783          	lhu	a5,12(s0)
   108f0:	fffff737          	lui	a4,0xfffff
   108f4:	fff70713          	addi	a4,a4,-1 # ffffefff <__BSS_END__+0xfffeb5bf>
   108f8:	00e7f7b3          	and	a5,a5,a4
   108fc:	01079793          	slli	a5,a5,0x10
   10900:	4107d793          	srai	a5,a5,0x10
   10904:	00c12083          	lw	ra,12(sp)
   10908:	00f41623          	sh	a5,12(s0)
   1090c:	00812403          	lw	s0,8(sp)
   10910:	01010113          	addi	sp,sp,16
   10914:	00008067          	ret

00010918 <__sclose>:
   10918:	00e59583          	lh	a1,14(a1)
   1091c:	0040006f          	j	10920 <_close_r>

00010920 <_close_r>:
   10920:	ff010113          	addi	sp,sp,-16
   10924:	00812423          	sw	s0,8(sp)
   10928:	00050413          	mv	s0,a0
   1092c:	00058513          	mv	a0,a1
   10930:	e001ae23          	sw	zero,-484(gp) # 1371c <errno>
   10934:	00112623          	sw	ra,12(sp)
   10938:	65c010ef          	jal	11f94 <_close>
   1093c:	fff00793          	li	a5,-1
   10940:	00f50a63          	beq	a0,a5,10954 <_close_r+0x34>
   10944:	00c12083          	lw	ra,12(sp)
   10948:	00812403          	lw	s0,8(sp)
   1094c:	01010113          	addi	sp,sp,16
   10950:	00008067          	ret
   10954:	e1c1a783          	lw	a5,-484(gp) # 1371c <errno>
   10958:	fe0786e3          	beqz	a5,10944 <_close_r+0x24>
   1095c:	00c12083          	lw	ra,12(sp)
   10960:	00f42023          	sw	a5,0(s0)
   10964:	00812403          	lw	s0,8(sp)
   10968:	01010113          	addi	sp,sp,16
   1096c:	00008067          	ret

00010970 <_reclaim_reent>:
   10970:	e0c1a783          	lw	a5,-500(gp) # 1370c <_impure_ptr>
   10974:	0aa78e63          	beq	a5,a0,10a30 <_reclaim_reent+0xc0>
   10978:	04452583          	lw	a1,68(a0)
   1097c:	fe010113          	addi	sp,sp,-32
   10980:	00912a23          	sw	s1,20(sp)
   10984:	00112e23          	sw	ra,28(sp)
   10988:	00050493          	mv	s1,a0
   1098c:	04058c63          	beqz	a1,109e4 <_reclaim_reent+0x74>
   10990:	01212823          	sw	s2,16(sp)
   10994:	01312623          	sw	s3,12(sp)
   10998:	00812c23          	sw	s0,24(sp)
   1099c:	00000913          	li	s2,0
   109a0:	08000993          	li	s3,128
   109a4:	012587b3          	add	a5,a1,s2
   109a8:	0007a403          	lw	s0,0(a5)
   109ac:	00040e63          	beqz	s0,109c8 <_reclaim_reent+0x58>
   109b0:	00040593          	mv	a1,s0
   109b4:	00042403          	lw	s0,0(s0)
   109b8:	00048513          	mv	a0,s1
   109bc:	580000ef          	jal	10f3c <_free_r>
   109c0:	fe0418e3          	bnez	s0,109b0 <_reclaim_reent+0x40>
   109c4:	0444a583          	lw	a1,68(s1)
   109c8:	00490913          	addi	s2,s2,4
   109cc:	fd391ce3          	bne	s2,s3,109a4 <_reclaim_reent+0x34>
   109d0:	00048513          	mv	a0,s1
   109d4:	568000ef          	jal	10f3c <_free_r>
   109d8:	01812403          	lw	s0,24(sp)
   109dc:	01012903          	lw	s2,16(sp)
   109e0:	00c12983          	lw	s3,12(sp)
   109e4:	0384a583          	lw	a1,56(s1)
   109e8:	00058663          	beqz	a1,109f4 <_reclaim_reent+0x84>
   109ec:	00048513          	mv	a0,s1
   109f0:	54c000ef          	jal	10f3c <_free_r>
   109f4:	04c4a583          	lw	a1,76(s1)
   109f8:	00058663          	beqz	a1,10a04 <_reclaim_reent+0x94>
   109fc:	00048513          	mv	a0,s1
   10a00:	53c000ef          	jal	10f3c <_free_r>
   10a04:	0344a783          	lw	a5,52(s1)
   10a08:	00078c63          	beqz	a5,10a20 <_reclaim_reent+0xb0>
   10a0c:	01c12083          	lw	ra,28(sp)
   10a10:	00048513          	mv	a0,s1
   10a14:	01412483          	lw	s1,20(sp)
   10a18:	02010113          	addi	sp,sp,32
   10a1c:	00078067          	jr	a5
   10a20:	01c12083          	lw	ra,28(sp)
   10a24:	01412483          	lw	s1,20(sp)
   10a28:	02010113          	addi	sp,sp,32
   10a2c:	00008067          	ret
   10a30:	00008067          	ret

00010a34 <_lseek_r>:
   10a34:	ff010113          	addi	sp,sp,-16
   10a38:	00058713          	mv	a4,a1
   10a3c:	00812423          	sw	s0,8(sp)
   10a40:	00060593          	mv	a1,a2
   10a44:	00050413          	mv	s0,a0
   10a48:	00068613          	mv	a2,a3
   10a4c:	00070513          	mv	a0,a4
   10a50:	e001ae23          	sw	zero,-484(gp) # 1371c <errno>
   10a54:	00112623          	sw	ra,12(sp)
   10a58:	54c010ef          	jal	11fa4 <_lseek>
   10a5c:	fff00793          	li	a5,-1
   10a60:	00f50a63          	beq	a0,a5,10a74 <_lseek_r+0x40>
   10a64:	00c12083          	lw	ra,12(sp)
   10a68:	00812403          	lw	s0,8(sp)
   10a6c:	01010113          	addi	sp,sp,16
   10a70:	00008067          	ret
   10a74:	e1c1a783          	lw	a5,-484(gp) # 1371c <errno>
   10a78:	fe0786e3          	beqz	a5,10a64 <_lseek_r+0x30>
   10a7c:	00c12083          	lw	ra,12(sp)
   10a80:	00f42023          	sw	a5,0(s0)
   10a84:	00812403          	lw	s0,8(sp)
   10a88:	01010113          	addi	sp,sp,16
   10a8c:	00008067          	ret

00010a90 <_read_r>:
   10a90:	ff010113          	addi	sp,sp,-16
   10a94:	00058713          	mv	a4,a1
   10a98:	00812423          	sw	s0,8(sp)
   10a9c:	00060593          	mv	a1,a2
   10aa0:	00050413          	mv	s0,a0
   10aa4:	00068613          	mv	a2,a3
   10aa8:	00070513          	mv	a0,a4
   10aac:	e001ae23          	sw	zero,-484(gp) # 1371c <errno>
   10ab0:	00112623          	sw	ra,12(sp)
   10ab4:	500010ef          	jal	11fb4 <_read>
   10ab8:	fff00793          	li	a5,-1
   10abc:	00f50a63          	beq	a0,a5,10ad0 <_read_r+0x40>
   10ac0:	00c12083          	lw	ra,12(sp)
   10ac4:	00812403          	lw	s0,8(sp)
   10ac8:	01010113          	addi	sp,sp,16
   10acc:	00008067          	ret
   10ad0:	e1c1a783          	lw	a5,-484(gp) # 1371c <errno>
   10ad4:	fe0786e3          	beqz	a5,10ac0 <_read_r+0x30>
   10ad8:	00c12083          	lw	ra,12(sp)
   10adc:	00f42023          	sw	a5,0(s0)
   10ae0:	00812403          	lw	s0,8(sp)
   10ae4:	01010113          	addi	sp,sp,16
   10ae8:	00008067          	ret

00010aec <_write_r>:
   10aec:	ff010113          	addi	sp,sp,-16
   10af0:	00058713          	mv	a4,a1
   10af4:	00812423          	sw	s0,8(sp)
   10af8:	00060593          	mv	a1,a2
   10afc:	00050413          	mv	s0,a0
   10b00:	00068613          	mv	a2,a3
   10b04:	00070513          	mv	a0,a4
   10b08:	e001ae23          	sw	zero,-484(gp) # 1371c <errno>
   10b0c:	00112623          	sw	ra,12(sp)
   10b10:	4e4010ef          	jal	11ff4 <_write>
   10b14:	fff00793          	li	a5,-1
   10b18:	00f50a63          	beq	a0,a5,10b2c <_write_r+0x40>
   10b1c:	00c12083          	lw	ra,12(sp)
   10b20:	00812403          	lw	s0,8(sp)
   10b24:	01010113          	addi	sp,sp,16
   10b28:	00008067          	ret
   10b2c:	e1c1a783          	lw	a5,-484(gp) # 1371c <errno>
   10b30:	fe0786e3          	beqz	a5,10b1c <_write_r+0x30>
   10b34:	00c12083          	lw	ra,12(sp)
   10b38:	00f42023          	sw	a5,0(s0)
   10b3c:	00812403          	lw	s0,8(sp)
   10b40:	01010113          	addi	sp,sp,16
   10b44:	00008067          	ret

00010b48 <__libc_init_array>:
   10b48:	ff010113          	addi	sp,sp,-16
   10b4c:	00812423          	sw	s0,8(sp)
   10b50:	01212023          	sw	s2,0(sp)
   10b54:	00002797          	auipc	a5,0x2
   10b58:	55078793          	addi	a5,a5,1360 # 130a4 <__init_array_start>
   10b5c:	00002417          	auipc	s0,0x2
   10b60:	54840413          	addi	s0,s0,1352 # 130a4 <__init_array_start>
   10b64:	00112623          	sw	ra,12(sp)
   10b68:	00912223          	sw	s1,4(sp)
   10b6c:	40878933          	sub	s2,a5,s0
   10b70:	02878063          	beq	a5,s0,10b90 <__libc_init_array+0x48>
   10b74:	40295913          	srai	s2,s2,0x2
   10b78:	00000493          	li	s1,0
   10b7c:	00042783          	lw	a5,0(s0)
   10b80:	00148493          	addi	s1,s1,1
   10b84:	00440413          	addi	s0,s0,4
   10b88:	000780e7          	jalr	a5
   10b8c:	ff24e8e3          	bltu	s1,s2,10b7c <__libc_init_array+0x34>
   10b90:	00002797          	auipc	a5,0x2
   10b94:	51c78793          	addi	a5,a5,1308 # 130ac <__do_global_dtors_aux_fini_array_entry>
   10b98:	00002417          	auipc	s0,0x2
   10b9c:	50c40413          	addi	s0,s0,1292 # 130a4 <__init_array_start>
   10ba0:	40878933          	sub	s2,a5,s0
   10ba4:	40295913          	srai	s2,s2,0x2
   10ba8:	00878e63          	beq	a5,s0,10bc4 <__libc_init_array+0x7c>
   10bac:	00000493          	li	s1,0
   10bb0:	00042783          	lw	a5,0(s0)
   10bb4:	00148493          	addi	s1,s1,1
   10bb8:	00440413          	addi	s0,s0,4
   10bbc:	000780e7          	jalr	a5
   10bc0:	ff24e8e3          	bltu	s1,s2,10bb0 <__libc_init_array+0x68>
   10bc4:	00c12083          	lw	ra,12(sp)
   10bc8:	00812403          	lw	s0,8(sp)
   10bcc:	00412483          	lw	s1,4(sp)
   10bd0:	00012903          	lw	s2,0(sp)
   10bd4:	01010113          	addi	sp,sp,16
   10bd8:	00008067          	ret

00010bdc <memset>:
   10bdc:	00f00313          	li	t1,15
   10be0:	00050713          	mv	a4,a0
   10be4:	02c37e63          	bgeu	t1,a2,10c20 <memset+0x44>
   10be8:	00f77793          	andi	a5,a4,15
   10bec:	0a079063          	bnez	a5,10c8c <memset+0xb0>
   10bf0:	08059263          	bnez	a1,10c74 <memset+0x98>
   10bf4:	ff067693          	andi	a3,a2,-16
   10bf8:	00f67613          	andi	a2,a2,15
   10bfc:	00e686b3          	add	a3,a3,a4
   10c00:	00b72023          	sw	a1,0(a4)
   10c04:	00b72223          	sw	a1,4(a4)
   10c08:	00b72423          	sw	a1,8(a4)
   10c0c:	00b72623          	sw	a1,12(a4)
   10c10:	01070713          	addi	a4,a4,16
   10c14:	fed766e3          	bltu	a4,a3,10c00 <memset+0x24>
   10c18:	00061463          	bnez	a2,10c20 <memset+0x44>
   10c1c:	00008067          	ret
   10c20:	40c306b3          	sub	a3,t1,a2
   10c24:	00269693          	slli	a3,a3,0x2
   10c28:	00000297          	auipc	t0,0x0
   10c2c:	005686b3          	add	a3,a3,t0
   10c30:	00c68067          	jr	12(a3)
   10c34:	00b70723          	sb	a1,14(a4)
   10c38:	00b706a3          	sb	a1,13(a4)
   10c3c:	00b70623          	sb	a1,12(a4)
   10c40:	00b705a3          	sb	a1,11(a4)
   10c44:	00b70523          	sb	a1,10(a4)
   10c48:	00b704a3          	sb	a1,9(a4)
   10c4c:	00b70423          	sb	a1,8(a4)
   10c50:	00b703a3          	sb	a1,7(a4)
   10c54:	00b70323          	sb	a1,6(a4)
   10c58:	00b702a3          	sb	a1,5(a4)
   10c5c:	00b70223          	sb	a1,4(a4)
   10c60:	00b701a3          	sb	a1,3(a4)
   10c64:	00b70123          	sb	a1,2(a4)
   10c68:	00b700a3          	sb	a1,1(a4)
   10c6c:	00b70023          	sb	a1,0(a4)
   10c70:	00008067          	ret
   10c74:	0ff5f593          	zext.b	a1,a1
   10c78:	00859693          	slli	a3,a1,0x8
   10c7c:	00d5e5b3          	or	a1,a1,a3
   10c80:	01059693          	slli	a3,a1,0x10
   10c84:	00d5e5b3          	or	a1,a1,a3
   10c88:	f6dff06f          	j	10bf4 <memset+0x18>
   10c8c:	00279693          	slli	a3,a5,0x2
   10c90:	00000297          	auipc	t0,0x0
   10c94:	005686b3          	add	a3,a3,t0
   10c98:	00008293          	mv	t0,ra
   10c9c:	fa0680e7          	jalr	-96(a3)
   10ca0:	00028093          	mv	ra,t0
   10ca4:	ff078793          	addi	a5,a5,-16
   10ca8:	40f70733          	sub	a4,a4,a5
   10cac:	00f60633          	add	a2,a2,a5
   10cb0:	f6c378e3          	bgeu	t1,a2,10c20 <memset+0x44>
   10cb4:	f3dff06f          	j	10bf0 <memset+0x14>

00010cb8 <__call_exitprocs>:
   10cb8:	fd010113          	addi	sp,sp,-48
   10cbc:	01412c23          	sw	s4,24(sp)
   10cc0:	e2018a13          	addi	s4,gp,-480 # 13720 <__atexit>
   10cc4:	03212023          	sw	s2,32(sp)
   10cc8:	000a2903          	lw	s2,0(s4)
   10ccc:	02112623          	sw	ra,44(sp)
   10cd0:	0a090863          	beqz	s2,10d80 <__call_exitprocs+0xc8>
   10cd4:	01312e23          	sw	s3,28(sp)
   10cd8:	01512a23          	sw	s5,20(sp)
   10cdc:	01612823          	sw	s6,16(sp)
   10ce0:	01712623          	sw	s7,12(sp)
   10ce4:	02812423          	sw	s0,40(sp)
   10ce8:	02912223          	sw	s1,36(sp)
   10cec:	01812423          	sw	s8,8(sp)
   10cf0:	00050b13          	mv	s6,a0
   10cf4:	00058b93          	mv	s7,a1
   10cf8:	fff00993          	li	s3,-1
   10cfc:	00100a93          	li	s5,1
   10d00:	00492483          	lw	s1,4(s2)
   10d04:	fff48413          	addi	s0,s1,-1
   10d08:	04044e63          	bltz	s0,10d64 <__call_exitprocs+0xac>
   10d0c:	00249493          	slli	s1,s1,0x2
   10d10:	009904b3          	add	s1,s2,s1
   10d14:	080b9063          	bnez	s7,10d94 <__call_exitprocs+0xdc>
   10d18:	00492783          	lw	a5,4(s2)
   10d1c:	0044a683          	lw	a3,4(s1)
   10d20:	fff78793          	addi	a5,a5,-1
   10d24:	0a878c63          	beq	a5,s0,10ddc <__call_exitprocs+0x124>
   10d28:	0004a223          	sw	zero,4(s1)
   10d2c:	02068663          	beqz	a3,10d58 <__call_exitprocs+0xa0>
   10d30:	18892783          	lw	a5,392(s2)
   10d34:	008a9733          	sll	a4,s5,s0
   10d38:	00492c03          	lw	s8,4(s2)
   10d3c:	00f777b3          	and	a5,a4,a5
   10d40:	06079663          	bnez	a5,10dac <__call_exitprocs+0xf4>
   10d44:	000680e7          	jalr	a3
   10d48:	00492703          	lw	a4,4(s2)
   10d4c:	000a2783          	lw	a5,0(s4)
   10d50:	09871063          	bne	a4,s8,10dd0 <__call_exitprocs+0x118>
   10d54:	07279e63          	bne	a5,s2,10dd0 <__call_exitprocs+0x118>
   10d58:	fff40413          	addi	s0,s0,-1
   10d5c:	ffc48493          	addi	s1,s1,-4
   10d60:	fb341ae3          	bne	s0,s3,10d14 <__call_exitprocs+0x5c>
   10d64:	02812403          	lw	s0,40(sp)
   10d68:	02412483          	lw	s1,36(sp)
   10d6c:	01c12983          	lw	s3,28(sp)
   10d70:	01412a83          	lw	s5,20(sp)
   10d74:	01012b03          	lw	s6,16(sp)
   10d78:	00c12b83          	lw	s7,12(sp)
   10d7c:	00812c03          	lw	s8,8(sp)
   10d80:	02c12083          	lw	ra,44(sp)
   10d84:	02012903          	lw	s2,32(sp)
   10d88:	01812a03          	lw	s4,24(sp)
   10d8c:	03010113          	addi	sp,sp,48
   10d90:	00008067          	ret
   10d94:	1044a783          	lw	a5,260(s1)
   10d98:	f97780e3          	beq	a5,s7,10d18 <__call_exitprocs+0x60>
   10d9c:	fff40413          	addi	s0,s0,-1
   10da0:	ffc48493          	addi	s1,s1,-4
   10da4:	ff3418e3          	bne	s0,s3,10d94 <__call_exitprocs+0xdc>
   10da8:	fbdff06f          	j	10d64 <__call_exitprocs+0xac>
   10dac:	18c92783          	lw	a5,396(s2)
   10db0:	0844a583          	lw	a1,132(s1)
   10db4:	00f77733          	and	a4,a4,a5
   10db8:	02071663          	bnez	a4,10de4 <__call_exitprocs+0x12c>
   10dbc:	000b0513          	mv	a0,s6
   10dc0:	000680e7          	jalr	a3
   10dc4:	00492703          	lw	a4,4(s2)
   10dc8:	000a2783          	lw	a5,0(s4)
   10dcc:	f98704e3          	beq	a4,s8,10d54 <__call_exitprocs+0x9c>
   10dd0:	f8078ae3          	beqz	a5,10d64 <__call_exitprocs+0xac>
   10dd4:	00078913          	mv	s2,a5
   10dd8:	f29ff06f          	j	10d00 <__call_exitprocs+0x48>
   10ddc:	00892223          	sw	s0,4(s2)
   10de0:	f4dff06f          	j	10d2c <__call_exitprocs+0x74>
   10de4:	00058513          	mv	a0,a1
   10de8:	000680e7          	jalr	a3
   10dec:	f5dff06f          	j	10d48 <__call_exitprocs+0x90>

00010df0 <atexit>:
   10df0:	00050593          	mv	a1,a0
   10df4:	00000693          	li	a3,0
   10df8:	00000613          	li	a2,0
   10dfc:	00000513          	li	a0,0
   10e00:	0fc0106f          	j	11efc <__register_exitproc>

00010e04 <_malloc_trim_r>:
   10e04:	fe010113          	addi	sp,sp,-32
   10e08:	00812c23          	sw	s0,24(sp)
   10e0c:	00912a23          	sw	s1,20(sp)
   10e10:	01212823          	sw	s2,16(sp)
   10e14:	01312623          	sw	s3,12(sp)
   10e18:	01412423          	sw	s4,8(sp)
   10e1c:	00058993          	mv	s3,a1
   10e20:	00112e23          	sw	ra,28(sp)
   10e24:	00050913          	mv	s2,a0
   10e28:	00002a17          	auipc	s4,0x2
   10e2c:	4d8a0a13          	addi	s4,s4,1240 # 13300 <__malloc_av_>
   10e30:	3d9000ef          	jal	11a08 <__malloc_lock>
   10e34:	008a2703          	lw	a4,8(s4)
   10e38:	000017b7          	lui	a5,0x1
   10e3c:	fef78793          	addi	a5,a5,-17 # fef <exit-0xf0c5>
   10e40:	00472483          	lw	s1,4(a4)
   10e44:	00001737          	lui	a4,0x1
   10e48:	ffc4f493          	andi	s1,s1,-4
   10e4c:	00f48433          	add	s0,s1,a5
   10e50:	41340433          	sub	s0,s0,s3
   10e54:	00c45413          	srli	s0,s0,0xc
   10e58:	fff40413          	addi	s0,s0,-1
   10e5c:	00c41413          	slli	s0,s0,0xc
   10e60:	00e44e63          	blt	s0,a4,10e7c <_malloc_trim_r+0x78>
   10e64:	00000593          	li	a1,0
   10e68:	00090513          	mv	a0,s2
   10e6c:	7e5000ef          	jal	11e50 <_sbrk_r>
   10e70:	008a2783          	lw	a5,8(s4)
   10e74:	009787b3          	add	a5,a5,s1
   10e78:	02f50863          	beq	a0,a5,10ea8 <_malloc_trim_r+0xa4>
   10e7c:	00090513          	mv	a0,s2
   10e80:	38d000ef          	jal	11a0c <__malloc_unlock>
   10e84:	01c12083          	lw	ra,28(sp)
   10e88:	01812403          	lw	s0,24(sp)
   10e8c:	01412483          	lw	s1,20(sp)
   10e90:	01012903          	lw	s2,16(sp)
   10e94:	00c12983          	lw	s3,12(sp)
   10e98:	00812a03          	lw	s4,8(sp)
   10e9c:	00000513          	li	a0,0
   10ea0:	02010113          	addi	sp,sp,32
   10ea4:	00008067          	ret
   10ea8:	408005b3          	neg	a1,s0
   10eac:	00090513          	mv	a0,s2
   10eb0:	7a1000ef          	jal	11e50 <_sbrk_r>
   10eb4:	fff00793          	li	a5,-1
   10eb8:	04f50863          	beq	a0,a5,10f08 <_malloc_trim_r+0x104>
   10ebc:	f8818713          	addi	a4,gp,-120 # 13888 <__malloc_current_mallinfo>
   10ec0:	00072783          	lw	a5,0(a4) # 1000 <exit-0xf0b4>
   10ec4:	008a2683          	lw	a3,8(s4)
   10ec8:	408484b3          	sub	s1,s1,s0
   10ecc:	0014e493          	ori	s1,s1,1
   10ed0:	408787b3          	sub	a5,a5,s0
   10ed4:	00090513          	mv	a0,s2
   10ed8:	0096a223          	sw	s1,4(a3)
   10edc:	00f72023          	sw	a5,0(a4)
   10ee0:	32d000ef          	jal	11a0c <__malloc_unlock>
   10ee4:	01c12083          	lw	ra,28(sp)
   10ee8:	01812403          	lw	s0,24(sp)
   10eec:	01412483          	lw	s1,20(sp)
   10ef0:	01012903          	lw	s2,16(sp)
   10ef4:	00c12983          	lw	s3,12(sp)
   10ef8:	00812a03          	lw	s4,8(sp)
   10efc:	00100513          	li	a0,1
   10f00:	02010113          	addi	sp,sp,32
   10f04:	00008067          	ret
   10f08:	00000593          	li	a1,0
   10f0c:	00090513          	mv	a0,s2
   10f10:	741000ef          	jal	11e50 <_sbrk_r>
   10f14:	008a2703          	lw	a4,8(s4)
   10f18:	00f00693          	li	a3,15
   10f1c:	40e507b3          	sub	a5,a0,a4
   10f20:	f4f6dee3          	bge	a3,a5,10e7c <_malloc_trim_r+0x78>
   10f24:	e101a683          	lw	a3,-496(gp) # 13710 <__malloc_sbrk_base>
   10f28:	40d50533          	sub	a0,a0,a3
   10f2c:	0017e793          	ori	a5,a5,1
   10f30:	f8a1a423          	sw	a0,-120(gp) # 13888 <__malloc_current_mallinfo>
   10f34:	00f72223          	sw	a5,4(a4)
   10f38:	f45ff06f          	j	10e7c <_malloc_trim_r+0x78>

00010f3c <_free_r>:
   10f3c:	18058263          	beqz	a1,110c0 <_free_r+0x184>
   10f40:	ff010113          	addi	sp,sp,-16
   10f44:	00812423          	sw	s0,8(sp)
   10f48:	00912223          	sw	s1,4(sp)
   10f4c:	00058413          	mv	s0,a1
   10f50:	00050493          	mv	s1,a0
   10f54:	00112623          	sw	ra,12(sp)
   10f58:	2b1000ef          	jal	11a08 <__malloc_lock>
   10f5c:	ffc42583          	lw	a1,-4(s0)
   10f60:	ff840713          	addi	a4,s0,-8
   10f64:	00002517          	auipc	a0,0x2
   10f68:	39c50513          	addi	a0,a0,924 # 13300 <__malloc_av_>
   10f6c:	ffe5f793          	andi	a5,a1,-2
   10f70:	00f70633          	add	a2,a4,a5
   10f74:	00462683          	lw	a3,4(a2)
   10f78:	00852803          	lw	a6,8(a0)
   10f7c:	ffc6f693          	andi	a3,a3,-4
   10f80:	1ac80263          	beq	a6,a2,11124 <_free_r+0x1e8>
   10f84:	00d62223          	sw	a3,4(a2)
   10f88:	0015f593          	andi	a1,a1,1
   10f8c:	00d60833          	add	a6,a2,a3
   10f90:	0a059063          	bnez	a1,11030 <_free_r+0xf4>
   10f94:	ff842303          	lw	t1,-8(s0)
   10f98:	00482583          	lw	a1,4(a6)
   10f9c:	00002897          	auipc	a7,0x2
   10fa0:	36c88893          	addi	a7,a7,876 # 13308 <__malloc_av_+0x8>
   10fa4:	40670733          	sub	a4,a4,t1
   10fa8:	00872803          	lw	a6,8(a4)
   10fac:	006787b3          	add	a5,a5,t1
   10fb0:	0015f593          	andi	a1,a1,1
   10fb4:	15180263          	beq	a6,a7,110f8 <_free_r+0x1bc>
   10fb8:	00c72303          	lw	t1,12(a4)
   10fbc:	00682623          	sw	t1,12(a6)
   10fc0:	01032423          	sw	a6,8(t1) # 101a8 <frame_dummy+0x1c>
   10fc4:	1a058663          	beqz	a1,11170 <_free_r+0x234>
   10fc8:	0017e693          	ori	a3,a5,1
   10fcc:	00d72223          	sw	a3,4(a4)
   10fd0:	00f62023          	sw	a5,0(a2)
   10fd4:	1ff00693          	li	a3,511
   10fd8:	06f6ec63          	bltu	a3,a5,11050 <_free_r+0x114>
   10fdc:	ff87f693          	andi	a3,a5,-8
   10fe0:	00868693          	addi	a3,a3,8
   10fe4:	00452583          	lw	a1,4(a0)
   10fe8:	00d506b3          	add	a3,a0,a3
   10fec:	0006a603          	lw	a2,0(a3)
   10ff0:	0057d813          	srli	a6,a5,0x5
   10ff4:	00100793          	li	a5,1
   10ff8:	010797b3          	sll	a5,a5,a6
   10ffc:	00b7e7b3          	or	a5,a5,a1
   11000:	ff868593          	addi	a1,a3,-8
   11004:	00b72623          	sw	a1,12(a4)
   11008:	00c72423          	sw	a2,8(a4)
   1100c:	00f52223          	sw	a5,4(a0)
   11010:	00e6a023          	sw	a4,0(a3)
   11014:	00e62623          	sw	a4,12(a2)
   11018:	00812403          	lw	s0,8(sp)
   1101c:	00c12083          	lw	ra,12(sp)
   11020:	00048513          	mv	a0,s1
   11024:	00412483          	lw	s1,4(sp)
   11028:	01010113          	addi	sp,sp,16
   1102c:	1e10006f          	j	11a0c <__malloc_unlock>
   11030:	00482583          	lw	a1,4(a6)
   11034:	0015f593          	andi	a1,a1,1
   11038:	08058663          	beqz	a1,110c4 <_free_r+0x188>
   1103c:	0017e693          	ori	a3,a5,1
   11040:	fed42e23          	sw	a3,-4(s0)
   11044:	00f62023          	sw	a5,0(a2)
   11048:	1ff00693          	li	a3,511
   1104c:	f8f6f8e3          	bgeu	a3,a5,10fdc <_free_r+0xa0>
   11050:	0097d693          	srli	a3,a5,0x9
   11054:	00400613          	li	a2,4
   11058:	12d66063          	bltu	a2,a3,11178 <_free_r+0x23c>
   1105c:	0067d693          	srli	a3,a5,0x6
   11060:	03968593          	addi	a1,a3,57
   11064:	03868613          	addi	a2,a3,56
   11068:	00359593          	slli	a1,a1,0x3
   1106c:	00b505b3          	add	a1,a0,a1
   11070:	0005a683          	lw	a3,0(a1)
   11074:	ff858593          	addi	a1,a1,-8
   11078:	00d59863          	bne	a1,a3,11088 <_free_r+0x14c>
   1107c:	1540006f          	j	111d0 <_free_r+0x294>
   11080:	0086a683          	lw	a3,8(a3)
   11084:	00d58863          	beq	a1,a3,11094 <_free_r+0x158>
   11088:	0046a603          	lw	a2,4(a3)
   1108c:	ffc67613          	andi	a2,a2,-4
   11090:	fec7e8e3          	bltu	a5,a2,11080 <_free_r+0x144>
   11094:	00c6a583          	lw	a1,12(a3)
   11098:	00b72623          	sw	a1,12(a4)
   1109c:	00d72423          	sw	a3,8(a4)
   110a0:	00812403          	lw	s0,8(sp)
   110a4:	00c12083          	lw	ra,12(sp)
   110a8:	00e5a423          	sw	a4,8(a1)
   110ac:	00048513          	mv	a0,s1
   110b0:	00412483          	lw	s1,4(sp)
   110b4:	00e6a623          	sw	a4,12(a3)
   110b8:	01010113          	addi	sp,sp,16
   110bc:	1510006f          	j	11a0c <__malloc_unlock>
   110c0:	00008067          	ret
   110c4:	00d787b3          	add	a5,a5,a3
   110c8:	00002897          	auipc	a7,0x2
   110cc:	24088893          	addi	a7,a7,576 # 13308 <__malloc_av_+0x8>
   110d0:	00862683          	lw	a3,8(a2)
   110d4:	0d168c63          	beq	a3,a7,111ac <_free_r+0x270>
   110d8:	00c62803          	lw	a6,12(a2)
   110dc:	0017e593          	ori	a1,a5,1
   110e0:	00f70633          	add	a2,a4,a5
   110e4:	0106a623          	sw	a6,12(a3)
   110e8:	00d82423          	sw	a3,8(a6)
   110ec:	00b72223          	sw	a1,4(a4)
   110f0:	00f62023          	sw	a5,0(a2)
   110f4:	ee1ff06f          	j	10fd4 <_free_r+0x98>
   110f8:	12059c63          	bnez	a1,11230 <_free_r+0x2f4>
   110fc:	00862583          	lw	a1,8(a2)
   11100:	00c62603          	lw	a2,12(a2)
   11104:	00f686b3          	add	a3,a3,a5
   11108:	0016e793          	ori	a5,a3,1
   1110c:	00c5a623          	sw	a2,12(a1)
   11110:	00b62423          	sw	a1,8(a2)
   11114:	00f72223          	sw	a5,4(a4)
   11118:	00d70733          	add	a4,a4,a3
   1111c:	00d72023          	sw	a3,0(a4)
   11120:	ef9ff06f          	j	11018 <_free_r+0xdc>
   11124:	0015f593          	andi	a1,a1,1
   11128:	00d786b3          	add	a3,a5,a3
   1112c:	02059063          	bnez	a1,1114c <_free_r+0x210>
   11130:	ff842583          	lw	a1,-8(s0)
   11134:	40b70733          	sub	a4,a4,a1
   11138:	00c72783          	lw	a5,12(a4)
   1113c:	00872603          	lw	a2,8(a4)
   11140:	00b686b3          	add	a3,a3,a1
   11144:	00f62623          	sw	a5,12(a2)
   11148:	00c7a423          	sw	a2,8(a5)
   1114c:	0016e793          	ori	a5,a3,1
   11150:	00f72223          	sw	a5,4(a4)
   11154:	00e52423          	sw	a4,8(a0)
   11158:	e141a783          	lw	a5,-492(gp) # 13714 <__malloc_trim_threshold>
   1115c:	eaf6eee3          	bltu	a3,a5,11018 <_free_r+0xdc>
   11160:	e2c1a583          	lw	a1,-468(gp) # 1372c <__malloc_top_pad>
   11164:	00048513          	mv	a0,s1
   11168:	c9dff0ef          	jal	10e04 <_malloc_trim_r>
   1116c:	eadff06f          	j	11018 <_free_r+0xdc>
   11170:	00d787b3          	add	a5,a5,a3
   11174:	f5dff06f          	j	110d0 <_free_r+0x194>
   11178:	01400613          	li	a2,20
   1117c:	02d67063          	bgeu	a2,a3,1119c <_free_r+0x260>
   11180:	05400613          	li	a2,84
   11184:	06d66463          	bltu	a2,a3,111ec <_free_r+0x2b0>
   11188:	00c7d693          	srli	a3,a5,0xc
   1118c:	06f68593          	addi	a1,a3,111
   11190:	06e68613          	addi	a2,a3,110
   11194:	00359593          	slli	a1,a1,0x3
   11198:	ed5ff06f          	j	1106c <_free_r+0x130>
   1119c:	05c68593          	addi	a1,a3,92
   111a0:	05b68613          	addi	a2,a3,91
   111a4:	00359593          	slli	a1,a1,0x3
   111a8:	ec5ff06f          	j	1106c <_free_r+0x130>
   111ac:	00e52a23          	sw	a4,20(a0)
   111b0:	00e52823          	sw	a4,16(a0)
   111b4:	0017e693          	ori	a3,a5,1
   111b8:	01172623          	sw	a7,12(a4)
   111bc:	01172423          	sw	a7,8(a4)
   111c0:	00d72223          	sw	a3,4(a4)
   111c4:	00f70733          	add	a4,a4,a5
   111c8:	00f72023          	sw	a5,0(a4)
   111cc:	e4dff06f          	j	11018 <_free_r+0xdc>
   111d0:	00452803          	lw	a6,4(a0)
   111d4:	40265613          	srai	a2,a2,0x2
   111d8:	00100793          	li	a5,1
   111dc:	00c797b3          	sll	a5,a5,a2
   111e0:	0107e7b3          	or	a5,a5,a6
   111e4:	00f52223          	sw	a5,4(a0)
   111e8:	eb1ff06f          	j	11098 <_free_r+0x15c>
   111ec:	15400613          	li	a2,340
   111f0:	00d66c63          	bltu	a2,a3,11208 <_free_r+0x2cc>
   111f4:	00f7d693          	srli	a3,a5,0xf
   111f8:	07868593          	addi	a1,a3,120
   111fc:	07768613          	addi	a2,a3,119
   11200:	00359593          	slli	a1,a1,0x3
   11204:	e69ff06f          	j	1106c <_free_r+0x130>
   11208:	55400613          	li	a2,1364
   1120c:	00d66c63          	bltu	a2,a3,11224 <_free_r+0x2e8>
   11210:	0127d693          	srli	a3,a5,0x12
   11214:	07d68593          	addi	a1,a3,125
   11218:	07c68613          	addi	a2,a3,124
   1121c:	00359593          	slli	a1,a1,0x3
   11220:	e4dff06f          	j	1106c <_free_r+0x130>
   11224:	3f800593          	li	a1,1016
   11228:	07e00613          	li	a2,126
   1122c:	e41ff06f          	j	1106c <_free_r+0x130>
   11230:	0017e693          	ori	a3,a5,1
   11234:	00d72223          	sw	a3,4(a4)
   11238:	00f62023          	sw	a5,0(a2)
   1123c:	dddff06f          	j	11018 <_free_r+0xdc>

00011240 <_malloc_r>:
   11240:	fd010113          	addi	sp,sp,-48
   11244:	03212023          	sw	s2,32(sp)
   11248:	02112623          	sw	ra,44(sp)
   1124c:	02812423          	sw	s0,40(sp)
   11250:	02912223          	sw	s1,36(sp)
   11254:	01312e23          	sw	s3,28(sp)
   11258:	00b58793          	addi	a5,a1,11
   1125c:	01600713          	li	a4,22
   11260:	00050913          	mv	s2,a0
   11264:	08f76263          	bltu	a4,a5,112e8 <_malloc_r+0xa8>
   11268:	01000793          	li	a5,16
   1126c:	20b7e663          	bltu	a5,a1,11478 <_malloc_r+0x238>
   11270:	798000ef          	jal	11a08 <__malloc_lock>
   11274:	01800793          	li	a5,24
   11278:	00200593          	li	a1,2
   1127c:	01000493          	li	s1,16
   11280:	00002997          	auipc	s3,0x2
   11284:	08098993          	addi	s3,s3,128 # 13300 <__malloc_av_>
   11288:	00f987b3          	add	a5,s3,a5
   1128c:	0047a403          	lw	s0,4(a5)
   11290:	ff878713          	addi	a4,a5,-8
   11294:	34e40a63          	beq	s0,a4,115e8 <_malloc_r+0x3a8>
   11298:	00442783          	lw	a5,4(s0)
   1129c:	00c42683          	lw	a3,12(s0)
   112a0:	00842603          	lw	a2,8(s0)
   112a4:	ffc7f793          	andi	a5,a5,-4
   112a8:	00f407b3          	add	a5,s0,a5
   112ac:	0047a703          	lw	a4,4(a5)
   112b0:	00d62623          	sw	a3,12(a2)
   112b4:	00c6a423          	sw	a2,8(a3)
   112b8:	00176713          	ori	a4,a4,1
   112bc:	00090513          	mv	a0,s2
   112c0:	00e7a223          	sw	a4,4(a5)
   112c4:	748000ef          	jal	11a0c <__malloc_unlock>
   112c8:	00840513          	addi	a0,s0,8
   112cc:	02c12083          	lw	ra,44(sp)
   112d0:	02812403          	lw	s0,40(sp)
   112d4:	02412483          	lw	s1,36(sp)
   112d8:	02012903          	lw	s2,32(sp)
   112dc:	01c12983          	lw	s3,28(sp)
   112e0:	03010113          	addi	sp,sp,48
   112e4:	00008067          	ret
   112e8:	ff87f493          	andi	s1,a5,-8
   112ec:	1807c663          	bltz	a5,11478 <_malloc_r+0x238>
   112f0:	18b4e463          	bltu	s1,a1,11478 <_malloc_r+0x238>
   112f4:	714000ef          	jal	11a08 <__malloc_lock>
   112f8:	1f700793          	li	a5,503
   112fc:	4097f063          	bgeu	a5,s1,116fc <_malloc_r+0x4bc>
   11300:	0094d793          	srli	a5,s1,0x9
   11304:	18078263          	beqz	a5,11488 <_malloc_r+0x248>
   11308:	00400713          	li	a4,4
   1130c:	34f76663          	bltu	a4,a5,11658 <_malloc_r+0x418>
   11310:	0064d793          	srli	a5,s1,0x6
   11314:	03978593          	addi	a1,a5,57
   11318:	03878813          	addi	a6,a5,56
   1131c:	00359613          	slli	a2,a1,0x3
   11320:	00002997          	auipc	s3,0x2
   11324:	fe098993          	addi	s3,s3,-32 # 13300 <__malloc_av_>
   11328:	00c98633          	add	a2,s3,a2
   1132c:	00462403          	lw	s0,4(a2)
   11330:	ff860613          	addi	a2,a2,-8
   11334:	02860863          	beq	a2,s0,11364 <_malloc_r+0x124>
   11338:	00f00513          	li	a0,15
   1133c:	0140006f          	j	11350 <_malloc_r+0x110>
   11340:	00c42683          	lw	a3,12(s0)
   11344:	28075e63          	bgez	a4,115e0 <_malloc_r+0x3a0>
   11348:	00d60e63          	beq	a2,a3,11364 <_malloc_r+0x124>
   1134c:	00068413          	mv	s0,a3
   11350:	00442783          	lw	a5,4(s0)
   11354:	ffc7f793          	andi	a5,a5,-4
   11358:	40978733          	sub	a4,a5,s1
   1135c:	fee552e3          	bge	a0,a4,11340 <_malloc_r+0x100>
   11360:	00080593          	mv	a1,a6
   11364:	0109a403          	lw	s0,16(s3)
   11368:	00002897          	auipc	a7,0x2
   1136c:	fa088893          	addi	a7,a7,-96 # 13308 <__malloc_av_+0x8>
   11370:	27140463          	beq	s0,a7,115d8 <_malloc_r+0x398>
   11374:	00442783          	lw	a5,4(s0)
   11378:	00f00693          	li	a3,15
   1137c:	ffc7f793          	andi	a5,a5,-4
   11380:	40978733          	sub	a4,a5,s1
   11384:	38e6c263          	blt	a3,a4,11708 <_malloc_r+0x4c8>
   11388:	0119aa23          	sw	a7,20(s3)
   1138c:	0119a823          	sw	a7,16(s3)
   11390:	34075663          	bgez	a4,116dc <_malloc_r+0x49c>
   11394:	1ff00713          	li	a4,511
   11398:	0049a503          	lw	a0,4(s3)
   1139c:	24f76e63          	bltu	a4,a5,115f8 <_malloc_r+0x3b8>
   113a0:	ff87f713          	andi	a4,a5,-8
   113a4:	00870713          	addi	a4,a4,8
   113a8:	00e98733          	add	a4,s3,a4
   113ac:	00072683          	lw	a3,0(a4)
   113b0:	0057d613          	srli	a2,a5,0x5
   113b4:	00100793          	li	a5,1
   113b8:	00c797b3          	sll	a5,a5,a2
   113bc:	00f56533          	or	a0,a0,a5
   113c0:	ff870793          	addi	a5,a4,-8
   113c4:	00f42623          	sw	a5,12(s0)
   113c8:	00d42423          	sw	a3,8(s0)
   113cc:	00a9a223          	sw	a0,4(s3)
   113d0:	00872023          	sw	s0,0(a4)
   113d4:	0086a623          	sw	s0,12(a3)
   113d8:	4025d793          	srai	a5,a1,0x2
   113dc:	00100613          	li	a2,1
   113e0:	00f61633          	sll	a2,a2,a5
   113e4:	0ac56a63          	bltu	a0,a2,11498 <_malloc_r+0x258>
   113e8:	00a677b3          	and	a5,a2,a0
   113ec:	02079463          	bnez	a5,11414 <_malloc_r+0x1d4>
   113f0:	00161613          	slli	a2,a2,0x1
   113f4:	ffc5f593          	andi	a1,a1,-4
   113f8:	00a677b3          	and	a5,a2,a0
   113fc:	00458593          	addi	a1,a1,4
   11400:	00079a63          	bnez	a5,11414 <_malloc_r+0x1d4>
   11404:	00161613          	slli	a2,a2,0x1
   11408:	00a677b3          	and	a5,a2,a0
   1140c:	00458593          	addi	a1,a1,4
   11410:	fe078ae3          	beqz	a5,11404 <_malloc_r+0x1c4>
   11414:	00f00813          	li	a6,15
   11418:	00359313          	slli	t1,a1,0x3
   1141c:	00698333          	add	t1,s3,t1
   11420:	00030513          	mv	a0,t1
   11424:	00c52783          	lw	a5,12(a0)
   11428:	00058e13          	mv	t3,a1
   1142c:	24f50863          	beq	a0,a5,1167c <_malloc_r+0x43c>
   11430:	0047a703          	lw	a4,4(a5)
   11434:	00078413          	mv	s0,a5
   11438:	00c7a783          	lw	a5,12(a5)
   1143c:	ffc77713          	andi	a4,a4,-4
   11440:	409706b3          	sub	a3,a4,s1
   11444:	24d84863          	blt	a6,a3,11694 <_malloc_r+0x454>
   11448:	fe06c2e3          	bltz	a3,1142c <_malloc_r+0x1ec>
   1144c:	00e40733          	add	a4,s0,a4
   11450:	00472683          	lw	a3,4(a4)
   11454:	00842603          	lw	a2,8(s0)
   11458:	00090513          	mv	a0,s2
   1145c:	0016e693          	ori	a3,a3,1
   11460:	00d72223          	sw	a3,4(a4)
   11464:	00f62623          	sw	a5,12(a2)
   11468:	00c7a423          	sw	a2,8(a5)
   1146c:	5a0000ef          	jal	11a0c <__malloc_unlock>
   11470:	00840513          	addi	a0,s0,8
   11474:	e59ff06f          	j	112cc <_malloc_r+0x8c>
   11478:	00c00793          	li	a5,12
   1147c:	00f92023          	sw	a5,0(s2)
   11480:	00000513          	li	a0,0
   11484:	e49ff06f          	j	112cc <_malloc_r+0x8c>
   11488:	20000613          	li	a2,512
   1148c:	04000593          	li	a1,64
   11490:	03f00813          	li	a6,63
   11494:	e8dff06f          	j	11320 <_malloc_r+0xe0>
   11498:	0089a403          	lw	s0,8(s3)
   1149c:	01612823          	sw	s6,16(sp)
   114a0:	00442783          	lw	a5,4(s0)
   114a4:	ffc7fb13          	andi	s6,a5,-4
   114a8:	009b6863          	bltu	s6,s1,114b8 <_malloc_r+0x278>
   114ac:	409b0733          	sub	a4,s6,s1
   114b0:	00f00793          	li	a5,15
   114b4:	0ee7c063          	blt	a5,a4,11594 <_malloc_r+0x354>
   114b8:	01912223          	sw	s9,4(sp)
   114bc:	e1018c93          	addi	s9,gp,-496 # 13710 <__malloc_sbrk_base>
   114c0:	000ca703          	lw	a4,0(s9)
   114c4:	01412c23          	sw	s4,24(sp)
   114c8:	01512a23          	sw	s5,20(sp)
   114cc:	01712623          	sw	s7,12(sp)
   114d0:	e2c1aa83          	lw	s5,-468(gp) # 1372c <__malloc_top_pad>
   114d4:	fff00793          	li	a5,-1
   114d8:	01640a33          	add	s4,s0,s6
   114dc:	01548ab3          	add	s5,s1,s5
   114e0:	3cf70a63          	beq	a4,a5,118b4 <_malloc_r+0x674>
   114e4:	000017b7          	lui	a5,0x1
   114e8:	00f78793          	addi	a5,a5,15 # 100f <exit-0xf0a5>
   114ec:	00fa8ab3          	add	s5,s5,a5
   114f0:	fffff7b7          	lui	a5,0xfffff
   114f4:	00fafab3          	and	s5,s5,a5
   114f8:	000a8593          	mv	a1,s5
   114fc:	00090513          	mv	a0,s2
   11500:	151000ef          	jal	11e50 <_sbrk_r>
   11504:	fff00793          	li	a5,-1
   11508:	00050b93          	mv	s7,a0
   1150c:	44f50e63          	beq	a0,a5,11968 <_malloc_r+0x728>
   11510:	01812423          	sw	s8,8(sp)
   11514:	25456263          	bltu	a0,s4,11758 <_malloc_r+0x518>
   11518:	f8818c13          	addi	s8,gp,-120 # 13888 <__malloc_current_mallinfo>
   1151c:	000c2583          	lw	a1,0(s8)
   11520:	00ba85b3          	add	a1,s5,a1
   11524:	00bc2023          	sw	a1,0(s8)
   11528:	00058713          	mv	a4,a1
   1152c:	2aaa1a63          	bne	s4,a0,117e0 <_malloc_r+0x5a0>
   11530:	01451793          	slli	a5,a0,0x14
   11534:	2a079663          	bnez	a5,117e0 <_malloc_r+0x5a0>
   11538:	0089ab83          	lw	s7,8(s3)
   1153c:	015b07b3          	add	a5,s6,s5
   11540:	0017e793          	ori	a5,a5,1
   11544:	00fba223          	sw	a5,4(s7)
   11548:	e2818713          	addi	a4,gp,-472 # 13728 <__malloc_max_sbrked_mem>
   1154c:	00072683          	lw	a3,0(a4)
   11550:	00b6f463          	bgeu	a3,a1,11558 <_malloc_r+0x318>
   11554:	00b72023          	sw	a1,0(a4)
   11558:	e2418713          	addi	a4,gp,-476 # 13724 <__malloc_max_total_mem>
   1155c:	00072683          	lw	a3,0(a4)
   11560:	00b6f463          	bgeu	a3,a1,11568 <_malloc_r+0x328>
   11564:	00b72023          	sw	a1,0(a4)
   11568:	00812c03          	lw	s8,8(sp)
   1156c:	000b8413          	mv	s0,s7
   11570:	ffc7f793          	andi	a5,a5,-4
   11574:	40978733          	sub	a4,a5,s1
   11578:	3897ea63          	bltu	a5,s1,1190c <_malloc_r+0x6cc>
   1157c:	00f00793          	li	a5,15
   11580:	38e7d663          	bge	a5,a4,1190c <_malloc_r+0x6cc>
   11584:	01812a03          	lw	s4,24(sp)
   11588:	01412a83          	lw	s5,20(sp)
   1158c:	00c12b83          	lw	s7,12(sp)
   11590:	00412c83          	lw	s9,4(sp)
   11594:	0014e793          	ori	a5,s1,1
   11598:	00f42223          	sw	a5,4(s0)
   1159c:	009404b3          	add	s1,s0,s1
   115a0:	0099a423          	sw	s1,8(s3)
   115a4:	00176713          	ori	a4,a4,1
   115a8:	00090513          	mv	a0,s2
   115ac:	00e4a223          	sw	a4,4(s1)
   115b0:	45c000ef          	jal	11a0c <__malloc_unlock>
   115b4:	02c12083          	lw	ra,44(sp)
   115b8:	00840513          	addi	a0,s0,8
   115bc:	02812403          	lw	s0,40(sp)
   115c0:	01012b03          	lw	s6,16(sp)
   115c4:	02412483          	lw	s1,36(sp)
   115c8:	02012903          	lw	s2,32(sp)
   115cc:	01c12983          	lw	s3,28(sp)
   115d0:	03010113          	addi	sp,sp,48
   115d4:	00008067          	ret
   115d8:	0049a503          	lw	a0,4(s3)
   115dc:	dfdff06f          	j	113d8 <_malloc_r+0x198>
   115e0:	00842603          	lw	a2,8(s0)
   115e4:	cc5ff06f          	j	112a8 <_malloc_r+0x68>
   115e8:	00c7a403          	lw	s0,12(a5) # fffff00c <__BSS_END__+0xfffeb5cc>
   115ec:	00258593          	addi	a1,a1,2
   115f0:	d6878ae3          	beq	a5,s0,11364 <_malloc_r+0x124>
   115f4:	ca5ff06f          	j	11298 <_malloc_r+0x58>
   115f8:	0097d713          	srli	a4,a5,0x9
   115fc:	00400693          	li	a3,4
   11600:	14e6f263          	bgeu	a3,a4,11744 <_malloc_r+0x504>
   11604:	01400693          	li	a3,20
   11608:	32e6e463          	bltu	a3,a4,11930 <_malloc_r+0x6f0>
   1160c:	05c70613          	addi	a2,a4,92
   11610:	05b70693          	addi	a3,a4,91
   11614:	00361613          	slli	a2,a2,0x3
   11618:	00c98633          	add	a2,s3,a2
   1161c:	00062703          	lw	a4,0(a2)
   11620:	ff860613          	addi	a2,a2,-8
   11624:	00e61863          	bne	a2,a4,11634 <_malloc_r+0x3f4>
   11628:	2940006f          	j	118bc <_malloc_r+0x67c>
   1162c:	00872703          	lw	a4,8(a4)
   11630:	00e60863          	beq	a2,a4,11640 <_malloc_r+0x400>
   11634:	00472683          	lw	a3,4(a4)
   11638:	ffc6f693          	andi	a3,a3,-4
   1163c:	fed7e8e3          	bltu	a5,a3,1162c <_malloc_r+0x3ec>
   11640:	00c72603          	lw	a2,12(a4)
   11644:	00c42623          	sw	a2,12(s0)
   11648:	00e42423          	sw	a4,8(s0)
   1164c:	00862423          	sw	s0,8(a2)
   11650:	00872623          	sw	s0,12(a4)
   11654:	d85ff06f          	j	113d8 <_malloc_r+0x198>
   11658:	01400713          	li	a4,20
   1165c:	10f77863          	bgeu	a4,a5,1176c <_malloc_r+0x52c>
   11660:	05400713          	li	a4,84
   11664:	2ef76463          	bltu	a4,a5,1194c <_malloc_r+0x70c>
   11668:	00c4d793          	srli	a5,s1,0xc
   1166c:	06f78593          	addi	a1,a5,111
   11670:	06e78813          	addi	a6,a5,110
   11674:	00359613          	slli	a2,a1,0x3
   11678:	ca9ff06f          	j	11320 <_malloc_r+0xe0>
   1167c:	001e0e13          	addi	t3,t3,1
   11680:	003e7793          	andi	a5,t3,3
   11684:	00850513          	addi	a0,a0,8
   11688:	10078063          	beqz	a5,11788 <_malloc_r+0x548>
   1168c:	00c52783          	lw	a5,12(a0)
   11690:	d9dff06f          	j	1142c <_malloc_r+0x1ec>
   11694:	00842603          	lw	a2,8(s0)
   11698:	0014e593          	ori	a1,s1,1
   1169c:	00b42223          	sw	a1,4(s0)
   116a0:	00f62623          	sw	a5,12(a2)
   116a4:	00c7a423          	sw	a2,8(a5)
   116a8:	009404b3          	add	s1,s0,s1
   116ac:	0099aa23          	sw	s1,20(s3)
   116b0:	0099a823          	sw	s1,16(s3)
   116b4:	0016e793          	ori	a5,a3,1
   116b8:	0114a623          	sw	a7,12(s1)
   116bc:	0114a423          	sw	a7,8(s1)
   116c0:	00f4a223          	sw	a5,4(s1)
   116c4:	00e40733          	add	a4,s0,a4
   116c8:	00090513          	mv	a0,s2
   116cc:	00d72023          	sw	a3,0(a4)
   116d0:	33c000ef          	jal	11a0c <__malloc_unlock>
   116d4:	00840513          	addi	a0,s0,8
   116d8:	bf5ff06f          	j	112cc <_malloc_r+0x8c>
   116dc:	00f407b3          	add	a5,s0,a5
   116e0:	0047a703          	lw	a4,4(a5)
   116e4:	00090513          	mv	a0,s2
   116e8:	00176713          	ori	a4,a4,1
   116ec:	00e7a223          	sw	a4,4(a5)
   116f0:	31c000ef          	jal	11a0c <__malloc_unlock>
   116f4:	00840513          	addi	a0,s0,8
   116f8:	bd5ff06f          	j	112cc <_malloc_r+0x8c>
   116fc:	0034d593          	srli	a1,s1,0x3
   11700:	00848793          	addi	a5,s1,8
   11704:	b7dff06f          	j	11280 <_malloc_r+0x40>
   11708:	0014e693          	ori	a3,s1,1
   1170c:	00d42223          	sw	a3,4(s0)
   11710:	009404b3          	add	s1,s0,s1
   11714:	0099aa23          	sw	s1,20(s3)
   11718:	0099a823          	sw	s1,16(s3)
   1171c:	00176693          	ori	a3,a4,1
   11720:	0114a623          	sw	a7,12(s1)
   11724:	0114a423          	sw	a7,8(s1)
   11728:	00d4a223          	sw	a3,4(s1)
   1172c:	00f407b3          	add	a5,s0,a5
   11730:	00090513          	mv	a0,s2
   11734:	00e7a023          	sw	a4,0(a5)
   11738:	2d4000ef          	jal	11a0c <__malloc_unlock>
   1173c:	00840513          	addi	a0,s0,8
   11740:	b8dff06f          	j	112cc <_malloc_r+0x8c>
   11744:	0067d713          	srli	a4,a5,0x6
   11748:	03970613          	addi	a2,a4,57
   1174c:	03870693          	addi	a3,a4,56
   11750:	00361613          	slli	a2,a2,0x3
   11754:	ec5ff06f          	j	11618 <_malloc_r+0x3d8>
   11758:	07340c63          	beq	s0,s3,117d0 <_malloc_r+0x590>
   1175c:	0089a403          	lw	s0,8(s3)
   11760:	00812c03          	lw	s8,8(sp)
   11764:	00442783          	lw	a5,4(s0)
   11768:	e09ff06f          	j	11570 <_malloc_r+0x330>
   1176c:	05c78593          	addi	a1,a5,92
   11770:	05b78813          	addi	a6,a5,91
   11774:	00359613          	slli	a2,a1,0x3
   11778:	ba9ff06f          	j	11320 <_malloc_r+0xe0>
   1177c:	00832783          	lw	a5,8(t1)
   11780:	fff58593          	addi	a1,a1,-1
   11784:	26679e63          	bne	a5,t1,11a00 <_malloc_r+0x7c0>
   11788:	0035f793          	andi	a5,a1,3
   1178c:	ff830313          	addi	t1,t1,-8
   11790:	fe0796e3          	bnez	a5,1177c <_malloc_r+0x53c>
   11794:	0049a703          	lw	a4,4(s3)
   11798:	fff64793          	not	a5,a2
   1179c:	00e7f7b3          	and	a5,a5,a4
   117a0:	00f9a223          	sw	a5,4(s3)
   117a4:	00161613          	slli	a2,a2,0x1
   117a8:	cec7e8e3          	bltu	a5,a2,11498 <_malloc_r+0x258>
   117ac:	ce0606e3          	beqz	a2,11498 <_malloc_r+0x258>
   117b0:	00f67733          	and	a4,a2,a5
   117b4:	00071a63          	bnez	a4,117c8 <_malloc_r+0x588>
   117b8:	00161613          	slli	a2,a2,0x1
   117bc:	00f67733          	and	a4,a2,a5
   117c0:	004e0e13          	addi	t3,t3,4
   117c4:	fe070ae3          	beqz	a4,117b8 <_malloc_r+0x578>
   117c8:	000e0593          	mv	a1,t3
   117cc:	c4dff06f          	j	11418 <_malloc_r+0x1d8>
   117d0:	f8818c13          	addi	s8,gp,-120 # 13888 <__malloc_current_mallinfo>
   117d4:	000c2703          	lw	a4,0(s8)
   117d8:	00ea8733          	add	a4,s5,a4
   117dc:	00ec2023          	sw	a4,0(s8)
   117e0:	000ca683          	lw	a3,0(s9)
   117e4:	fff00793          	li	a5,-1
   117e8:	18f68663          	beq	a3,a5,11974 <_malloc_r+0x734>
   117ec:	414b87b3          	sub	a5,s7,s4
   117f0:	00e787b3          	add	a5,a5,a4
   117f4:	00fc2023          	sw	a5,0(s8)
   117f8:	007bfc93          	andi	s9,s7,7
   117fc:	0c0c8c63          	beqz	s9,118d4 <_malloc_r+0x694>
   11800:	419b8bb3          	sub	s7,s7,s9
   11804:	000017b7          	lui	a5,0x1
   11808:	00878793          	addi	a5,a5,8 # 1008 <exit-0xf0ac>
   1180c:	008b8b93          	addi	s7,s7,8
   11810:	419785b3          	sub	a1,a5,s9
   11814:	015b8ab3          	add	s5,s7,s5
   11818:	415585b3          	sub	a1,a1,s5
   1181c:	01459593          	slli	a1,a1,0x14
   11820:	0145da13          	srli	s4,a1,0x14
   11824:	000a0593          	mv	a1,s4
   11828:	00090513          	mv	a0,s2
   1182c:	624000ef          	jal	11e50 <_sbrk_r>
   11830:	fff00793          	li	a5,-1
   11834:	18f50063          	beq	a0,a5,119b4 <_malloc_r+0x774>
   11838:	41750533          	sub	a0,a0,s7
   1183c:	01450ab3          	add	s5,a0,s4
   11840:	000c2703          	lw	a4,0(s8)
   11844:	0179a423          	sw	s7,8(s3)
   11848:	001ae793          	ori	a5,s5,1
   1184c:	00ea05b3          	add	a1,s4,a4
   11850:	00bc2023          	sw	a1,0(s8)
   11854:	00fba223          	sw	a5,4(s7)
   11858:	cf3408e3          	beq	s0,s3,11548 <_malloc_r+0x308>
   1185c:	00f00693          	li	a3,15
   11860:	0b66f063          	bgeu	a3,s6,11900 <_malloc_r+0x6c0>
   11864:	00442703          	lw	a4,4(s0)
   11868:	ff4b0793          	addi	a5,s6,-12
   1186c:	ff87f793          	andi	a5,a5,-8
   11870:	00177713          	andi	a4,a4,1
   11874:	00f76733          	or	a4,a4,a5
   11878:	00e42223          	sw	a4,4(s0)
   1187c:	00500613          	li	a2,5
   11880:	00f40733          	add	a4,s0,a5
   11884:	00c72223          	sw	a2,4(a4)
   11888:	00c72423          	sw	a2,8(a4)
   1188c:	00f6e663          	bltu	a3,a5,11898 <_malloc_r+0x658>
   11890:	004ba783          	lw	a5,4(s7)
   11894:	cb5ff06f          	j	11548 <_malloc_r+0x308>
   11898:	00840593          	addi	a1,s0,8
   1189c:	00090513          	mv	a0,s2
   118a0:	e9cff0ef          	jal	10f3c <_free_r>
   118a4:	0089ab83          	lw	s7,8(s3)
   118a8:	000c2583          	lw	a1,0(s8)
   118ac:	004ba783          	lw	a5,4(s7)
   118b0:	c99ff06f          	j	11548 <_malloc_r+0x308>
   118b4:	010a8a93          	addi	s5,s5,16
   118b8:	c41ff06f          	j	114f8 <_malloc_r+0x2b8>
   118bc:	4026d693          	srai	a3,a3,0x2
   118c0:	00100793          	li	a5,1
   118c4:	00d797b3          	sll	a5,a5,a3
   118c8:	00f56533          	or	a0,a0,a5
   118cc:	00a9a223          	sw	a0,4(s3)
   118d0:	d75ff06f          	j	11644 <_malloc_r+0x404>
   118d4:	015b85b3          	add	a1,s7,s5
   118d8:	40b005b3          	neg	a1,a1
   118dc:	01459593          	slli	a1,a1,0x14
   118e0:	0145da13          	srli	s4,a1,0x14
   118e4:	000a0593          	mv	a1,s4
   118e8:	00090513          	mv	a0,s2
   118ec:	564000ef          	jal	11e50 <_sbrk_r>
   118f0:	fff00793          	li	a5,-1
   118f4:	f4f512e3          	bne	a0,a5,11838 <_malloc_r+0x5f8>
   118f8:	00000a13          	li	s4,0
   118fc:	f45ff06f          	j	11840 <_malloc_r+0x600>
   11900:	00812c03          	lw	s8,8(sp)
   11904:	00100793          	li	a5,1
   11908:	00fba223          	sw	a5,4(s7)
   1190c:	00090513          	mv	a0,s2
   11910:	0fc000ef          	jal	11a0c <__malloc_unlock>
   11914:	00000513          	li	a0,0
   11918:	01812a03          	lw	s4,24(sp)
   1191c:	01412a83          	lw	s5,20(sp)
   11920:	01012b03          	lw	s6,16(sp)
   11924:	00c12b83          	lw	s7,12(sp)
   11928:	00412c83          	lw	s9,4(sp)
   1192c:	9a1ff06f          	j	112cc <_malloc_r+0x8c>
   11930:	05400693          	li	a3,84
   11934:	04e6e463          	bltu	a3,a4,1197c <_malloc_r+0x73c>
   11938:	00c7d713          	srli	a4,a5,0xc
   1193c:	06f70613          	addi	a2,a4,111
   11940:	06e70693          	addi	a3,a4,110
   11944:	00361613          	slli	a2,a2,0x3
   11948:	cd1ff06f          	j	11618 <_malloc_r+0x3d8>
   1194c:	15400713          	li	a4,340
   11950:	04f76463          	bltu	a4,a5,11998 <_malloc_r+0x758>
   11954:	00f4d793          	srli	a5,s1,0xf
   11958:	07878593          	addi	a1,a5,120
   1195c:	07778813          	addi	a6,a5,119
   11960:	00359613          	slli	a2,a1,0x3
   11964:	9bdff06f          	j	11320 <_malloc_r+0xe0>
   11968:	0089a403          	lw	s0,8(s3)
   1196c:	00442783          	lw	a5,4(s0)
   11970:	c01ff06f          	j	11570 <_malloc_r+0x330>
   11974:	017ca023          	sw	s7,0(s9)
   11978:	e81ff06f          	j	117f8 <_malloc_r+0x5b8>
   1197c:	15400693          	li	a3,340
   11980:	04e6e463          	bltu	a3,a4,119c8 <_malloc_r+0x788>
   11984:	00f7d713          	srli	a4,a5,0xf
   11988:	07870613          	addi	a2,a4,120
   1198c:	07770693          	addi	a3,a4,119
   11990:	00361613          	slli	a2,a2,0x3
   11994:	c85ff06f          	j	11618 <_malloc_r+0x3d8>
   11998:	55400713          	li	a4,1364
   1199c:	04f76463          	bltu	a4,a5,119e4 <_malloc_r+0x7a4>
   119a0:	0124d793          	srli	a5,s1,0x12
   119a4:	07d78593          	addi	a1,a5,125
   119a8:	07c78813          	addi	a6,a5,124
   119ac:	00359613          	slli	a2,a1,0x3
   119b0:	971ff06f          	j	11320 <_malloc_r+0xe0>
   119b4:	ff8c8c93          	addi	s9,s9,-8
   119b8:	019a8ab3          	add	s5,s5,s9
   119bc:	417a8ab3          	sub	s5,s5,s7
   119c0:	00000a13          	li	s4,0
   119c4:	e7dff06f          	j	11840 <_malloc_r+0x600>
   119c8:	55400693          	li	a3,1364
   119cc:	02e6e463          	bltu	a3,a4,119f4 <_malloc_r+0x7b4>
   119d0:	0127d713          	srli	a4,a5,0x12
   119d4:	07d70613          	addi	a2,a4,125
   119d8:	07c70693          	addi	a3,a4,124
   119dc:	00361613          	slli	a2,a2,0x3
   119e0:	c39ff06f          	j	11618 <_malloc_r+0x3d8>
   119e4:	3f800613          	li	a2,1016
   119e8:	07f00593          	li	a1,127
   119ec:	07e00813          	li	a6,126
   119f0:	931ff06f          	j	11320 <_malloc_r+0xe0>
   119f4:	3f800613          	li	a2,1016
   119f8:	07e00693          	li	a3,126
   119fc:	c1dff06f          	j	11618 <_malloc_r+0x3d8>
   11a00:	0049a783          	lw	a5,4(s3)
   11a04:	da1ff06f          	j	117a4 <_malloc_r+0x564>

00011a08 <__malloc_lock>:
   11a08:	00008067          	ret

00011a0c <__malloc_unlock>:
   11a0c:	00008067          	ret

00011a10 <_fclose_r>:
   11a10:	ff010113          	addi	sp,sp,-16
   11a14:	00112623          	sw	ra,12(sp)
   11a18:	01212023          	sw	s2,0(sp)
   11a1c:	02058863          	beqz	a1,11a4c <_fclose_r+0x3c>
   11a20:	00812423          	sw	s0,8(sp)
   11a24:	00912223          	sw	s1,4(sp)
   11a28:	00058413          	mv	s0,a1
   11a2c:	00050493          	mv	s1,a0
   11a30:	00050663          	beqz	a0,11a3c <_fclose_r+0x2c>
   11a34:	03452783          	lw	a5,52(a0)
   11a38:	0c078c63          	beqz	a5,11b10 <_fclose_r+0x100>
   11a3c:	00c41783          	lh	a5,12(s0)
   11a40:	02079263          	bnez	a5,11a64 <_fclose_r+0x54>
   11a44:	00812403          	lw	s0,8(sp)
   11a48:	00412483          	lw	s1,4(sp)
   11a4c:	00c12083          	lw	ra,12(sp)
   11a50:	00000913          	li	s2,0
   11a54:	00090513          	mv	a0,s2
   11a58:	00012903          	lw	s2,0(sp)
   11a5c:	01010113          	addi	sp,sp,16
   11a60:	00008067          	ret
   11a64:	00040593          	mv	a1,s0
   11a68:	00048513          	mv	a0,s1
   11a6c:	0b8000ef          	jal	11b24 <__sflush_r>
   11a70:	02c42783          	lw	a5,44(s0)
   11a74:	00050913          	mv	s2,a0
   11a78:	00078a63          	beqz	a5,11a8c <_fclose_r+0x7c>
   11a7c:	01c42583          	lw	a1,28(s0)
   11a80:	00048513          	mv	a0,s1
   11a84:	000780e7          	jalr	a5
   11a88:	06054463          	bltz	a0,11af0 <_fclose_r+0xe0>
   11a8c:	00c45783          	lhu	a5,12(s0)
   11a90:	0807f793          	andi	a5,a5,128
   11a94:	06079663          	bnez	a5,11b00 <_fclose_r+0xf0>
   11a98:	03042583          	lw	a1,48(s0)
   11a9c:	00058c63          	beqz	a1,11ab4 <_fclose_r+0xa4>
   11aa0:	04040793          	addi	a5,s0,64
   11aa4:	00f58663          	beq	a1,a5,11ab0 <_fclose_r+0xa0>
   11aa8:	00048513          	mv	a0,s1
   11aac:	c90ff0ef          	jal	10f3c <_free_r>
   11ab0:	02042823          	sw	zero,48(s0)
   11ab4:	04442583          	lw	a1,68(s0)
   11ab8:	00058863          	beqz	a1,11ac8 <_fclose_r+0xb8>
   11abc:	00048513          	mv	a0,s1
   11ac0:	c7cff0ef          	jal	10f3c <_free_r>
   11ac4:	04042223          	sw	zero,68(s0)
   11ac8:	c09fe0ef          	jal	106d0 <__sfp_lock_acquire>
   11acc:	00041623          	sh	zero,12(s0)
   11ad0:	c05fe0ef          	jal	106d4 <__sfp_lock_release>
   11ad4:	00c12083          	lw	ra,12(sp)
   11ad8:	00812403          	lw	s0,8(sp)
   11adc:	00412483          	lw	s1,4(sp)
   11ae0:	00090513          	mv	a0,s2
   11ae4:	00012903          	lw	s2,0(sp)
   11ae8:	01010113          	addi	sp,sp,16
   11aec:	00008067          	ret
   11af0:	00c45783          	lhu	a5,12(s0)
   11af4:	fff00913          	li	s2,-1
   11af8:	0807f793          	andi	a5,a5,128
   11afc:	f8078ee3          	beqz	a5,11a98 <_fclose_r+0x88>
   11b00:	01042583          	lw	a1,16(s0)
   11b04:	00048513          	mv	a0,s1
   11b08:	c34ff0ef          	jal	10f3c <_free_r>
   11b0c:	f8dff06f          	j	11a98 <_fclose_r+0x88>
   11b10:	b9dfe0ef          	jal	106ac <__sinit>
   11b14:	f29ff06f          	j	11a3c <_fclose_r+0x2c>

00011b18 <fclose>:
   11b18:	00050593          	mv	a1,a0
   11b1c:	e0c1a503          	lw	a0,-500(gp) # 1370c <_impure_ptr>
   11b20:	ef1ff06f          	j	11a10 <_fclose_r>

00011b24 <__sflush_r>:
   11b24:	00c59703          	lh	a4,12(a1)
   11b28:	fe010113          	addi	sp,sp,-32
   11b2c:	00812c23          	sw	s0,24(sp)
   11b30:	01312623          	sw	s3,12(sp)
   11b34:	00112e23          	sw	ra,28(sp)
   11b38:	00877793          	andi	a5,a4,8
   11b3c:	00058413          	mv	s0,a1
   11b40:	00050993          	mv	s3,a0
   11b44:	12079063          	bnez	a5,11c64 <__sflush_r+0x140>
   11b48:	000017b7          	lui	a5,0x1
   11b4c:	80078793          	addi	a5,a5,-2048 # 800 <exit-0xf8b4>
   11b50:	0045a683          	lw	a3,4(a1)
   11b54:	00f767b3          	or	a5,a4,a5
   11b58:	00f59623          	sh	a5,12(a1)
   11b5c:	18d05263          	blez	a3,11ce0 <__sflush_r+0x1bc>
   11b60:	02842803          	lw	a6,40(s0)
   11b64:	0e080463          	beqz	a6,11c4c <__sflush_r+0x128>
   11b68:	00912a23          	sw	s1,20(sp)
   11b6c:	01371693          	slli	a3,a4,0x13
   11b70:	0009a483          	lw	s1,0(s3)
   11b74:	0009a023          	sw	zero,0(s3)
   11b78:	01c42583          	lw	a1,28(s0)
   11b7c:	1606ce63          	bltz	a3,11cf8 <__sflush_r+0x1d4>
   11b80:	00000613          	li	a2,0
   11b84:	00100693          	li	a3,1
   11b88:	00098513          	mv	a0,s3
   11b8c:	000800e7          	jalr	a6
   11b90:	fff00793          	li	a5,-1
   11b94:	00050613          	mv	a2,a0
   11b98:	1af50463          	beq	a0,a5,11d40 <__sflush_r+0x21c>
   11b9c:	00c41783          	lh	a5,12(s0)
   11ba0:	02842803          	lw	a6,40(s0)
   11ba4:	01c42583          	lw	a1,28(s0)
   11ba8:	0047f793          	andi	a5,a5,4
   11bac:	00078e63          	beqz	a5,11bc8 <__sflush_r+0xa4>
   11bb0:	00442703          	lw	a4,4(s0)
   11bb4:	03042783          	lw	a5,48(s0)
   11bb8:	40e60633          	sub	a2,a2,a4
   11bbc:	00078663          	beqz	a5,11bc8 <__sflush_r+0xa4>
   11bc0:	03c42783          	lw	a5,60(s0)
   11bc4:	40f60633          	sub	a2,a2,a5
   11bc8:	00000693          	li	a3,0
   11bcc:	00098513          	mv	a0,s3
   11bd0:	000800e7          	jalr	a6
   11bd4:	fff00793          	li	a5,-1
   11bd8:	12f51463          	bne	a0,a5,11d00 <__sflush_r+0x1dc>
   11bdc:	0009a683          	lw	a3,0(s3)
   11be0:	01d00793          	li	a5,29
   11be4:	00c41703          	lh	a4,12(s0)
   11be8:	16d7ea63          	bltu	a5,a3,11d5c <__sflush_r+0x238>
   11bec:	204007b7          	lui	a5,0x20400
   11bf0:	00178793          	addi	a5,a5,1 # 20400001 <__BSS_END__+0x203ec5c1>
   11bf4:	00d7d7b3          	srl	a5,a5,a3
   11bf8:	0017f793          	andi	a5,a5,1
   11bfc:	16078063          	beqz	a5,11d5c <__sflush_r+0x238>
   11c00:	01042603          	lw	a2,16(s0)
   11c04:	fffff7b7          	lui	a5,0xfffff
   11c08:	7ff78793          	addi	a5,a5,2047 # fffff7ff <__BSS_END__+0xfffebdbf>
   11c0c:	00f777b3          	and	a5,a4,a5
   11c10:	00f41623          	sh	a5,12(s0)
   11c14:	00042223          	sw	zero,4(s0)
   11c18:	00c42023          	sw	a2,0(s0)
   11c1c:	01371793          	slli	a5,a4,0x13
   11c20:	0007d463          	bgez	a5,11c28 <__sflush_r+0x104>
   11c24:	10068263          	beqz	a3,11d28 <__sflush_r+0x204>
   11c28:	03042583          	lw	a1,48(s0)
   11c2c:	0099a023          	sw	s1,0(s3)
   11c30:	10058463          	beqz	a1,11d38 <__sflush_r+0x214>
   11c34:	04040793          	addi	a5,s0,64
   11c38:	00f58663          	beq	a1,a5,11c44 <__sflush_r+0x120>
   11c3c:	00098513          	mv	a0,s3
   11c40:	afcff0ef          	jal	10f3c <_free_r>
   11c44:	01412483          	lw	s1,20(sp)
   11c48:	02042823          	sw	zero,48(s0)
   11c4c:	00000513          	li	a0,0
   11c50:	01c12083          	lw	ra,28(sp)
   11c54:	01812403          	lw	s0,24(sp)
   11c58:	00c12983          	lw	s3,12(sp)
   11c5c:	02010113          	addi	sp,sp,32
   11c60:	00008067          	ret
   11c64:	01212823          	sw	s2,16(sp)
   11c68:	0105a903          	lw	s2,16(a1)
   11c6c:	08090263          	beqz	s2,11cf0 <__sflush_r+0x1cc>
   11c70:	00912a23          	sw	s1,20(sp)
   11c74:	0005a483          	lw	s1,0(a1)
   11c78:	00377713          	andi	a4,a4,3
   11c7c:	0125a023          	sw	s2,0(a1)
   11c80:	412484b3          	sub	s1,s1,s2
   11c84:	00000793          	li	a5,0
   11c88:	00071463          	bnez	a4,11c90 <__sflush_r+0x16c>
   11c8c:	0145a783          	lw	a5,20(a1)
   11c90:	00f42423          	sw	a5,8(s0)
   11c94:	00904863          	bgtz	s1,11ca4 <__sflush_r+0x180>
   11c98:	0540006f          	j	11cec <__sflush_r+0x1c8>
   11c9c:	00a90933          	add	s2,s2,a0
   11ca0:	04905663          	blez	s1,11cec <__sflush_r+0x1c8>
   11ca4:	02442783          	lw	a5,36(s0)
   11ca8:	01c42583          	lw	a1,28(s0)
   11cac:	00048693          	mv	a3,s1
   11cb0:	00090613          	mv	a2,s2
   11cb4:	00098513          	mv	a0,s3
   11cb8:	000780e7          	jalr	a5
   11cbc:	40a484b3          	sub	s1,s1,a0
   11cc0:	fca04ee3          	bgtz	a0,11c9c <__sflush_r+0x178>
   11cc4:	00c41703          	lh	a4,12(s0)
   11cc8:	01012903          	lw	s2,16(sp)
   11ccc:	04076713          	ori	a4,a4,64
   11cd0:	01412483          	lw	s1,20(sp)
   11cd4:	00e41623          	sh	a4,12(s0)
   11cd8:	fff00513          	li	a0,-1
   11cdc:	f75ff06f          	j	11c50 <__sflush_r+0x12c>
   11ce0:	03c5a683          	lw	a3,60(a1)
   11ce4:	e6d04ee3          	bgtz	a3,11b60 <__sflush_r+0x3c>
   11ce8:	f65ff06f          	j	11c4c <__sflush_r+0x128>
   11cec:	01412483          	lw	s1,20(sp)
   11cf0:	01012903          	lw	s2,16(sp)
   11cf4:	f59ff06f          	j	11c4c <__sflush_r+0x128>
   11cf8:	05042603          	lw	a2,80(s0)
   11cfc:	eadff06f          	j	11ba8 <__sflush_r+0x84>
   11d00:	00c41703          	lh	a4,12(s0)
   11d04:	01042683          	lw	a3,16(s0)
   11d08:	fffff7b7          	lui	a5,0xfffff
   11d0c:	7ff78793          	addi	a5,a5,2047 # fffff7ff <__BSS_END__+0xfffebdbf>
   11d10:	00f777b3          	and	a5,a4,a5
   11d14:	00f41623          	sh	a5,12(s0)
   11d18:	00042223          	sw	zero,4(s0)
   11d1c:	00d42023          	sw	a3,0(s0)
   11d20:	01371793          	slli	a5,a4,0x13
   11d24:	f007d2e3          	bgez	a5,11c28 <__sflush_r+0x104>
   11d28:	03042583          	lw	a1,48(s0)
   11d2c:	04a42823          	sw	a0,80(s0)
   11d30:	0099a023          	sw	s1,0(s3)
   11d34:	f00590e3          	bnez	a1,11c34 <__sflush_r+0x110>
   11d38:	01412483          	lw	s1,20(sp)
   11d3c:	f11ff06f          	j	11c4c <__sflush_r+0x128>
   11d40:	0009a783          	lw	a5,0(s3)
   11d44:	e4078ce3          	beqz	a5,11b9c <__sflush_r+0x78>
   11d48:	01d00713          	li	a4,29
   11d4c:	00e78c63          	beq	a5,a4,11d64 <__sflush_r+0x240>
   11d50:	01600713          	li	a4,22
   11d54:	00e78863          	beq	a5,a4,11d64 <__sflush_r+0x240>
   11d58:	00c41703          	lh	a4,12(s0)
   11d5c:	04076713          	ori	a4,a4,64
   11d60:	f71ff06f          	j	11cd0 <__sflush_r+0x1ac>
   11d64:	0099a023          	sw	s1,0(s3)
   11d68:	01412483          	lw	s1,20(sp)
   11d6c:	ee1ff06f          	j	11c4c <__sflush_r+0x128>

00011d70 <_fflush_r>:
   11d70:	fe010113          	addi	sp,sp,-32
   11d74:	00812c23          	sw	s0,24(sp)
   11d78:	00112e23          	sw	ra,28(sp)
   11d7c:	00050413          	mv	s0,a0
   11d80:	00050663          	beqz	a0,11d8c <_fflush_r+0x1c>
   11d84:	03452783          	lw	a5,52(a0)
   11d88:	02078a63          	beqz	a5,11dbc <_fflush_r+0x4c>
   11d8c:	00c59783          	lh	a5,12(a1)
   11d90:	00079c63          	bnez	a5,11da8 <_fflush_r+0x38>
   11d94:	01c12083          	lw	ra,28(sp)
   11d98:	01812403          	lw	s0,24(sp)
   11d9c:	00000513          	li	a0,0
   11da0:	02010113          	addi	sp,sp,32
   11da4:	00008067          	ret
   11da8:	00040513          	mv	a0,s0
   11dac:	01812403          	lw	s0,24(sp)
   11db0:	01c12083          	lw	ra,28(sp)
   11db4:	02010113          	addi	sp,sp,32
   11db8:	d6dff06f          	j	11b24 <__sflush_r>
   11dbc:	00b12623          	sw	a1,12(sp)
   11dc0:	8edfe0ef          	jal	106ac <__sinit>
   11dc4:	00c12583          	lw	a1,12(sp)
   11dc8:	fc5ff06f          	j	11d8c <_fflush_r+0x1c>

00011dcc <fflush>:
   11dcc:	06050063          	beqz	a0,11e2c <fflush+0x60>
   11dd0:	00050593          	mv	a1,a0
   11dd4:	e0c1a503          	lw	a0,-500(gp) # 1370c <_impure_ptr>
   11dd8:	00050663          	beqz	a0,11de4 <fflush+0x18>
   11ddc:	03452783          	lw	a5,52(a0)
   11de0:	00078c63          	beqz	a5,11df8 <fflush+0x2c>
   11de4:	00c59783          	lh	a5,12(a1)
   11de8:	00079663          	bnez	a5,11df4 <fflush+0x28>
   11dec:	00000513          	li	a0,0
   11df0:	00008067          	ret
   11df4:	d31ff06f          	j	11b24 <__sflush_r>
   11df8:	fe010113          	addi	sp,sp,-32
   11dfc:	00b12623          	sw	a1,12(sp)
   11e00:	00a12423          	sw	a0,8(sp)
   11e04:	00112e23          	sw	ra,28(sp)
   11e08:	8a5fe0ef          	jal	106ac <__sinit>
   11e0c:	00c12583          	lw	a1,12(sp)
   11e10:	00812503          	lw	a0,8(sp)
   11e14:	00c59783          	lh	a5,12(a1)
   11e18:	02079663          	bnez	a5,11e44 <fflush+0x78>
   11e1c:	01c12083          	lw	ra,28(sp)
   11e20:	00000513          	li	a0,0
   11e24:	02010113          	addi	sp,sp,32
   11e28:	00008067          	ret
   11e2c:	8d018613          	addi	a2,gp,-1840 # 131d0 <__sglue>
   11e30:	00000597          	auipc	a1,0x0
   11e34:	f4058593          	addi	a1,a1,-192 # 11d70 <_fflush_r>
   11e38:	00001517          	auipc	a0,0x1
   11e3c:	3a850513          	addi	a0,a0,936 # 131e0 <_impure_data>
   11e40:	8c1fe06f          	j	10700 <_fwalk_sglue>
   11e44:	01c12083          	lw	ra,28(sp)
   11e48:	02010113          	addi	sp,sp,32
   11e4c:	cd9ff06f          	j	11b24 <__sflush_r>

00011e50 <_sbrk_r>:
   11e50:	ff010113          	addi	sp,sp,-16
   11e54:	00812423          	sw	s0,8(sp)
   11e58:	00050413          	mv	s0,a0
   11e5c:	00058513          	mv	a0,a1
   11e60:	e001ae23          	sw	zero,-484(gp) # 1371c <errno>
   11e64:	00112623          	sw	ra,12(sp)
   11e68:	15c000ef          	jal	11fc4 <_sbrk>
   11e6c:	fff00793          	li	a5,-1
   11e70:	00f50a63          	beq	a0,a5,11e84 <_sbrk_r+0x34>
   11e74:	00c12083          	lw	ra,12(sp)
   11e78:	00812403          	lw	s0,8(sp)
   11e7c:	01010113          	addi	sp,sp,16
   11e80:	00008067          	ret
   11e84:	e1c1a783          	lw	a5,-484(gp) # 1371c <errno>
   11e88:	fe0786e3          	beqz	a5,11e74 <_sbrk_r+0x24>
   11e8c:	00c12083          	lw	ra,12(sp)
   11e90:	00f42023          	sw	a5,0(s0)
   11e94:	00812403          	lw	s0,8(sp)
   11e98:	01010113          	addi	sp,sp,16
   11e9c:	00008067          	ret

00011ea0 <__libc_fini_array>:
   11ea0:	ff010113          	addi	sp,sp,-16
   11ea4:	00812423          	sw	s0,8(sp)
   11ea8:	00001797          	auipc	a5,0x1
   11eac:	20478793          	addi	a5,a5,516 # 130ac <__do_global_dtors_aux_fini_array_entry>
   11eb0:	00001417          	auipc	s0,0x1
   11eb4:	20040413          	addi	s0,s0,512 # 130b0 <__fini_array_end>
   11eb8:	40f40433          	sub	s0,s0,a5
   11ebc:	00912223          	sw	s1,4(sp)
   11ec0:	00112623          	sw	ra,12(sp)
   11ec4:	40245493          	srai	s1,s0,0x2
   11ec8:	02048063          	beqz	s1,11ee8 <__libc_fini_array+0x48>
   11ecc:	ffc40413          	addi	s0,s0,-4
   11ed0:	00f40433          	add	s0,s0,a5
   11ed4:	00042783          	lw	a5,0(s0)
   11ed8:	fff48493          	addi	s1,s1,-1
   11edc:	ffc40413          	addi	s0,s0,-4
   11ee0:	000780e7          	jalr	a5
   11ee4:	fe0498e3          	bnez	s1,11ed4 <__libc_fini_array+0x34>
   11ee8:	00c12083          	lw	ra,12(sp)
   11eec:	00812403          	lw	s0,8(sp)
   11ef0:	00412483          	lw	s1,4(sp)
   11ef4:	01010113          	addi	sp,sp,16
   11ef8:	00008067          	ret

00011efc <__register_exitproc>:
   11efc:	e2018713          	addi	a4,gp,-480 # 13720 <__atexit>
   11f00:	00072783          	lw	a5,0(a4)
   11f04:	04078c63          	beqz	a5,11f5c <__register_exitproc+0x60>
   11f08:	0047a703          	lw	a4,4(a5)
   11f0c:	01f00813          	li	a6,31
   11f10:	06e84e63          	blt	a6,a4,11f8c <__register_exitproc+0x90>
   11f14:	00271813          	slli	a6,a4,0x2
   11f18:	02050663          	beqz	a0,11f44 <__register_exitproc+0x48>
   11f1c:	01078333          	add	t1,a5,a6
   11f20:	08c32423          	sw	a2,136(t1)
   11f24:	1887a883          	lw	a7,392(a5)
   11f28:	00100613          	li	a2,1
   11f2c:	00e61633          	sll	a2,a2,a4
   11f30:	00c8e8b3          	or	a7,a7,a2
   11f34:	1917a423          	sw	a7,392(a5)
   11f38:	10d32423          	sw	a3,264(t1)
   11f3c:	00200693          	li	a3,2
   11f40:	02d50463          	beq	a0,a3,11f68 <__register_exitproc+0x6c>
   11f44:	00170713          	addi	a4,a4,1
   11f48:	00e7a223          	sw	a4,4(a5)
   11f4c:	010787b3          	add	a5,a5,a6
   11f50:	00b7a423          	sw	a1,8(a5)
   11f54:	00000513          	li	a0,0
   11f58:	00008067          	ret
   11f5c:	fb018793          	addi	a5,gp,-80 # 138b0 <__atexit0>
   11f60:	00f72023          	sw	a5,0(a4)
   11f64:	fa5ff06f          	j	11f08 <__register_exitproc+0xc>
   11f68:	18c7a683          	lw	a3,396(a5)
   11f6c:	00170713          	addi	a4,a4,1
   11f70:	00e7a223          	sw	a4,4(a5)
   11f74:	00c6e6b3          	or	a3,a3,a2
   11f78:	18d7a623          	sw	a3,396(a5)
   11f7c:	010787b3          	add	a5,a5,a6
   11f80:	00b7a423          	sw	a1,8(a5)
   11f84:	00000513          	li	a0,0
   11f88:	00008067          	ret
   11f8c:	fff00513          	li	a0,-1
   11f90:	00008067          	ret

00011f94 <_close>:
   11f94:	05800793          	li	a5,88
   11f98:	e0f1ae23          	sw	a5,-484(gp) # 1371c <errno>
   11f9c:	fff00513          	li	a0,-1
   11fa0:	00008067          	ret

00011fa4 <_lseek>:
   11fa4:	05800793          	li	a5,88
   11fa8:	e0f1ae23          	sw	a5,-484(gp) # 1371c <errno>
   11fac:	fff00513          	li	a0,-1
   11fb0:	00008067          	ret

00011fb4 <_read>:
   11fb4:	05800793          	li	a5,88
   11fb8:	e0f1ae23          	sw	a5,-484(gp) # 1371c <errno>
   11fbc:	fff00513          	li	a0,-1
   11fc0:	00008067          	ret

00011fc4 <_sbrk>:
   11fc4:	e3018713          	addi	a4,gp,-464 # 13730 <heap_end.0>
   11fc8:	00072783          	lw	a5,0(a4)
   11fcc:	00078a63          	beqz	a5,11fe0 <_sbrk+0x1c>
   11fd0:	00a78533          	add	a0,a5,a0
   11fd4:	00a72023          	sw	a0,0(a4)
   11fd8:	00078513          	mv	a0,a5
   11fdc:	00008067          	ret
   11fe0:	14018793          	addi	a5,gp,320 # 13a40 <__BSS_END__>
   11fe4:	00a78533          	add	a0,a5,a0
   11fe8:	00a72023          	sw	a0,0(a4)
   11fec:	00078513          	mv	a0,a5
   11ff0:	00008067          	ret

00011ff4 <_write>:
   11ff4:	05800793          	li	a5,88
   11ff8:	e0f1ae23          	sw	a5,-484(gp) # 1371c <errno>
   11ffc:	fff00513          	li	a0,-1
   12000:	00008067          	ret

00012004 <_exit>:
   12004:	0000006f          	j	12004 <_exit>

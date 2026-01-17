
/home/cloud/aps-mlir/examples/diff_match/horner3/horner3.out:     file format elf32-littleriscv


Disassembly of section .text:

000100b4 <exit>:
   100b4:	ff010113          	addi	sp,sp,-16
   100b8:	00000593          	li	a1,0
   100bc:	00812423          	sw	s0,8(sp)
   100c0:	00112623          	sw	ra,12(sp)
   100c4:	00050413          	mv	s0,a0
   100c8:	3dd000ef          	jal	10ca4 <__call_exitprocs>
   100cc:	d981a783          	lw	a5,-616(gp) # 13698 <__stdio_exit_handler>
   100d0:	00078463          	beqz	a5,100d8 <exit+0x24>
   100d4:	000780e7          	jalr	a5
   100d8:	00040513          	mv	a0,s0
   100dc:	719010ef          	jal	11ff4 <_exit>

000100e0 <register_fini>:
   100e0:	00000793          	li	a5,0
   100e4:	00078863          	beqz	a5,100f4 <register_fini+0x14>
   100e8:	00002517          	auipc	a0,0x2
   100ec:	da850513          	addi	a0,a0,-600 # 11e90 <__libc_fini_array>
   100f0:	4ed0006f          	j	10ddc <atexit>
   100f4:	00008067          	ret

000100f8 <_start>:
   100f8:	00004197          	auipc	gp,0x4
   100fc:	80818193          	addi	gp,gp,-2040 # 13900 <__global_pointer$>
   10100:	d9818513          	addi	a0,gp,-616 # 13698 <__stdio_exit_handler>
   10104:	0c018613          	addi	a2,gp,192 # 139c0 <__BSS_END__>
   10108:	40a60633          	sub	a2,a2,a0
   1010c:	00000593          	li	a1,0
   10110:	2b9000ef          	jal	10bc8 <memset>
   10114:	00001517          	auipc	a0,0x1
   10118:	cc850513          	addi	a0,a0,-824 # 10ddc <atexit>
   1011c:	00050863          	beqz	a0,1012c <_start+0x34>
   10120:	00002517          	auipc	a0,0x2
   10124:	d7050513          	addi	a0,a0,-656 # 11e90 <__libc_fini_array>
   10128:	4b5000ef          	jal	10ddc <atexit>
   1012c:	209000ef          	jal	10b34 <__libc_init_array>
   10130:	00012503          	lw	a0,0(sp)
   10134:	00410593          	addi	a1,sp,4
   10138:	00000613          	li	a2,0
   1013c:	0a8000ef          	jal	101e4 <main>
   10140:	f75ff06f          	j	100b4 <exit>

00010144 <__do_global_dtors_aux>:
   10144:	ff010113          	addi	sp,sp,-16
   10148:	00812423          	sw	s0,8(sp)
   1014c:	db418413          	addi	s0,gp,-588 # 136b4 <completed.1>
   10150:	00044783          	lbu	a5,0(s0)
   10154:	00112623          	sw	ra,12(sp)
   10158:	02079263          	bnez	a5,1017c <__do_global_dtors_aux+0x38>
   1015c:	00000793          	li	a5,0
   10160:	00078a63          	beqz	a5,10174 <__do_global_dtors_aux+0x30>
   10164:	00003517          	auipc	a0,0x3
   10168:	ec850513          	addi	a0,a0,-312 # 1302c <__EH_FRAME_BEGIN__>
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
   10194:	db818593          	addi	a1,gp,-584 # 136b8 <object.0>
   10198:	00003517          	auipc	a0,0x3
   1019c:	e9450513          	addi	a0,a0,-364 # 1302c <__EH_FRAME_BEGIN__>
   101a0:	00000317          	auipc	t1,0x0
   101a4:	00000067          	jr	zero # 0 <exit-0x100b4>
   101a8:	00008067          	ret

000101ac <horner3>:
   101ac:	52a5752b          	.insn	4, 0x52a5752b
   101b0:	00008067          	ret

000101b4 <get_march>:
   101b4:	fff50513          	addi	a0,a0,-1
   101b8:	00400593          	li	a1,4
   101bc:	00a5ee63          	bltu	a1,a0,101d8 <get_march+0x24>
   101c0:	00251513          	slli	a0,a0,0x2
   101c4:	00002597          	auipc	a1,0x2
   101c8:	e5458593          	addi	a1,a1,-428 # 12018 <_exit+0x24>
   101cc:	00a58533          	add	a0,a1,a0
   101d0:	00052503          	lw	a0,0(a0)
   101d4:	00008067          	ret
   101d8:	00002517          	auipc	a0,0x2
   101dc:	e3750513          	addi	a0,a0,-457 # 1200f <_exit+0x1b>
   101e0:	00008067          	ret

000101e4 <main>:
   101e4:	ff010113          	addi	sp,sp,-16
   101e8:	00112623          	sw	ra,12(sp)
   101ec:	00812423          	sw	s0,8(sp)
   101f0:	00912223          	sw	s1,4(sp)
   101f4:	01212023          	sw	s2,0(sp)
   101f8:	f1202573          	.insn	4, 0xf1202573
   101fc:	0ff0000f          	fence
   10200:	00003417          	auipc	s0,0x3
   10204:	f0040413          	addi	s0,s0,-256 # 13100 <test_inputs>
   10208:	00042503          	lw	a0,0(s0)
   1020c:	fa1ff0ef          	jal	101ac <horner3>
   10210:	0ff0000f          	fence
   10214:	02842603          	lw	a2,40(s0)
   10218:	0ff0000f          	fence
   1021c:	00442583          	lw	a1,4(s0)
   10220:	00c54533          	xor	a0,a0,a2
   10224:	00153493          	seqz	s1,a0
   10228:	00058513          	mv	a0,a1
   1022c:	f81ff0ef          	jal	101ac <horner3>
   10230:	0ff0000f          	fence
   10234:	02c42603          	lw	a2,44(s0)
   10238:	0ff0000f          	fence
   1023c:	00842583          	lw	a1,8(s0)
   10240:	00c54533          	xor	a0,a0,a2
   10244:	00153513          	seqz	a0,a0
   10248:	00a484b3          	add	s1,s1,a0
   1024c:	00058513          	mv	a0,a1
   10250:	f5dff0ef          	jal	101ac <horner3>
   10254:	0ff0000f          	fence
   10258:	03042603          	lw	a2,48(s0)
   1025c:	0ff0000f          	fence
   10260:	00c42583          	lw	a1,12(s0)
   10264:	00c54533          	xor	a0,a0,a2
   10268:	00153913          	seqz	s2,a0
   1026c:	00058513          	mv	a0,a1
   10270:	f3dff0ef          	jal	101ac <horner3>
   10274:	0ff0000f          	fence
   10278:	03442603          	lw	a2,52(s0)
   1027c:	0ff0000f          	fence
   10280:	01042583          	lw	a1,16(s0)
   10284:	00c54533          	xor	a0,a0,a2
   10288:	00153513          	seqz	a0,a0
   1028c:	00a90533          	add	a0,s2,a0
   10290:	00a484b3          	add	s1,s1,a0
   10294:	00058513          	mv	a0,a1
   10298:	f15ff0ef          	jal	101ac <horner3>
   1029c:	0ff0000f          	fence
   102a0:	03842603          	lw	a2,56(s0)
   102a4:	0ff0000f          	fence
   102a8:	01442583          	lw	a1,20(s0)
   102ac:	00c54533          	xor	a0,a0,a2
   102b0:	00153913          	seqz	s2,a0
   102b4:	00058513          	mv	a0,a1
   102b8:	ef5ff0ef          	jal	101ac <horner3>
   102bc:	0ff0000f          	fence
   102c0:	03c42603          	lw	a2,60(s0)
   102c4:	0ff0000f          	fence
   102c8:	01842583          	lw	a1,24(s0)
   102cc:	00c54533          	xor	a0,a0,a2
   102d0:	00153513          	seqz	a0,a0
   102d4:	00a90933          	add	s2,s2,a0
   102d8:	00058513          	mv	a0,a1
   102dc:	ed1ff0ef          	jal	101ac <horner3>
   102e0:	0ff0000f          	fence
   102e4:	04042603          	lw	a2,64(s0)
   102e8:	0ff0000f          	fence
   102ec:	01c42583          	lw	a1,28(s0)
   102f0:	00c54533          	xor	a0,a0,a2
   102f4:	00153513          	seqz	a0,a0
   102f8:	00a90533          	add	a0,s2,a0
   102fc:	00a484b3          	add	s1,s1,a0
   10300:	00058513          	mv	a0,a1
   10304:	ea9ff0ef          	jal	101ac <horner3>
   10308:	0ff0000f          	fence
   1030c:	04442603          	lw	a2,68(s0)
   10310:	0ff0000f          	fence
   10314:	02042583          	lw	a1,32(s0)
   10318:	00c54533          	xor	a0,a0,a2
   1031c:	00153913          	seqz	s2,a0
   10320:	00058513          	mv	a0,a1
   10324:	e89ff0ef          	jal	101ac <horner3>
   10328:	0ff0000f          	fence
   1032c:	04842603          	lw	a2,72(s0)
   10330:	0ff0000f          	fence
   10334:	02442583          	lw	a1,36(s0)
   10338:	00c54533          	xor	a0,a0,a2
   1033c:	00153513          	seqz	a0,a0
   10340:	00a90933          	add	s2,s2,a0
   10344:	00058513          	mv	a0,a1
   10348:	e65ff0ef          	jal	101ac <horner3>
   1034c:	0ff0000f          	fence
   10350:	04c42583          	lw	a1,76(s0)
   10354:	00b54533          	xor	a0,a0,a1
   10358:	00153513          	seqz	a0,a0
   1035c:	00a90533          	add	a0,s2,a0
   10360:	00a48533          	add	a0,s1,a0
   10364:	ff650513          	addi	a0,a0,-10
   10368:	00a03533          	snez	a0,a0
   1036c:	00c12083          	lw	ra,12(sp)
   10370:	00812403          	lw	s0,8(sp)
   10374:	00412483          	lw	s1,4(sp)
   10378:	00012903          	lw	s2,0(sp)
   1037c:	01010113          	addi	sp,sp,16
   10380:	00008067          	ret

00010384 <__fp_lock>:
   10384:	00000513          	li	a0,0
   10388:	00008067          	ret

0001038c <stdio_exit_handler>:
   1038c:	00003617          	auipc	a2,0x3
   10390:	dc460613          	addi	a2,a2,-572 # 13150 <__sglue>
   10394:	00001597          	auipc	a1,0x1
   10398:	66858593          	addi	a1,a1,1640 # 119fc <_fclose_r>
   1039c:	00003517          	auipc	a0,0x3
   103a0:	dc450513          	addi	a0,a0,-572 # 13160 <_impure_data>
   103a4:	3480006f          	j	106ec <_fwalk_sglue>

000103a8 <cleanup_stdio>:
   103a8:	00452583          	lw	a1,4(a0)
   103ac:	ff010113          	addi	sp,sp,-16
   103b0:	00812423          	sw	s0,8(sp)
   103b4:	00112623          	sw	ra,12(sp)
   103b8:	dd018793          	addi	a5,gp,-560 # 136d0 <__sf>
   103bc:	00050413          	mv	s0,a0
   103c0:	00f58463          	beq	a1,a5,103c8 <cleanup_stdio+0x20>
   103c4:	638010ef          	jal	119fc <_fclose_r>
   103c8:	00842583          	lw	a1,8(s0)
   103cc:	e3818793          	addi	a5,gp,-456 # 13738 <__sf+0x68>
   103d0:	00f58663          	beq	a1,a5,103dc <cleanup_stdio+0x34>
   103d4:	00040513          	mv	a0,s0
   103d8:	624010ef          	jal	119fc <_fclose_r>
   103dc:	00c42583          	lw	a1,12(s0)
   103e0:	ea018793          	addi	a5,gp,-352 # 137a0 <__sf+0xd0>
   103e4:	00f58c63          	beq	a1,a5,103fc <cleanup_stdio+0x54>
   103e8:	00040513          	mv	a0,s0
   103ec:	00812403          	lw	s0,8(sp)
   103f0:	00c12083          	lw	ra,12(sp)
   103f4:	01010113          	addi	sp,sp,16
   103f8:	6040106f          	j	119fc <_fclose_r>
   103fc:	00c12083          	lw	ra,12(sp)
   10400:	00812403          	lw	s0,8(sp)
   10404:	01010113          	addi	sp,sp,16
   10408:	00008067          	ret

0001040c <__fp_unlock>:
   1040c:	00000513          	li	a0,0
   10410:	00008067          	ret

00010414 <global_stdio_init.part.0>:
   10414:	fe010113          	addi	sp,sp,-32
   10418:	00000797          	auipc	a5,0x0
   1041c:	f7478793          	addi	a5,a5,-140 # 1038c <stdio_exit_handler>
   10420:	00112e23          	sw	ra,28(sp)
   10424:	00812c23          	sw	s0,24(sp)
   10428:	00912a23          	sw	s1,20(sp)
   1042c:	dd018413          	addi	s0,gp,-560 # 136d0 <__sf>
   10430:	01212823          	sw	s2,16(sp)
   10434:	01312623          	sw	s3,12(sp)
   10438:	01412423          	sw	s4,8(sp)
   1043c:	d8f1ac23          	sw	a5,-616(gp) # 13698 <__stdio_exit_handler>
   10440:	00800613          	li	a2,8
   10444:	00400793          	li	a5,4
   10448:	00000593          	li	a1,0
   1044c:	e2c18513          	addi	a0,gp,-468 # 1372c <__sf+0x5c>
   10450:	00f42623          	sw	a5,12(s0)
   10454:	00042023          	sw	zero,0(s0)
   10458:	00042223          	sw	zero,4(s0)
   1045c:	00042423          	sw	zero,8(s0)
   10460:	06042223          	sw	zero,100(s0)
   10464:	00042823          	sw	zero,16(s0)
   10468:	00042a23          	sw	zero,20(s0)
   1046c:	00042c23          	sw	zero,24(s0)
   10470:	758000ef          	jal	10bc8 <memset>
   10474:	000107b7          	lui	a5,0x10
   10478:	00000a17          	auipc	s4,0x0
   1047c:	328a0a13          	addi	s4,s4,808 # 107a0 <__sread>
   10480:	00000997          	auipc	s3,0x0
   10484:	38498993          	addi	s3,s3,900 # 10804 <__swrite>
   10488:	00000917          	auipc	s2,0x0
   1048c:	40490913          	addi	s2,s2,1028 # 1088c <__sseek>
   10490:	00000497          	auipc	s1,0x0
   10494:	47448493          	addi	s1,s1,1140 # 10904 <__sclose>
   10498:	00978793          	addi	a5,a5,9 # 10009 <exit-0xab>
   1049c:	00800613          	li	a2,8
   104a0:	00000593          	li	a1,0
   104a4:	e9418513          	addi	a0,gp,-364 # 13794 <__sf+0xc4>
   104a8:	03442023          	sw	s4,32(s0)
   104ac:	03342223          	sw	s3,36(s0)
   104b0:	03242423          	sw	s2,40(s0)
   104b4:	02942623          	sw	s1,44(s0)
   104b8:	06f42a23          	sw	a5,116(s0)
   104bc:	00842e23          	sw	s0,28(s0)
   104c0:	06042423          	sw	zero,104(s0)
   104c4:	06042623          	sw	zero,108(s0)
   104c8:	06042823          	sw	zero,112(s0)
   104cc:	0c042623          	sw	zero,204(s0)
   104d0:	06042c23          	sw	zero,120(s0)
   104d4:	06042e23          	sw	zero,124(s0)
   104d8:	08042023          	sw	zero,128(s0)
   104dc:	6ec000ef          	jal	10bc8 <memset>
   104e0:	000207b7          	lui	a5,0x20
   104e4:	01278793          	addi	a5,a5,18 # 20012 <__BSS_END__+0xc652>
   104e8:	e3818713          	addi	a4,gp,-456 # 13738 <__sf+0x68>
   104ec:	00800613          	li	a2,8
   104f0:	00000593          	li	a1,0
   104f4:	efc18513          	addi	a0,gp,-260 # 137fc <__sf+0x12c>
   104f8:	09442423          	sw	s4,136(s0)
   104fc:	09342623          	sw	s3,140(s0)
   10500:	09242823          	sw	s2,144(s0)
   10504:	08942a23          	sw	s1,148(s0)
   10508:	0cf42e23          	sw	a5,220(s0)
   1050c:	08e42223          	sw	a4,132(s0)
   10510:	0c042823          	sw	zero,208(s0)
   10514:	0c042a23          	sw	zero,212(s0)
   10518:	0c042c23          	sw	zero,216(s0)
   1051c:	12042a23          	sw	zero,308(s0)
   10520:	0e042023          	sw	zero,224(s0)
   10524:	0e042223          	sw	zero,228(s0)
   10528:	0e042423          	sw	zero,232(s0)
   1052c:	69c000ef          	jal	10bc8 <memset>
   10530:	ea018793          	addi	a5,gp,-352 # 137a0 <__sf+0xd0>
   10534:	0f442823          	sw	s4,240(s0)
   10538:	0f342a23          	sw	s3,244(s0)
   1053c:	0f242c23          	sw	s2,248(s0)
   10540:	0e942e23          	sw	s1,252(s0)
   10544:	01c12083          	lw	ra,28(sp)
   10548:	0ef42623          	sw	a5,236(s0)
   1054c:	01812403          	lw	s0,24(sp)
   10550:	01412483          	lw	s1,20(sp)
   10554:	01012903          	lw	s2,16(sp)
   10558:	00c12983          	lw	s3,12(sp)
   1055c:	00812a03          	lw	s4,8(sp)
   10560:	02010113          	addi	sp,sp,32
   10564:	00008067          	ret

00010568 <__sfp>:
   10568:	fe010113          	addi	sp,sp,-32
   1056c:	01312623          	sw	s3,12(sp)
   10570:	00112e23          	sw	ra,28(sp)
   10574:	00812c23          	sw	s0,24(sp)
   10578:	00912a23          	sw	s1,20(sp)
   1057c:	01212823          	sw	s2,16(sp)
   10580:	d981a783          	lw	a5,-616(gp) # 13698 <__stdio_exit_handler>
   10584:	00050993          	mv	s3,a0
   10588:	0e078863          	beqz	a5,10678 <__sfp+0x110>
   1058c:	00003917          	auipc	s2,0x3
   10590:	bc490913          	addi	s2,s2,-1084 # 13150 <__sglue>
   10594:	fff00493          	li	s1,-1
   10598:	00492783          	lw	a5,4(s2)
   1059c:	00892403          	lw	s0,8(s2)
   105a0:	fff78793          	addi	a5,a5,-1
   105a4:	0007d863          	bgez	a5,105b4 <__sfp+0x4c>
   105a8:	0800006f          	j	10628 <__sfp+0xc0>
   105ac:	06840413          	addi	s0,s0,104
   105b0:	06978c63          	beq	a5,s1,10628 <__sfp+0xc0>
   105b4:	00c41703          	lh	a4,12(s0)
   105b8:	fff78793          	addi	a5,a5,-1
   105bc:	fe0718e3          	bnez	a4,105ac <__sfp+0x44>
   105c0:	ffff07b7          	lui	a5,0xffff0
   105c4:	00178793          	addi	a5,a5,1 # ffff0001 <__BSS_END__+0xfffdc641>
   105c8:	00f42623          	sw	a5,12(s0)
   105cc:	06042223          	sw	zero,100(s0)
   105d0:	00042023          	sw	zero,0(s0)
   105d4:	00042423          	sw	zero,8(s0)
   105d8:	00042223          	sw	zero,4(s0)
   105dc:	00042823          	sw	zero,16(s0)
   105e0:	00042a23          	sw	zero,20(s0)
   105e4:	00042c23          	sw	zero,24(s0)
   105e8:	00800613          	li	a2,8
   105ec:	00000593          	li	a1,0
   105f0:	05c40513          	addi	a0,s0,92
   105f4:	5d4000ef          	jal	10bc8 <memset>
   105f8:	02042823          	sw	zero,48(s0)
   105fc:	02042a23          	sw	zero,52(s0)
   10600:	04042223          	sw	zero,68(s0)
   10604:	04042423          	sw	zero,72(s0)
   10608:	01c12083          	lw	ra,28(sp)
   1060c:	00040513          	mv	a0,s0
   10610:	01812403          	lw	s0,24(sp)
   10614:	01412483          	lw	s1,20(sp)
   10618:	01012903          	lw	s2,16(sp)
   1061c:	00c12983          	lw	s3,12(sp)
   10620:	02010113          	addi	sp,sp,32
   10624:	00008067          	ret
   10628:	00092403          	lw	s0,0(s2)
   1062c:	00040663          	beqz	s0,10638 <__sfp+0xd0>
   10630:	00040913          	mv	s2,s0
   10634:	f65ff06f          	j	10598 <__sfp+0x30>
   10638:	1ac00593          	li	a1,428
   1063c:	00098513          	mv	a0,s3
   10640:	3ed000ef          	jal	1122c <_malloc_r>
   10644:	00050413          	mv	s0,a0
   10648:	02050c63          	beqz	a0,10680 <__sfp+0x118>
   1064c:	00c50513          	addi	a0,a0,12
   10650:	00400793          	li	a5,4
   10654:	00042023          	sw	zero,0(s0)
   10658:	00f42223          	sw	a5,4(s0)
   1065c:	00a42423          	sw	a0,8(s0)
   10660:	1a000613          	li	a2,416
   10664:	00000593          	li	a1,0
   10668:	560000ef          	jal	10bc8 <memset>
   1066c:	00892023          	sw	s0,0(s2)
   10670:	00040913          	mv	s2,s0
   10674:	f25ff06f          	j	10598 <__sfp+0x30>
   10678:	d9dff0ef          	jal	10414 <global_stdio_init.part.0>
   1067c:	f11ff06f          	j	1058c <__sfp+0x24>
   10680:	00092023          	sw	zero,0(s2)
   10684:	00c00793          	li	a5,12
   10688:	00f9a023          	sw	a5,0(s3)
   1068c:	f7dff06f          	j	10608 <__sfp+0xa0>

00010690 <__sinit>:
   10690:	03452783          	lw	a5,52(a0)
   10694:	00078463          	beqz	a5,1069c <__sinit+0xc>
   10698:	00008067          	ret
   1069c:	00000797          	auipc	a5,0x0
   106a0:	d0c78793          	addi	a5,a5,-756 # 103a8 <cleanup_stdio>
   106a4:	02f52a23          	sw	a5,52(a0)
   106a8:	d981a783          	lw	a5,-616(gp) # 13698 <__stdio_exit_handler>
   106ac:	fe0796e3          	bnez	a5,10698 <__sinit+0x8>
   106b0:	d65ff06f          	j	10414 <global_stdio_init.part.0>

000106b4 <__sfp_lock_acquire>:
   106b4:	00008067          	ret

000106b8 <__sfp_lock_release>:
   106b8:	00008067          	ret

000106bc <__fp_lock_all>:
   106bc:	00003617          	auipc	a2,0x3
   106c0:	a9460613          	addi	a2,a2,-1388 # 13150 <__sglue>
   106c4:	00000597          	auipc	a1,0x0
   106c8:	cc058593          	addi	a1,a1,-832 # 10384 <__fp_lock>
   106cc:	00000513          	li	a0,0
   106d0:	01c0006f          	j	106ec <_fwalk_sglue>

000106d4 <__fp_unlock_all>:
   106d4:	00003617          	auipc	a2,0x3
   106d8:	a7c60613          	addi	a2,a2,-1412 # 13150 <__sglue>
   106dc:	00000597          	auipc	a1,0x0
   106e0:	d3058593          	addi	a1,a1,-720 # 1040c <__fp_unlock>
   106e4:	00000513          	li	a0,0
   106e8:	0040006f          	j	106ec <_fwalk_sglue>

000106ec <_fwalk_sglue>:
   106ec:	fd010113          	addi	sp,sp,-48
   106f0:	03212023          	sw	s2,32(sp)
   106f4:	01312e23          	sw	s3,28(sp)
   106f8:	01412c23          	sw	s4,24(sp)
   106fc:	01512a23          	sw	s5,20(sp)
   10700:	01612823          	sw	s6,16(sp)
   10704:	01712623          	sw	s7,12(sp)
   10708:	02112623          	sw	ra,44(sp)
   1070c:	02812423          	sw	s0,40(sp)
   10710:	02912223          	sw	s1,36(sp)
   10714:	00050b13          	mv	s6,a0
   10718:	00058b93          	mv	s7,a1
   1071c:	00060a93          	mv	s5,a2
   10720:	00000a13          	li	s4,0
   10724:	00100993          	li	s3,1
   10728:	fff00913          	li	s2,-1
   1072c:	004aa483          	lw	s1,4(s5)
   10730:	008aa403          	lw	s0,8(s5)
   10734:	fff48493          	addi	s1,s1,-1
   10738:	0204c863          	bltz	s1,10768 <_fwalk_sglue+0x7c>
   1073c:	00c45783          	lhu	a5,12(s0)
   10740:	00f9fe63          	bgeu	s3,a5,1075c <_fwalk_sglue+0x70>
   10744:	00e41783          	lh	a5,14(s0)
   10748:	00040593          	mv	a1,s0
   1074c:	000b0513          	mv	a0,s6
   10750:	01278663          	beq	a5,s2,1075c <_fwalk_sglue+0x70>
   10754:	000b80e7          	jalr	s7
   10758:	00aa6a33          	or	s4,s4,a0
   1075c:	fff48493          	addi	s1,s1,-1
   10760:	06840413          	addi	s0,s0,104
   10764:	fd249ce3          	bne	s1,s2,1073c <_fwalk_sglue+0x50>
   10768:	000aaa83          	lw	s5,0(s5)
   1076c:	fc0a90e3          	bnez	s5,1072c <_fwalk_sglue+0x40>
   10770:	02c12083          	lw	ra,44(sp)
   10774:	02812403          	lw	s0,40(sp)
   10778:	02412483          	lw	s1,36(sp)
   1077c:	02012903          	lw	s2,32(sp)
   10780:	01c12983          	lw	s3,28(sp)
   10784:	01412a83          	lw	s5,20(sp)
   10788:	01012b03          	lw	s6,16(sp)
   1078c:	00c12b83          	lw	s7,12(sp)
   10790:	000a0513          	mv	a0,s4
   10794:	01812a03          	lw	s4,24(sp)
   10798:	03010113          	addi	sp,sp,48
   1079c:	00008067          	ret

000107a0 <__sread>:
   107a0:	ff010113          	addi	sp,sp,-16
   107a4:	00812423          	sw	s0,8(sp)
   107a8:	00058413          	mv	s0,a1
   107ac:	00e59583          	lh	a1,14(a1)
   107b0:	00112623          	sw	ra,12(sp)
   107b4:	2c8000ef          	jal	10a7c <_read_r>
   107b8:	02054063          	bltz	a0,107d8 <__sread+0x38>
   107bc:	05042783          	lw	a5,80(s0)
   107c0:	00c12083          	lw	ra,12(sp)
   107c4:	00a787b3          	add	a5,a5,a0
   107c8:	04f42823          	sw	a5,80(s0)
   107cc:	00812403          	lw	s0,8(sp)
   107d0:	01010113          	addi	sp,sp,16
   107d4:	00008067          	ret
   107d8:	00c45783          	lhu	a5,12(s0)
   107dc:	fffff737          	lui	a4,0xfffff
   107e0:	fff70713          	addi	a4,a4,-1 # ffffefff <__BSS_END__+0xfffeb63f>
   107e4:	00e7f7b3          	and	a5,a5,a4
   107e8:	00c12083          	lw	ra,12(sp)
   107ec:	00f41623          	sh	a5,12(s0)
   107f0:	00812403          	lw	s0,8(sp)
   107f4:	01010113          	addi	sp,sp,16
   107f8:	00008067          	ret

000107fc <__seofread>:
   107fc:	00000513          	li	a0,0
   10800:	00008067          	ret

00010804 <__swrite>:
   10804:	00c59783          	lh	a5,12(a1)
   10808:	fe010113          	addi	sp,sp,-32
   1080c:	00812c23          	sw	s0,24(sp)
   10810:	00912a23          	sw	s1,20(sp)
   10814:	01212823          	sw	s2,16(sp)
   10818:	01312623          	sw	s3,12(sp)
   1081c:	00112e23          	sw	ra,28(sp)
   10820:	1007f713          	andi	a4,a5,256
   10824:	00058413          	mv	s0,a1
   10828:	00050493          	mv	s1,a0
   1082c:	00060913          	mv	s2,a2
   10830:	00068993          	mv	s3,a3
   10834:	04071063          	bnez	a4,10874 <__swrite+0x70>
   10838:	fffff737          	lui	a4,0xfffff
   1083c:	fff70713          	addi	a4,a4,-1 # ffffefff <__BSS_END__+0xfffeb63f>
   10840:	00e7f7b3          	and	a5,a5,a4
   10844:	00e41583          	lh	a1,14(s0)
   10848:	00f41623          	sh	a5,12(s0)
   1084c:	01812403          	lw	s0,24(sp)
   10850:	01c12083          	lw	ra,28(sp)
   10854:	00098693          	mv	a3,s3
   10858:	00090613          	mv	a2,s2
   1085c:	00c12983          	lw	s3,12(sp)
   10860:	01012903          	lw	s2,16(sp)
   10864:	00048513          	mv	a0,s1
   10868:	01412483          	lw	s1,20(sp)
   1086c:	02010113          	addi	sp,sp,32
   10870:	2680006f          	j	10ad8 <_write_r>
   10874:	00e59583          	lh	a1,14(a1)
   10878:	00200693          	li	a3,2
   1087c:	00000613          	li	a2,0
   10880:	1a0000ef          	jal	10a20 <_lseek_r>
   10884:	00c41783          	lh	a5,12(s0)
   10888:	fb1ff06f          	j	10838 <__swrite+0x34>

0001088c <__sseek>:
   1088c:	ff010113          	addi	sp,sp,-16
   10890:	00812423          	sw	s0,8(sp)
   10894:	00058413          	mv	s0,a1
   10898:	00e59583          	lh	a1,14(a1)
   1089c:	00112623          	sw	ra,12(sp)
   108a0:	180000ef          	jal	10a20 <_lseek_r>
   108a4:	fff00793          	li	a5,-1
   108a8:	02f50863          	beq	a0,a5,108d8 <__sseek+0x4c>
   108ac:	00c45783          	lhu	a5,12(s0)
   108b0:	00001737          	lui	a4,0x1
   108b4:	00c12083          	lw	ra,12(sp)
   108b8:	00e7e7b3          	or	a5,a5,a4
   108bc:	01079793          	slli	a5,a5,0x10
   108c0:	4107d793          	srai	a5,a5,0x10
   108c4:	04a42823          	sw	a0,80(s0)
   108c8:	00f41623          	sh	a5,12(s0)
   108cc:	00812403          	lw	s0,8(sp)
   108d0:	01010113          	addi	sp,sp,16
   108d4:	00008067          	ret
   108d8:	00c45783          	lhu	a5,12(s0)
   108dc:	fffff737          	lui	a4,0xfffff
   108e0:	fff70713          	addi	a4,a4,-1 # ffffefff <__BSS_END__+0xfffeb63f>
   108e4:	00e7f7b3          	and	a5,a5,a4
   108e8:	01079793          	slli	a5,a5,0x10
   108ec:	4107d793          	srai	a5,a5,0x10
   108f0:	00c12083          	lw	ra,12(sp)
   108f4:	00f41623          	sh	a5,12(s0)
   108f8:	00812403          	lw	s0,8(sp)
   108fc:	01010113          	addi	sp,sp,16
   10900:	00008067          	ret

00010904 <__sclose>:
   10904:	00e59583          	lh	a1,14(a1)
   10908:	0040006f          	j	1090c <_close_r>

0001090c <_close_r>:
   1090c:	ff010113          	addi	sp,sp,-16
   10910:	00812423          	sw	s0,8(sp)
   10914:	00050413          	mv	s0,a0
   10918:	00058513          	mv	a0,a1
   1091c:	d801ae23          	sw	zero,-612(gp) # 1369c <errno>
   10920:	00112623          	sw	ra,12(sp)
   10924:	660010ef          	jal	11f84 <_close>
   10928:	fff00793          	li	a5,-1
   1092c:	00f50a63          	beq	a0,a5,10940 <_close_r+0x34>
   10930:	00c12083          	lw	ra,12(sp)
   10934:	00812403          	lw	s0,8(sp)
   10938:	01010113          	addi	sp,sp,16
   1093c:	00008067          	ret
   10940:	d9c1a783          	lw	a5,-612(gp) # 1369c <errno>
   10944:	fe0786e3          	beqz	a5,10930 <_close_r+0x24>
   10948:	00c12083          	lw	ra,12(sp)
   1094c:	00f42023          	sw	a5,0(s0)
   10950:	00812403          	lw	s0,8(sp)
   10954:	01010113          	addi	sp,sp,16
   10958:	00008067          	ret

0001095c <_reclaim_reent>:
   1095c:	d8c1a783          	lw	a5,-628(gp) # 1368c <_impure_ptr>
   10960:	0aa78e63          	beq	a5,a0,10a1c <_reclaim_reent+0xc0>
   10964:	04452583          	lw	a1,68(a0)
   10968:	fe010113          	addi	sp,sp,-32
   1096c:	00912a23          	sw	s1,20(sp)
   10970:	00112e23          	sw	ra,28(sp)
   10974:	00050493          	mv	s1,a0
   10978:	04058c63          	beqz	a1,109d0 <_reclaim_reent+0x74>
   1097c:	01212823          	sw	s2,16(sp)
   10980:	01312623          	sw	s3,12(sp)
   10984:	00812c23          	sw	s0,24(sp)
   10988:	00000913          	li	s2,0
   1098c:	08000993          	li	s3,128
   10990:	012587b3          	add	a5,a1,s2
   10994:	0007a403          	lw	s0,0(a5)
   10998:	00040e63          	beqz	s0,109b4 <_reclaim_reent+0x58>
   1099c:	00040593          	mv	a1,s0
   109a0:	00042403          	lw	s0,0(s0)
   109a4:	00048513          	mv	a0,s1
   109a8:	580000ef          	jal	10f28 <_free_r>
   109ac:	fe0418e3          	bnez	s0,1099c <_reclaim_reent+0x40>
   109b0:	0444a583          	lw	a1,68(s1)
   109b4:	00490913          	addi	s2,s2,4
   109b8:	fd391ce3          	bne	s2,s3,10990 <_reclaim_reent+0x34>
   109bc:	00048513          	mv	a0,s1
   109c0:	568000ef          	jal	10f28 <_free_r>
   109c4:	01812403          	lw	s0,24(sp)
   109c8:	01012903          	lw	s2,16(sp)
   109cc:	00c12983          	lw	s3,12(sp)
   109d0:	0384a583          	lw	a1,56(s1)
   109d4:	00058663          	beqz	a1,109e0 <_reclaim_reent+0x84>
   109d8:	00048513          	mv	a0,s1
   109dc:	54c000ef          	jal	10f28 <_free_r>
   109e0:	04c4a583          	lw	a1,76(s1)
   109e4:	00058663          	beqz	a1,109f0 <_reclaim_reent+0x94>
   109e8:	00048513          	mv	a0,s1
   109ec:	53c000ef          	jal	10f28 <_free_r>
   109f0:	0344a783          	lw	a5,52(s1)
   109f4:	00078c63          	beqz	a5,10a0c <_reclaim_reent+0xb0>
   109f8:	01c12083          	lw	ra,28(sp)
   109fc:	00048513          	mv	a0,s1
   10a00:	01412483          	lw	s1,20(sp)
   10a04:	02010113          	addi	sp,sp,32
   10a08:	00078067          	jr	a5
   10a0c:	01c12083          	lw	ra,28(sp)
   10a10:	01412483          	lw	s1,20(sp)
   10a14:	02010113          	addi	sp,sp,32
   10a18:	00008067          	ret
   10a1c:	00008067          	ret

00010a20 <_lseek_r>:
   10a20:	ff010113          	addi	sp,sp,-16
   10a24:	00058713          	mv	a4,a1
   10a28:	00812423          	sw	s0,8(sp)
   10a2c:	00060593          	mv	a1,a2
   10a30:	00050413          	mv	s0,a0
   10a34:	00068613          	mv	a2,a3
   10a38:	00070513          	mv	a0,a4
   10a3c:	d801ae23          	sw	zero,-612(gp) # 1369c <errno>
   10a40:	00112623          	sw	ra,12(sp)
   10a44:	550010ef          	jal	11f94 <_lseek>
   10a48:	fff00793          	li	a5,-1
   10a4c:	00f50a63          	beq	a0,a5,10a60 <_lseek_r+0x40>
   10a50:	00c12083          	lw	ra,12(sp)
   10a54:	00812403          	lw	s0,8(sp)
   10a58:	01010113          	addi	sp,sp,16
   10a5c:	00008067          	ret
   10a60:	d9c1a783          	lw	a5,-612(gp) # 1369c <errno>
   10a64:	fe0786e3          	beqz	a5,10a50 <_lseek_r+0x30>
   10a68:	00c12083          	lw	ra,12(sp)
   10a6c:	00f42023          	sw	a5,0(s0)
   10a70:	00812403          	lw	s0,8(sp)
   10a74:	01010113          	addi	sp,sp,16
   10a78:	00008067          	ret

00010a7c <_read_r>:
   10a7c:	ff010113          	addi	sp,sp,-16
   10a80:	00058713          	mv	a4,a1
   10a84:	00812423          	sw	s0,8(sp)
   10a88:	00060593          	mv	a1,a2
   10a8c:	00050413          	mv	s0,a0
   10a90:	00068613          	mv	a2,a3
   10a94:	00070513          	mv	a0,a4
   10a98:	d801ae23          	sw	zero,-612(gp) # 1369c <errno>
   10a9c:	00112623          	sw	ra,12(sp)
   10aa0:	504010ef          	jal	11fa4 <_read>
   10aa4:	fff00793          	li	a5,-1
   10aa8:	00f50a63          	beq	a0,a5,10abc <_read_r+0x40>
   10aac:	00c12083          	lw	ra,12(sp)
   10ab0:	00812403          	lw	s0,8(sp)
   10ab4:	01010113          	addi	sp,sp,16
   10ab8:	00008067          	ret
   10abc:	d9c1a783          	lw	a5,-612(gp) # 1369c <errno>
   10ac0:	fe0786e3          	beqz	a5,10aac <_read_r+0x30>
   10ac4:	00c12083          	lw	ra,12(sp)
   10ac8:	00f42023          	sw	a5,0(s0)
   10acc:	00812403          	lw	s0,8(sp)
   10ad0:	01010113          	addi	sp,sp,16
   10ad4:	00008067          	ret

00010ad8 <_write_r>:
   10ad8:	ff010113          	addi	sp,sp,-16
   10adc:	00058713          	mv	a4,a1
   10ae0:	00812423          	sw	s0,8(sp)
   10ae4:	00060593          	mv	a1,a2
   10ae8:	00050413          	mv	s0,a0
   10aec:	00068613          	mv	a2,a3
   10af0:	00070513          	mv	a0,a4
   10af4:	d801ae23          	sw	zero,-612(gp) # 1369c <errno>
   10af8:	00112623          	sw	ra,12(sp)
   10afc:	4e8010ef          	jal	11fe4 <_write>
   10b00:	fff00793          	li	a5,-1
   10b04:	00f50a63          	beq	a0,a5,10b18 <_write_r+0x40>
   10b08:	00c12083          	lw	ra,12(sp)
   10b0c:	00812403          	lw	s0,8(sp)
   10b10:	01010113          	addi	sp,sp,16
   10b14:	00008067          	ret
   10b18:	d9c1a783          	lw	a5,-612(gp) # 1369c <errno>
   10b1c:	fe0786e3          	beqz	a5,10b08 <_write_r+0x30>
   10b20:	00c12083          	lw	ra,12(sp)
   10b24:	00f42023          	sw	a5,0(s0)
   10b28:	00812403          	lw	s0,8(sp)
   10b2c:	01010113          	addi	sp,sp,16
   10b30:	00008067          	ret

00010b34 <__libc_init_array>:
   10b34:	ff010113          	addi	sp,sp,-16
   10b38:	00812423          	sw	s0,8(sp)
   10b3c:	01212023          	sw	s2,0(sp)
   10b40:	00002797          	auipc	a5,0x2
   10b44:	55478793          	addi	a5,a5,1364 # 13094 <__init_array_start>
   10b48:	00002417          	auipc	s0,0x2
   10b4c:	54c40413          	addi	s0,s0,1356 # 13094 <__init_array_start>
   10b50:	00112623          	sw	ra,12(sp)
   10b54:	00912223          	sw	s1,4(sp)
   10b58:	40878933          	sub	s2,a5,s0
   10b5c:	02878063          	beq	a5,s0,10b7c <__libc_init_array+0x48>
   10b60:	40295913          	srai	s2,s2,0x2
   10b64:	00000493          	li	s1,0
   10b68:	00042783          	lw	a5,0(s0)
   10b6c:	00148493          	addi	s1,s1,1
   10b70:	00440413          	addi	s0,s0,4
   10b74:	000780e7          	jalr	a5
   10b78:	ff24e8e3          	bltu	s1,s2,10b68 <__libc_init_array+0x34>
   10b7c:	00002797          	auipc	a5,0x2
   10b80:	52078793          	addi	a5,a5,1312 # 1309c <__do_global_dtors_aux_fini_array_entry>
   10b84:	00002417          	auipc	s0,0x2
   10b88:	51040413          	addi	s0,s0,1296 # 13094 <__init_array_start>
   10b8c:	40878933          	sub	s2,a5,s0
   10b90:	40295913          	srai	s2,s2,0x2
   10b94:	00878e63          	beq	a5,s0,10bb0 <__libc_init_array+0x7c>
   10b98:	00000493          	li	s1,0
   10b9c:	00042783          	lw	a5,0(s0)
   10ba0:	00148493          	addi	s1,s1,1
   10ba4:	00440413          	addi	s0,s0,4
   10ba8:	000780e7          	jalr	a5
   10bac:	ff24e8e3          	bltu	s1,s2,10b9c <__libc_init_array+0x68>
   10bb0:	00c12083          	lw	ra,12(sp)
   10bb4:	00812403          	lw	s0,8(sp)
   10bb8:	00412483          	lw	s1,4(sp)
   10bbc:	00012903          	lw	s2,0(sp)
   10bc0:	01010113          	addi	sp,sp,16
   10bc4:	00008067          	ret

00010bc8 <memset>:
   10bc8:	00f00313          	li	t1,15
   10bcc:	00050713          	mv	a4,a0
   10bd0:	02c37e63          	bgeu	t1,a2,10c0c <memset+0x44>
   10bd4:	00f77793          	andi	a5,a4,15
   10bd8:	0a079063          	bnez	a5,10c78 <memset+0xb0>
   10bdc:	08059263          	bnez	a1,10c60 <memset+0x98>
   10be0:	ff067693          	andi	a3,a2,-16
   10be4:	00f67613          	andi	a2,a2,15
   10be8:	00e686b3          	add	a3,a3,a4
   10bec:	00b72023          	sw	a1,0(a4)
   10bf0:	00b72223          	sw	a1,4(a4)
   10bf4:	00b72423          	sw	a1,8(a4)
   10bf8:	00b72623          	sw	a1,12(a4)
   10bfc:	01070713          	addi	a4,a4,16
   10c00:	fed766e3          	bltu	a4,a3,10bec <memset+0x24>
   10c04:	00061463          	bnez	a2,10c0c <memset+0x44>
   10c08:	00008067          	ret
   10c0c:	40c306b3          	sub	a3,t1,a2
   10c10:	00269693          	slli	a3,a3,0x2
   10c14:	00000297          	auipc	t0,0x0
   10c18:	005686b3          	add	a3,a3,t0
   10c1c:	00c68067          	jr	12(a3)
   10c20:	00b70723          	sb	a1,14(a4)
   10c24:	00b706a3          	sb	a1,13(a4)
   10c28:	00b70623          	sb	a1,12(a4)
   10c2c:	00b705a3          	sb	a1,11(a4)
   10c30:	00b70523          	sb	a1,10(a4)
   10c34:	00b704a3          	sb	a1,9(a4)
   10c38:	00b70423          	sb	a1,8(a4)
   10c3c:	00b703a3          	sb	a1,7(a4)
   10c40:	00b70323          	sb	a1,6(a4)
   10c44:	00b702a3          	sb	a1,5(a4)
   10c48:	00b70223          	sb	a1,4(a4)
   10c4c:	00b701a3          	sb	a1,3(a4)
   10c50:	00b70123          	sb	a1,2(a4)
   10c54:	00b700a3          	sb	a1,1(a4)
   10c58:	00b70023          	sb	a1,0(a4)
   10c5c:	00008067          	ret
   10c60:	0ff5f593          	zext.b	a1,a1
   10c64:	00859693          	slli	a3,a1,0x8
   10c68:	00d5e5b3          	or	a1,a1,a3
   10c6c:	01059693          	slli	a3,a1,0x10
   10c70:	00d5e5b3          	or	a1,a1,a3
   10c74:	f6dff06f          	j	10be0 <memset+0x18>
   10c78:	00279693          	slli	a3,a5,0x2
   10c7c:	00000297          	auipc	t0,0x0
   10c80:	005686b3          	add	a3,a3,t0
   10c84:	00008293          	mv	t0,ra
   10c88:	fa0680e7          	jalr	-96(a3)
   10c8c:	00028093          	mv	ra,t0
   10c90:	ff078793          	addi	a5,a5,-16
   10c94:	40f70733          	sub	a4,a4,a5
   10c98:	00f60633          	add	a2,a2,a5
   10c9c:	f6c378e3          	bgeu	t1,a2,10c0c <memset+0x44>
   10ca0:	f3dff06f          	j	10bdc <memset+0x14>

00010ca4 <__call_exitprocs>:
   10ca4:	fd010113          	addi	sp,sp,-48
   10ca8:	01412c23          	sw	s4,24(sp)
   10cac:	da018a13          	addi	s4,gp,-608 # 136a0 <__atexit>
   10cb0:	03212023          	sw	s2,32(sp)
   10cb4:	000a2903          	lw	s2,0(s4)
   10cb8:	02112623          	sw	ra,44(sp)
   10cbc:	0a090863          	beqz	s2,10d6c <__call_exitprocs+0xc8>
   10cc0:	01312e23          	sw	s3,28(sp)
   10cc4:	01512a23          	sw	s5,20(sp)
   10cc8:	01612823          	sw	s6,16(sp)
   10ccc:	01712623          	sw	s7,12(sp)
   10cd0:	02812423          	sw	s0,40(sp)
   10cd4:	02912223          	sw	s1,36(sp)
   10cd8:	01812423          	sw	s8,8(sp)
   10cdc:	00050b13          	mv	s6,a0
   10ce0:	00058b93          	mv	s7,a1
   10ce4:	fff00993          	li	s3,-1
   10ce8:	00100a93          	li	s5,1
   10cec:	00492483          	lw	s1,4(s2)
   10cf0:	fff48413          	addi	s0,s1,-1
   10cf4:	04044e63          	bltz	s0,10d50 <__call_exitprocs+0xac>
   10cf8:	00249493          	slli	s1,s1,0x2
   10cfc:	009904b3          	add	s1,s2,s1
   10d00:	080b9063          	bnez	s7,10d80 <__call_exitprocs+0xdc>
   10d04:	00492783          	lw	a5,4(s2)
   10d08:	0044a683          	lw	a3,4(s1)
   10d0c:	fff78793          	addi	a5,a5,-1
   10d10:	0a878c63          	beq	a5,s0,10dc8 <__call_exitprocs+0x124>
   10d14:	0004a223          	sw	zero,4(s1)
   10d18:	02068663          	beqz	a3,10d44 <__call_exitprocs+0xa0>
   10d1c:	18892783          	lw	a5,392(s2)
   10d20:	008a9733          	sll	a4,s5,s0
   10d24:	00492c03          	lw	s8,4(s2)
   10d28:	00f777b3          	and	a5,a4,a5
   10d2c:	06079663          	bnez	a5,10d98 <__call_exitprocs+0xf4>
   10d30:	000680e7          	jalr	a3
   10d34:	00492703          	lw	a4,4(s2)
   10d38:	000a2783          	lw	a5,0(s4)
   10d3c:	09871063          	bne	a4,s8,10dbc <__call_exitprocs+0x118>
   10d40:	07279e63          	bne	a5,s2,10dbc <__call_exitprocs+0x118>
   10d44:	fff40413          	addi	s0,s0,-1
   10d48:	ffc48493          	addi	s1,s1,-4
   10d4c:	fb341ae3          	bne	s0,s3,10d00 <__call_exitprocs+0x5c>
   10d50:	02812403          	lw	s0,40(sp)
   10d54:	02412483          	lw	s1,36(sp)
   10d58:	01c12983          	lw	s3,28(sp)
   10d5c:	01412a83          	lw	s5,20(sp)
   10d60:	01012b03          	lw	s6,16(sp)
   10d64:	00c12b83          	lw	s7,12(sp)
   10d68:	00812c03          	lw	s8,8(sp)
   10d6c:	02c12083          	lw	ra,44(sp)
   10d70:	02012903          	lw	s2,32(sp)
   10d74:	01812a03          	lw	s4,24(sp)
   10d78:	03010113          	addi	sp,sp,48
   10d7c:	00008067          	ret
   10d80:	1044a783          	lw	a5,260(s1)
   10d84:	f97780e3          	beq	a5,s7,10d04 <__call_exitprocs+0x60>
   10d88:	fff40413          	addi	s0,s0,-1
   10d8c:	ffc48493          	addi	s1,s1,-4
   10d90:	ff3418e3          	bne	s0,s3,10d80 <__call_exitprocs+0xdc>
   10d94:	fbdff06f          	j	10d50 <__call_exitprocs+0xac>
   10d98:	18c92783          	lw	a5,396(s2)
   10d9c:	0844a583          	lw	a1,132(s1)
   10da0:	00f77733          	and	a4,a4,a5
   10da4:	02071663          	bnez	a4,10dd0 <__call_exitprocs+0x12c>
   10da8:	000b0513          	mv	a0,s6
   10dac:	000680e7          	jalr	a3
   10db0:	00492703          	lw	a4,4(s2)
   10db4:	000a2783          	lw	a5,0(s4)
   10db8:	f98704e3          	beq	a4,s8,10d40 <__call_exitprocs+0x9c>
   10dbc:	f8078ae3          	beqz	a5,10d50 <__call_exitprocs+0xac>
   10dc0:	00078913          	mv	s2,a5
   10dc4:	f29ff06f          	j	10cec <__call_exitprocs+0x48>
   10dc8:	00892223          	sw	s0,4(s2)
   10dcc:	f4dff06f          	j	10d18 <__call_exitprocs+0x74>
   10dd0:	00058513          	mv	a0,a1
   10dd4:	000680e7          	jalr	a3
   10dd8:	f5dff06f          	j	10d34 <__call_exitprocs+0x90>

00010ddc <atexit>:
   10ddc:	00050593          	mv	a1,a0
   10de0:	00000693          	li	a3,0
   10de4:	00000613          	li	a2,0
   10de8:	00000513          	li	a0,0
   10dec:	1000106f          	j	11eec <__register_exitproc>

00010df0 <_malloc_trim_r>:
   10df0:	fe010113          	addi	sp,sp,-32
   10df4:	00812c23          	sw	s0,24(sp)
   10df8:	00912a23          	sw	s1,20(sp)
   10dfc:	01212823          	sw	s2,16(sp)
   10e00:	01312623          	sw	s3,12(sp)
   10e04:	01412423          	sw	s4,8(sp)
   10e08:	00058993          	mv	s3,a1
   10e0c:	00112e23          	sw	ra,28(sp)
   10e10:	00050913          	mv	s2,a0
   10e14:	00002a17          	auipc	s4,0x2
   10e18:	46ca0a13          	addi	s4,s4,1132 # 13280 <__malloc_av_>
   10e1c:	3d9000ef          	jal	119f4 <__malloc_lock>
   10e20:	008a2703          	lw	a4,8(s4)
   10e24:	000017b7          	lui	a5,0x1
   10e28:	fef78793          	addi	a5,a5,-17 # fef <exit-0xf0c5>
   10e2c:	00472483          	lw	s1,4(a4)
   10e30:	00001737          	lui	a4,0x1
   10e34:	ffc4f493          	andi	s1,s1,-4
   10e38:	00f48433          	add	s0,s1,a5
   10e3c:	41340433          	sub	s0,s0,s3
   10e40:	00c45413          	srli	s0,s0,0xc
   10e44:	fff40413          	addi	s0,s0,-1
   10e48:	00c41413          	slli	s0,s0,0xc
   10e4c:	00e44e63          	blt	s0,a4,10e68 <_malloc_trim_r+0x78>
   10e50:	00000593          	li	a1,0
   10e54:	00090513          	mv	a0,s2
   10e58:	7e9000ef          	jal	11e40 <_sbrk_r>
   10e5c:	008a2783          	lw	a5,8(s4)
   10e60:	009787b3          	add	a5,a5,s1
   10e64:	02f50863          	beq	a0,a5,10e94 <_malloc_trim_r+0xa4>
   10e68:	00090513          	mv	a0,s2
   10e6c:	38d000ef          	jal	119f8 <__malloc_unlock>
   10e70:	01c12083          	lw	ra,28(sp)
   10e74:	01812403          	lw	s0,24(sp)
   10e78:	01412483          	lw	s1,20(sp)
   10e7c:	01012903          	lw	s2,16(sp)
   10e80:	00c12983          	lw	s3,12(sp)
   10e84:	00812a03          	lw	s4,8(sp)
   10e88:	00000513          	li	a0,0
   10e8c:	02010113          	addi	sp,sp,32
   10e90:	00008067          	ret
   10e94:	408005b3          	neg	a1,s0
   10e98:	00090513          	mv	a0,s2
   10e9c:	7a5000ef          	jal	11e40 <_sbrk_r>
   10ea0:	fff00793          	li	a5,-1
   10ea4:	04f50863          	beq	a0,a5,10ef4 <_malloc_trim_r+0x104>
   10ea8:	f0818713          	addi	a4,gp,-248 # 13808 <__malloc_current_mallinfo>
   10eac:	00072783          	lw	a5,0(a4) # 1000 <exit-0xf0b4>
   10eb0:	008a2683          	lw	a3,8(s4)
   10eb4:	408484b3          	sub	s1,s1,s0
   10eb8:	0014e493          	ori	s1,s1,1
   10ebc:	408787b3          	sub	a5,a5,s0
   10ec0:	00090513          	mv	a0,s2
   10ec4:	0096a223          	sw	s1,4(a3)
   10ec8:	00f72023          	sw	a5,0(a4)
   10ecc:	32d000ef          	jal	119f8 <__malloc_unlock>
   10ed0:	01c12083          	lw	ra,28(sp)
   10ed4:	01812403          	lw	s0,24(sp)
   10ed8:	01412483          	lw	s1,20(sp)
   10edc:	01012903          	lw	s2,16(sp)
   10ee0:	00c12983          	lw	s3,12(sp)
   10ee4:	00812a03          	lw	s4,8(sp)
   10ee8:	00100513          	li	a0,1
   10eec:	02010113          	addi	sp,sp,32
   10ef0:	00008067          	ret
   10ef4:	00000593          	li	a1,0
   10ef8:	00090513          	mv	a0,s2
   10efc:	745000ef          	jal	11e40 <_sbrk_r>
   10f00:	008a2703          	lw	a4,8(s4)
   10f04:	00f00693          	li	a3,15
   10f08:	40e507b3          	sub	a5,a0,a4
   10f0c:	f4f6dee3          	bge	a3,a5,10e68 <_malloc_trim_r+0x78>
   10f10:	d901a683          	lw	a3,-624(gp) # 13690 <__malloc_sbrk_base>
   10f14:	40d50533          	sub	a0,a0,a3
   10f18:	0017e793          	ori	a5,a5,1
   10f1c:	f0a1a423          	sw	a0,-248(gp) # 13808 <__malloc_current_mallinfo>
   10f20:	00f72223          	sw	a5,4(a4)
   10f24:	f45ff06f          	j	10e68 <_malloc_trim_r+0x78>

00010f28 <_free_r>:
   10f28:	18058263          	beqz	a1,110ac <_free_r+0x184>
   10f2c:	ff010113          	addi	sp,sp,-16
   10f30:	00812423          	sw	s0,8(sp)
   10f34:	00912223          	sw	s1,4(sp)
   10f38:	00058413          	mv	s0,a1
   10f3c:	00050493          	mv	s1,a0
   10f40:	00112623          	sw	ra,12(sp)
   10f44:	2b1000ef          	jal	119f4 <__malloc_lock>
   10f48:	ffc42583          	lw	a1,-4(s0)
   10f4c:	ff840713          	addi	a4,s0,-8
   10f50:	00002517          	auipc	a0,0x2
   10f54:	33050513          	addi	a0,a0,816 # 13280 <__malloc_av_>
   10f58:	ffe5f793          	andi	a5,a1,-2
   10f5c:	00f70633          	add	a2,a4,a5
   10f60:	00462683          	lw	a3,4(a2)
   10f64:	00852803          	lw	a6,8(a0)
   10f68:	ffc6f693          	andi	a3,a3,-4
   10f6c:	1ac80263          	beq	a6,a2,11110 <_free_r+0x1e8>
   10f70:	00d62223          	sw	a3,4(a2)
   10f74:	0015f593          	andi	a1,a1,1
   10f78:	00d60833          	add	a6,a2,a3
   10f7c:	0a059063          	bnez	a1,1101c <_free_r+0xf4>
   10f80:	ff842303          	lw	t1,-8(s0)
   10f84:	00482583          	lw	a1,4(a6)
   10f88:	00002897          	auipc	a7,0x2
   10f8c:	30088893          	addi	a7,a7,768 # 13288 <__malloc_av_+0x8>
   10f90:	40670733          	sub	a4,a4,t1
   10f94:	00872803          	lw	a6,8(a4)
   10f98:	006787b3          	add	a5,a5,t1
   10f9c:	0015f593          	andi	a1,a1,1
   10fa0:	15180263          	beq	a6,a7,110e4 <_free_r+0x1bc>
   10fa4:	00c72303          	lw	t1,12(a4)
   10fa8:	00682623          	sw	t1,12(a6)
   10fac:	01032423          	sw	a6,8(t1) # 101a8 <frame_dummy+0x1c>
   10fb0:	1a058663          	beqz	a1,1115c <_free_r+0x234>
   10fb4:	0017e693          	ori	a3,a5,1
   10fb8:	00d72223          	sw	a3,4(a4)
   10fbc:	00f62023          	sw	a5,0(a2)
   10fc0:	1ff00693          	li	a3,511
   10fc4:	06f6ec63          	bltu	a3,a5,1103c <_free_r+0x114>
   10fc8:	ff87f693          	andi	a3,a5,-8
   10fcc:	00868693          	addi	a3,a3,8
   10fd0:	00452583          	lw	a1,4(a0)
   10fd4:	00d506b3          	add	a3,a0,a3
   10fd8:	0006a603          	lw	a2,0(a3)
   10fdc:	0057d813          	srli	a6,a5,0x5
   10fe0:	00100793          	li	a5,1
   10fe4:	010797b3          	sll	a5,a5,a6
   10fe8:	00b7e7b3          	or	a5,a5,a1
   10fec:	ff868593          	addi	a1,a3,-8
   10ff0:	00b72623          	sw	a1,12(a4)
   10ff4:	00c72423          	sw	a2,8(a4)
   10ff8:	00f52223          	sw	a5,4(a0)
   10ffc:	00e6a023          	sw	a4,0(a3)
   11000:	00e62623          	sw	a4,12(a2)
   11004:	00812403          	lw	s0,8(sp)
   11008:	00c12083          	lw	ra,12(sp)
   1100c:	00048513          	mv	a0,s1
   11010:	00412483          	lw	s1,4(sp)
   11014:	01010113          	addi	sp,sp,16
   11018:	1e10006f          	j	119f8 <__malloc_unlock>
   1101c:	00482583          	lw	a1,4(a6)
   11020:	0015f593          	andi	a1,a1,1
   11024:	08058663          	beqz	a1,110b0 <_free_r+0x188>
   11028:	0017e693          	ori	a3,a5,1
   1102c:	fed42e23          	sw	a3,-4(s0)
   11030:	00f62023          	sw	a5,0(a2)
   11034:	1ff00693          	li	a3,511
   11038:	f8f6f8e3          	bgeu	a3,a5,10fc8 <_free_r+0xa0>
   1103c:	0097d693          	srli	a3,a5,0x9
   11040:	00400613          	li	a2,4
   11044:	12d66063          	bltu	a2,a3,11164 <_free_r+0x23c>
   11048:	0067d693          	srli	a3,a5,0x6
   1104c:	03968593          	addi	a1,a3,57
   11050:	03868613          	addi	a2,a3,56
   11054:	00359593          	slli	a1,a1,0x3
   11058:	00b505b3          	add	a1,a0,a1
   1105c:	0005a683          	lw	a3,0(a1)
   11060:	ff858593          	addi	a1,a1,-8
   11064:	00d59863          	bne	a1,a3,11074 <_free_r+0x14c>
   11068:	1540006f          	j	111bc <_free_r+0x294>
   1106c:	0086a683          	lw	a3,8(a3)
   11070:	00d58863          	beq	a1,a3,11080 <_free_r+0x158>
   11074:	0046a603          	lw	a2,4(a3)
   11078:	ffc67613          	andi	a2,a2,-4
   1107c:	fec7e8e3          	bltu	a5,a2,1106c <_free_r+0x144>
   11080:	00c6a583          	lw	a1,12(a3)
   11084:	00b72623          	sw	a1,12(a4)
   11088:	00d72423          	sw	a3,8(a4)
   1108c:	00812403          	lw	s0,8(sp)
   11090:	00c12083          	lw	ra,12(sp)
   11094:	00e5a423          	sw	a4,8(a1)
   11098:	00048513          	mv	a0,s1
   1109c:	00412483          	lw	s1,4(sp)
   110a0:	00e6a623          	sw	a4,12(a3)
   110a4:	01010113          	addi	sp,sp,16
   110a8:	1510006f          	j	119f8 <__malloc_unlock>
   110ac:	00008067          	ret
   110b0:	00d787b3          	add	a5,a5,a3
   110b4:	00002897          	auipc	a7,0x2
   110b8:	1d488893          	addi	a7,a7,468 # 13288 <__malloc_av_+0x8>
   110bc:	00862683          	lw	a3,8(a2)
   110c0:	0d168c63          	beq	a3,a7,11198 <_free_r+0x270>
   110c4:	00c62803          	lw	a6,12(a2)
   110c8:	0017e593          	ori	a1,a5,1
   110cc:	00f70633          	add	a2,a4,a5
   110d0:	0106a623          	sw	a6,12(a3)
   110d4:	00d82423          	sw	a3,8(a6)
   110d8:	00b72223          	sw	a1,4(a4)
   110dc:	00f62023          	sw	a5,0(a2)
   110e0:	ee1ff06f          	j	10fc0 <_free_r+0x98>
   110e4:	12059c63          	bnez	a1,1121c <_free_r+0x2f4>
   110e8:	00862583          	lw	a1,8(a2)
   110ec:	00c62603          	lw	a2,12(a2)
   110f0:	00f686b3          	add	a3,a3,a5
   110f4:	0016e793          	ori	a5,a3,1
   110f8:	00c5a623          	sw	a2,12(a1)
   110fc:	00b62423          	sw	a1,8(a2)
   11100:	00f72223          	sw	a5,4(a4)
   11104:	00d70733          	add	a4,a4,a3
   11108:	00d72023          	sw	a3,0(a4)
   1110c:	ef9ff06f          	j	11004 <_free_r+0xdc>
   11110:	0015f593          	andi	a1,a1,1
   11114:	00d786b3          	add	a3,a5,a3
   11118:	02059063          	bnez	a1,11138 <_free_r+0x210>
   1111c:	ff842583          	lw	a1,-8(s0)
   11120:	40b70733          	sub	a4,a4,a1
   11124:	00c72783          	lw	a5,12(a4)
   11128:	00872603          	lw	a2,8(a4)
   1112c:	00b686b3          	add	a3,a3,a1
   11130:	00f62623          	sw	a5,12(a2)
   11134:	00c7a423          	sw	a2,8(a5)
   11138:	0016e793          	ori	a5,a3,1
   1113c:	00f72223          	sw	a5,4(a4)
   11140:	00e52423          	sw	a4,8(a0)
   11144:	d941a783          	lw	a5,-620(gp) # 13694 <__malloc_trim_threshold>
   11148:	eaf6eee3          	bltu	a3,a5,11004 <_free_r+0xdc>
   1114c:	dac1a583          	lw	a1,-596(gp) # 136ac <__malloc_top_pad>
   11150:	00048513          	mv	a0,s1
   11154:	c9dff0ef          	jal	10df0 <_malloc_trim_r>
   11158:	eadff06f          	j	11004 <_free_r+0xdc>
   1115c:	00d787b3          	add	a5,a5,a3
   11160:	f5dff06f          	j	110bc <_free_r+0x194>
   11164:	01400613          	li	a2,20
   11168:	02d67063          	bgeu	a2,a3,11188 <_free_r+0x260>
   1116c:	05400613          	li	a2,84
   11170:	06d66463          	bltu	a2,a3,111d8 <_free_r+0x2b0>
   11174:	00c7d693          	srli	a3,a5,0xc
   11178:	06f68593          	addi	a1,a3,111
   1117c:	06e68613          	addi	a2,a3,110
   11180:	00359593          	slli	a1,a1,0x3
   11184:	ed5ff06f          	j	11058 <_free_r+0x130>
   11188:	05c68593          	addi	a1,a3,92
   1118c:	05b68613          	addi	a2,a3,91
   11190:	00359593          	slli	a1,a1,0x3
   11194:	ec5ff06f          	j	11058 <_free_r+0x130>
   11198:	00e52a23          	sw	a4,20(a0)
   1119c:	00e52823          	sw	a4,16(a0)
   111a0:	0017e693          	ori	a3,a5,1
   111a4:	01172623          	sw	a7,12(a4)
   111a8:	01172423          	sw	a7,8(a4)
   111ac:	00d72223          	sw	a3,4(a4)
   111b0:	00f70733          	add	a4,a4,a5
   111b4:	00f72023          	sw	a5,0(a4)
   111b8:	e4dff06f          	j	11004 <_free_r+0xdc>
   111bc:	00452803          	lw	a6,4(a0)
   111c0:	40265613          	srai	a2,a2,0x2
   111c4:	00100793          	li	a5,1
   111c8:	00c797b3          	sll	a5,a5,a2
   111cc:	0107e7b3          	or	a5,a5,a6
   111d0:	00f52223          	sw	a5,4(a0)
   111d4:	eb1ff06f          	j	11084 <_free_r+0x15c>
   111d8:	15400613          	li	a2,340
   111dc:	00d66c63          	bltu	a2,a3,111f4 <_free_r+0x2cc>
   111e0:	00f7d693          	srli	a3,a5,0xf
   111e4:	07868593          	addi	a1,a3,120
   111e8:	07768613          	addi	a2,a3,119
   111ec:	00359593          	slli	a1,a1,0x3
   111f0:	e69ff06f          	j	11058 <_free_r+0x130>
   111f4:	55400613          	li	a2,1364
   111f8:	00d66c63          	bltu	a2,a3,11210 <_free_r+0x2e8>
   111fc:	0127d693          	srli	a3,a5,0x12
   11200:	07d68593          	addi	a1,a3,125
   11204:	07c68613          	addi	a2,a3,124
   11208:	00359593          	slli	a1,a1,0x3
   1120c:	e4dff06f          	j	11058 <_free_r+0x130>
   11210:	3f800593          	li	a1,1016
   11214:	07e00613          	li	a2,126
   11218:	e41ff06f          	j	11058 <_free_r+0x130>
   1121c:	0017e693          	ori	a3,a5,1
   11220:	00d72223          	sw	a3,4(a4)
   11224:	00f62023          	sw	a5,0(a2)
   11228:	dddff06f          	j	11004 <_free_r+0xdc>

0001122c <_malloc_r>:
   1122c:	fd010113          	addi	sp,sp,-48
   11230:	03212023          	sw	s2,32(sp)
   11234:	02112623          	sw	ra,44(sp)
   11238:	02812423          	sw	s0,40(sp)
   1123c:	02912223          	sw	s1,36(sp)
   11240:	01312e23          	sw	s3,28(sp)
   11244:	00b58793          	addi	a5,a1,11
   11248:	01600713          	li	a4,22
   1124c:	00050913          	mv	s2,a0
   11250:	08f76263          	bltu	a4,a5,112d4 <_malloc_r+0xa8>
   11254:	01000793          	li	a5,16
   11258:	20b7e663          	bltu	a5,a1,11464 <_malloc_r+0x238>
   1125c:	798000ef          	jal	119f4 <__malloc_lock>
   11260:	01800793          	li	a5,24
   11264:	00200593          	li	a1,2
   11268:	01000493          	li	s1,16
   1126c:	00002997          	auipc	s3,0x2
   11270:	01498993          	addi	s3,s3,20 # 13280 <__malloc_av_>
   11274:	00f987b3          	add	a5,s3,a5
   11278:	0047a403          	lw	s0,4(a5)
   1127c:	ff878713          	addi	a4,a5,-8
   11280:	34e40a63          	beq	s0,a4,115d4 <_malloc_r+0x3a8>
   11284:	00442783          	lw	a5,4(s0)
   11288:	00c42683          	lw	a3,12(s0)
   1128c:	00842603          	lw	a2,8(s0)
   11290:	ffc7f793          	andi	a5,a5,-4
   11294:	00f407b3          	add	a5,s0,a5
   11298:	0047a703          	lw	a4,4(a5)
   1129c:	00d62623          	sw	a3,12(a2)
   112a0:	00c6a423          	sw	a2,8(a3)
   112a4:	00176713          	ori	a4,a4,1
   112a8:	00090513          	mv	a0,s2
   112ac:	00e7a223          	sw	a4,4(a5)
   112b0:	748000ef          	jal	119f8 <__malloc_unlock>
   112b4:	00840513          	addi	a0,s0,8
   112b8:	02c12083          	lw	ra,44(sp)
   112bc:	02812403          	lw	s0,40(sp)
   112c0:	02412483          	lw	s1,36(sp)
   112c4:	02012903          	lw	s2,32(sp)
   112c8:	01c12983          	lw	s3,28(sp)
   112cc:	03010113          	addi	sp,sp,48
   112d0:	00008067          	ret
   112d4:	ff87f493          	andi	s1,a5,-8
   112d8:	1807c663          	bltz	a5,11464 <_malloc_r+0x238>
   112dc:	18b4e463          	bltu	s1,a1,11464 <_malloc_r+0x238>
   112e0:	714000ef          	jal	119f4 <__malloc_lock>
   112e4:	1f700793          	li	a5,503
   112e8:	4097f063          	bgeu	a5,s1,116e8 <_malloc_r+0x4bc>
   112ec:	0094d793          	srli	a5,s1,0x9
   112f0:	18078263          	beqz	a5,11474 <_malloc_r+0x248>
   112f4:	00400713          	li	a4,4
   112f8:	34f76663          	bltu	a4,a5,11644 <_malloc_r+0x418>
   112fc:	0064d793          	srli	a5,s1,0x6
   11300:	03978593          	addi	a1,a5,57
   11304:	03878813          	addi	a6,a5,56
   11308:	00359613          	slli	a2,a1,0x3
   1130c:	00002997          	auipc	s3,0x2
   11310:	f7498993          	addi	s3,s3,-140 # 13280 <__malloc_av_>
   11314:	00c98633          	add	a2,s3,a2
   11318:	00462403          	lw	s0,4(a2)
   1131c:	ff860613          	addi	a2,a2,-8
   11320:	02860863          	beq	a2,s0,11350 <_malloc_r+0x124>
   11324:	00f00513          	li	a0,15
   11328:	0140006f          	j	1133c <_malloc_r+0x110>
   1132c:	00c42683          	lw	a3,12(s0)
   11330:	28075e63          	bgez	a4,115cc <_malloc_r+0x3a0>
   11334:	00d60e63          	beq	a2,a3,11350 <_malloc_r+0x124>
   11338:	00068413          	mv	s0,a3
   1133c:	00442783          	lw	a5,4(s0)
   11340:	ffc7f793          	andi	a5,a5,-4
   11344:	40978733          	sub	a4,a5,s1
   11348:	fee552e3          	bge	a0,a4,1132c <_malloc_r+0x100>
   1134c:	00080593          	mv	a1,a6
   11350:	0109a403          	lw	s0,16(s3)
   11354:	00002897          	auipc	a7,0x2
   11358:	f3488893          	addi	a7,a7,-204 # 13288 <__malloc_av_+0x8>
   1135c:	27140463          	beq	s0,a7,115c4 <_malloc_r+0x398>
   11360:	00442783          	lw	a5,4(s0)
   11364:	00f00693          	li	a3,15
   11368:	ffc7f793          	andi	a5,a5,-4
   1136c:	40978733          	sub	a4,a5,s1
   11370:	38e6c263          	blt	a3,a4,116f4 <_malloc_r+0x4c8>
   11374:	0119aa23          	sw	a7,20(s3)
   11378:	0119a823          	sw	a7,16(s3)
   1137c:	34075663          	bgez	a4,116c8 <_malloc_r+0x49c>
   11380:	1ff00713          	li	a4,511
   11384:	0049a503          	lw	a0,4(s3)
   11388:	24f76e63          	bltu	a4,a5,115e4 <_malloc_r+0x3b8>
   1138c:	ff87f713          	andi	a4,a5,-8
   11390:	00870713          	addi	a4,a4,8
   11394:	00e98733          	add	a4,s3,a4
   11398:	00072683          	lw	a3,0(a4)
   1139c:	0057d613          	srli	a2,a5,0x5
   113a0:	00100793          	li	a5,1
   113a4:	00c797b3          	sll	a5,a5,a2
   113a8:	00f56533          	or	a0,a0,a5
   113ac:	ff870793          	addi	a5,a4,-8
   113b0:	00f42623          	sw	a5,12(s0)
   113b4:	00d42423          	sw	a3,8(s0)
   113b8:	00a9a223          	sw	a0,4(s3)
   113bc:	00872023          	sw	s0,0(a4)
   113c0:	0086a623          	sw	s0,12(a3)
   113c4:	4025d793          	srai	a5,a1,0x2
   113c8:	00100613          	li	a2,1
   113cc:	00f61633          	sll	a2,a2,a5
   113d0:	0ac56a63          	bltu	a0,a2,11484 <_malloc_r+0x258>
   113d4:	00a677b3          	and	a5,a2,a0
   113d8:	02079463          	bnez	a5,11400 <_malloc_r+0x1d4>
   113dc:	00161613          	slli	a2,a2,0x1
   113e0:	ffc5f593          	andi	a1,a1,-4
   113e4:	00a677b3          	and	a5,a2,a0
   113e8:	00458593          	addi	a1,a1,4
   113ec:	00079a63          	bnez	a5,11400 <_malloc_r+0x1d4>
   113f0:	00161613          	slli	a2,a2,0x1
   113f4:	00a677b3          	and	a5,a2,a0
   113f8:	00458593          	addi	a1,a1,4
   113fc:	fe078ae3          	beqz	a5,113f0 <_malloc_r+0x1c4>
   11400:	00f00813          	li	a6,15
   11404:	00359313          	slli	t1,a1,0x3
   11408:	00698333          	add	t1,s3,t1
   1140c:	00030513          	mv	a0,t1
   11410:	00c52783          	lw	a5,12(a0)
   11414:	00058e13          	mv	t3,a1
   11418:	24f50863          	beq	a0,a5,11668 <_malloc_r+0x43c>
   1141c:	0047a703          	lw	a4,4(a5)
   11420:	00078413          	mv	s0,a5
   11424:	00c7a783          	lw	a5,12(a5)
   11428:	ffc77713          	andi	a4,a4,-4
   1142c:	409706b3          	sub	a3,a4,s1
   11430:	24d84863          	blt	a6,a3,11680 <_malloc_r+0x454>
   11434:	fe06c2e3          	bltz	a3,11418 <_malloc_r+0x1ec>
   11438:	00e40733          	add	a4,s0,a4
   1143c:	00472683          	lw	a3,4(a4)
   11440:	00842603          	lw	a2,8(s0)
   11444:	00090513          	mv	a0,s2
   11448:	0016e693          	ori	a3,a3,1
   1144c:	00d72223          	sw	a3,4(a4)
   11450:	00f62623          	sw	a5,12(a2)
   11454:	00c7a423          	sw	a2,8(a5)
   11458:	5a0000ef          	jal	119f8 <__malloc_unlock>
   1145c:	00840513          	addi	a0,s0,8
   11460:	e59ff06f          	j	112b8 <_malloc_r+0x8c>
   11464:	00c00793          	li	a5,12
   11468:	00f92023          	sw	a5,0(s2)
   1146c:	00000513          	li	a0,0
   11470:	e49ff06f          	j	112b8 <_malloc_r+0x8c>
   11474:	20000613          	li	a2,512
   11478:	04000593          	li	a1,64
   1147c:	03f00813          	li	a6,63
   11480:	e8dff06f          	j	1130c <_malloc_r+0xe0>
   11484:	0089a403          	lw	s0,8(s3)
   11488:	01612823          	sw	s6,16(sp)
   1148c:	00442783          	lw	a5,4(s0)
   11490:	ffc7fb13          	andi	s6,a5,-4
   11494:	009b6863          	bltu	s6,s1,114a4 <_malloc_r+0x278>
   11498:	409b0733          	sub	a4,s6,s1
   1149c:	00f00793          	li	a5,15
   114a0:	0ee7c063          	blt	a5,a4,11580 <_malloc_r+0x354>
   114a4:	01912223          	sw	s9,4(sp)
   114a8:	d9018c93          	addi	s9,gp,-624 # 13690 <__malloc_sbrk_base>
   114ac:	000ca703          	lw	a4,0(s9)
   114b0:	01412c23          	sw	s4,24(sp)
   114b4:	01512a23          	sw	s5,20(sp)
   114b8:	01712623          	sw	s7,12(sp)
   114bc:	dac1aa83          	lw	s5,-596(gp) # 136ac <__malloc_top_pad>
   114c0:	fff00793          	li	a5,-1
   114c4:	01640a33          	add	s4,s0,s6
   114c8:	01548ab3          	add	s5,s1,s5
   114cc:	3cf70a63          	beq	a4,a5,118a0 <_malloc_r+0x674>
   114d0:	000017b7          	lui	a5,0x1
   114d4:	00f78793          	addi	a5,a5,15 # 100f <exit-0xf0a5>
   114d8:	00fa8ab3          	add	s5,s5,a5
   114dc:	fffff7b7          	lui	a5,0xfffff
   114e0:	00fafab3          	and	s5,s5,a5
   114e4:	000a8593          	mv	a1,s5
   114e8:	00090513          	mv	a0,s2
   114ec:	155000ef          	jal	11e40 <_sbrk_r>
   114f0:	fff00793          	li	a5,-1
   114f4:	00050b93          	mv	s7,a0
   114f8:	44f50e63          	beq	a0,a5,11954 <_malloc_r+0x728>
   114fc:	01812423          	sw	s8,8(sp)
   11500:	25456263          	bltu	a0,s4,11744 <_malloc_r+0x518>
   11504:	f0818c13          	addi	s8,gp,-248 # 13808 <__malloc_current_mallinfo>
   11508:	000c2583          	lw	a1,0(s8)
   1150c:	00ba85b3          	add	a1,s5,a1
   11510:	00bc2023          	sw	a1,0(s8)
   11514:	00058713          	mv	a4,a1
   11518:	2aaa1a63          	bne	s4,a0,117cc <_malloc_r+0x5a0>
   1151c:	01451793          	slli	a5,a0,0x14
   11520:	2a079663          	bnez	a5,117cc <_malloc_r+0x5a0>
   11524:	0089ab83          	lw	s7,8(s3)
   11528:	015b07b3          	add	a5,s6,s5
   1152c:	0017e793          	ori	a5,a5,1
   11530:	00fba223          	sw	a5,4(s7)
   11534:	da818713          	addi	a4,gp,-600 # 136a8 <__malloc_max_sbrked_mem>
   11538:	00072683          	lw	a3,0(a4)
   1153c:	00b6f463          	bgeu	a3,a1,11544 <_malloc_r+0x318>
   11540:	00b72023          	sw	a1,0(a4)
   11544:	da418713          	addi	a4,gp,-604 # 136a4 <__malloc_max_total_mem>
   11548:	00072683          	lw	a3,0(a4)
   1154c:	00b6f463          	bgeu	a3,a1,11554 <_malloc_r+0x328>
   11550:	00b72023          	sw	a1,0(a4)
   11554:	00812c03          	lw	s8,8(sp)
   11558:	000b8413          	mv	s0,s7
   1155c:	ffc7f793          	andi	a5,a5,-4
   11560:	40978733          	sub	a4,a5,s1
   11564:	3897ea63          	bltu	a5,s1,118f8 <_malloc_r+0x6cc>
   11568:	00f00793          	li	a5,15
   1156c:	38e7d663          	bge	a5,a4,118f8 <_malloc_r+0x6cc>
   11570:	01812a03          	lw	s4,24(sp)
   11574:	01412a83          	lw	s5,20(sp)
   11578:	00c12b83          	lw	s7,12(sp)
   1157c:	00412c83          	lw	s9,4(sp)
   11580:	0014e793          	ori	a5,s1,1
   11584:	00f42223          	sw	a5,4(s0)
   11588:	009404b3          	add	s1,s0,s1
   1158c:	0099a423          	sw	s1,8(s3)
   11590:	00176713          	ori	a4,a4,1
   11594:	00090513          	mv	a0,s2
   11598:	00e4a223          	sw	a4,4(s1)
   1159c:	45c000ef          	jal	119f8 <__malloc_unlock>
   115a0:	02c12083          	lw	ra,44(sp)
   115a4:	00840513          	addi	a0,s0,8
   115a8:	02812403          	lw	s0,40(sp)
   115ac:	01012b03          	lw	s6,16(sp)
   115b0:	02412483          	lw	s1,36(sp)
   115b4:	02012903          	lw	s2,32(sp)
   115b8:	01c12983          	lw	s3,28(sp)
   115bc:	03010113          	addi	sp,sp,48
   115c0:	00008067          	ret
   115c4:	0049a503          	lw	a0,4(s3)
   115c8:	dfdff06f          	j	113c4 <_malloc_r+0x198>
   115cc:	00842603          	lw	a2,8(s0)
   115d0:	cc5ff06f          	j	11294 <_malloc_r+0x68>
   115d4:	00c7a403          	lw	s0,12(a5) # fffff00c <__BSS_END__+0xfffeb64c>
   115d8:	00258593          	addi	a1,a1,2
   115dc:	d6878ae3          	beq	a5,s0,11350 <_malloc_r+0x124>
   115e0:	ca5ff06f          	j	11284 <_malloc_r+0x58>
   115e4:	0097d713          	srli	a4,a5,0x9
   115e8:	00400693          	li	a3,4
   115ec:	14e6f263          	bgeu	a3,a4,11730 <_malloc_r+0x504>
   115f0:	01400693          	li	a3,20
   115f4:	32e6e463          	bltu	a3,a4,1191c <_malloc_r+0x6f0>
   115f8:	05c70613          	addi	a2,a4,92
   115fc:	05b70693          	addi	a3,a4,91
   11600:	00361613          	slli	a2,a2,0x3
   11604:	00c98633          	add	a2,s3,a2
   11608:	00062703          	lw	a4,0(a2)
   1160c:	ff860613          	addi	a2,a2,-8
   11610:	00e61863          	bne	a2,a4,11620 <_malloc_r+0x3f4>
   11614:	2940006f          	j	118a8 <_malloc_r+0x67c>
   11618:	00872703          	lw	a4,8(a4)
   1161c:	00e60863          	beq	a2,a4,1162c <_malloc_r+0x400>
   11620:	00472683          	lw	a3,4(a4)
   11624:	ffc6f693          	andi	a3,a3,-4
   11628:	fed7e8e3          	bltu	a5,a3,11618 <_malloc_r+0x3ec>
   1162c:	00c72603          	lw	a2,12(a4)
   11630:	00c42623          	sw	a2,12(s0)
   11634:	00e42423          	sw	a4,8(s0)
   11638:	00862423          	sw	s0,8(a2)
   1163c:	00872623          	sw	s0,12(a4)
   11640:	d85ff06f          	j	113c4 <_malloc_r+0x198>
   11644:	01400713          	li	a4,20
   11648:	10f77863          	bgeu	a4,a5,11758 <_malloc_r+0x52c>
   1164c:	05400713          	li	a4,84
   11650:	2ef76463          	bltu	a4,a5,11938 <_malloc_r+0x70c>
   11654:	00c4d793          	srli	a5,s1,0xc
   11658:	06f78593          	addi	a1,a5,111
   1165c:	06e78813          	addi	a6,a5,110
   11660:	00359613          	slli	a2,a1,0x3
   11664:	ca9ff06f          	j	1130c <_malloc_r+0xe0>
   11668:	001e0e13          	addi	t3,t3,1
   1166c:	003e7793          	andi	a5,t3,3
   11670:	00850513          	addi	a0,a0,8
   11674:	10078063          	beqz	a5,11774 <_malloc_r+0x548>
   11678:	00c52783          	lw	a5,12(a0)
   1167c:	d9dff06f          	j	11418 <_malloc_r+0x1ec>
   11680:	00842603          	lw	a2,8(s0)
   11684:	0014e593          	ori	a1,s1,1
   11688:	00b42223          	sw	a1,4(s0)
   1168c:	00f62623          	sw	a5,12(a2)
   11690:	00c7a423          	sw	a2,8(a5)
   11694:	009404b3          	add	s1,s0,s1
   11698:	0099aa23          	sw	s1,20(s3)
   1169c:	0099a823          	sw	s1,16(s3)
   116a0:	0016e793          	ori	a5,a3,1
   116a4:	0114a623          	sw	a7,12(s1)
   116a8:	0114a423          	sw	a7,8(s1)
   116ac:	00f4a223          	sw	a5,4(s1)
   116b0:	00e40733          	add	a4,s0,a4
   116b4:	00090513          	mv	a0,s2
   116b8:	00d72023          	sw	a3,0(a4)
   116bc:	33c000ef          	jal	119f8 <__malloc_unlock>
   116c0:	00840513          	addi	a0,s0,8
   116c4:	bf5ff06f          	j	112b8 <_malloc_r+0x8c>
   116c8:	00f407b3          	add	a5,s0,a5
   116cc:	0047a703          	lw	a4,4(a5)
   116d0:	00090513          	mv	a0,s2
   116d4:	00176713          	ori	a4,a4,1
   116d8:	00e7a223          	sw	a4,4(a5)
   116dc:	31c000ef          	jal	119f8 <__malloc_unlock>
   116e0:	00840513          	addi	a0,s0,8
   116e4:	bd5ff06f          	j	112b8 <_malloc_r+0x8c>
   116e8:	0034d593          	srli	a1,s1,0x3
   116ec:	00848793          	addi	a5,s1,8
   116f0:	b7dff06f          	j	1126c <_malloc_r+0x40>
   116f4:	0014e693          	ori	a3,s1,1
   116f8:	00d42223          	sw	a3,4(s0)
   116fc:	009404b3          	add	s1,s0,s1
   11700:	0099aa23          	sw	s1,20(s3)
   11704:	0099a823          	sw	s1,16(s3)
   11708:	00176693          	ori	a3,a4,1
   1170c:	0114a623          	sw	a7,12(s1)
   11710:	0114a423          	sw	a7,8(s1)
   11714:	00d4a223          	sw	a3,4(s1)
   11718:	00f407b3          	add	a5,s0,a5
   1171c:	00090513          	mv	a0,s2
   11720:	00e7a023          	sw	a4,0(a5)
   11724:	2d4000ef          	jal	119f8 <__malloc_unlock>
   11728:	00840513          	addi	a0,s0,8
   1172c:	b8dff06f          	j	112b8 <_malloc_r+0x8c>
   11730:	0067d713          	srli	a4,a5,0x6
   11734:	03970613          	addi	a2,a4,57
   11738:	03870693          	addi	a3,a4,56
   1173c:	00361613          	slli	a2,a2,0x3
   11740:	ec5ff06f          	j	11604 <_malloc_r+0x3d8>
   11744:	07340c63          	beq	s0,s3,117bc <_malloc_r+0x590>
   11748:	0089a403          	lw	s0,8(s3)
   1174c:	00812c03          	lw	s8,8(sp)
   11750:	00442783          	lw	a5,4(s0)
   11754:	e09ff06f          	j	1155c <_malloc_r+0x330>
   11758:	05c78593          	addi	a1,a5,92
   1175c:	05b78813          	addi	a6,a5,91
   11760:	00359613          	slli	a2,a1,0x3
   11764:	ba9ff06f          	j	1130c <_malloc_r+0xe0>
   11768:	00832783          	lw	a5,8(t1)
   1176c:	fff58593          	addi	a1,a1,-1
   11770:	26679e63          	bne	a5,t1,119ec <_malloc_r+0x7c0>
   11774:	0035f793          	andi	a5,a1,3
   11778:	ff830313          	addi	t1,t1,-8
   1177c:	fe0796e3          	bnez	a5,11768 <_malloc_r+0x53c>
   11780:	0049a703          	lw	a4,4(s3)
   11784:	fff64793          	not	a5,a2
   11788:	00e7f7b3          	and	a5,a5,a4
   1178c:	00f9a223          	sw	a5,4(s3)
   11790:	00161613          	slli	a2,a2,0x1
   11794:	cec7e8e3          	bltu	a5,a2,11484 <_malloc_r+0x258>
   11798:	ce0606e3          	beqz	a2,11484 <_malloc_r+0x258>
   1179c:	00f67733          	and	a4,a2,a5
   117a0:	00071a63          	bnez	a4,117b4 <_malloc_r+0x588>
   117a4:	00161613          	slli	a2,a2,0x1
   117a8:	00f67733          	and	a4,a2,a5
   117ac:	004e0e13          	addi	t3,t3,4
   117b0:	fe070ae3          	beqz	a4,117a4 <_malloc_r+0x578>
   117b4:	000e0593          	mv	a1,t3
   117b8:	c4dff06f          	j	11404 <_malloc_r+0x1d8>
   117bc:	f0818c13          	addi	s8,gp,-248 # 13808 <__malloc_current_mallinfo>
   117c0:	000c2703          	lw	a4,0(s8)
   117c4:	00ea8733          	add	a4,s5,a4
   117c8:	00ec2023          	sw	a4,0(s8)
   117cc:	000ca683          	lw	a3,0(s9)
   117d0:	fff00793          	li	a5,-1
   117d4:	18f68663          	beq	a3,a5,11960 <_malloc_r+0x734>
   117d8:	414b87b3          	sub	a5,s7,s4
   117dc:	00e787b3          	add	a5,a5,a4
   117e0:	00fc2023          	sw	a5,0(s8)
   117e4:	007bfc93          	andi	s9,s7,7
   117e8:	0c0c8c63          	beqz	s9,118c0 <_malloc_r+0x694>
   117ec:	419b8bb3          	sub	s7,s7,s9
   117f0:	000017b7          	lui	a5,0x1
   117f4:	00878793          	addi	a5,a5,8 # 1008 <exit-0xf0ac>
   117f8:	008b8b93          	addi	s7,s7,8
   117fc:	419785b3          	sub	a1,a5,s9
   11800:	015b8ab3          	add	s5,s7,s5
   11804:	415585b3          	sub	a1,a1,s5
   11808:	01459593          	slli	a1,a1,0x14
   1180c:	0145da13          	srli	s4,a1,0x14
   11810:	000a0593          	mv	a1,s4
   11814:	00090513          	mv	a0,s2
   11818:	628000ef          	jal	11e40 <_sbrk_r>
   1181c:	fff00793          	li	a5,-1
   11820:	18f50063          	beq	a0,a5,119a0 <_malloc_r+0x774>
   11824:	41750533          	sub	a0,a0,s7
   11828:	01450ab3          	add	s5,a0,s4
   1182c:	000c2703          	lw	a4,0(s8)
   11830:	0179a423          	sw	s7,8(s3)
   11834:	001ae793          	ori	a5,s5,1
   11838:	00ea05b3          	add	a1,s4,a4
   1183c:	00bc2023          	sw	a1,0(s8)
   11840:	00fba223          	sw	a5,4(s7)
   11844:	cf3408e3          	beq	s0,s3,11534 <_malloc_r+0x308>
   11848:	00f00693          	li	a3,15
   1184c:	0b66f063          	bgeu	a3,s6,118ec <_malloc_r+0x6c0>
   11850:	00442703          	lw	a4,4(s0)
   11854:	ff4b0793          	addi	a5,s6,-12
   11858:	ff87f793          	andi	a5,a5,-8
   1185c:	00177713          	andi	a4,a4,1
   11860:	00f76733          	or	a4,a4,a5
   11864:	00e42223          	sw	a4,4(s0)
   11868:	00500613          	li	a2,5
   1186c:	00f40733          	add	a4,s0,a5
   11870:	00c72223          	sw	a2,4(a4)
   11874:	00c72423          	sw	a2,8(a4)
   11878:	00f6e663          	bltu	a3,a5,11884 <_malloc_r+0x658>
   1187c:	004ba783          	lw	a5,4(s7)
   11880:	cb5ff06f          	j	11534 <_malloc_r+0x308>
   11884:	00840593          	addi	a1,s0,8
   11888:	00090513          	mv	a0,s2
   1188c:	e9cff0ef          	jal	10f28 <_free_r>
   11890:	0089ab83          	lw	s7,8(s3)
   11894:	000c2583          	lw	a1,0(s8)
   11898:	004ba783          	lw	a5,4(s7)
   1189c:	c99ff06f          	j	11534 <_malloc_r+0x308>
   118a0:	010a8a93          	addi	s5,s5,16
   118a4:	c41ff06f          	j	114e4 <_malloc_r+0x2b8>
   118a8:	4026d693          	srai	a3,a3,0x2
   118ac:	00100793          	li	a5,1
   118b0:	00d797b3          	sll	a5,a5,a3
   118b4:	00f56533          	or	a0,a0,a5
   118b8:	00a9a223          	sw	a0,4(s3)
   118bc:	d75ff06f          	j	11630 <_malloc_r+0x404>
   118c0:	015b85b3          	add	a1,s7,s5
   118c4:	40b005b3          	neg	a1,a1
   118c8:	01459593          	slli	a1,a1,0x14
   118cc:	0145da13          	srli	s4,a1,0x14
   118d0:	000a0593          	mv	a1,s4
   118d4:	00090513          	mv	a0,s2
   118d8:	568000ef          	jal	11e40 <_sbrk_r>
   118dc:	fff00793          	li	a5,-1
   118e0:	f4f512e3          	bne	a0,a5,11824 <_malloc_r+0x5f8>
   118e4:	00000a13          	li	s4,0
   118e8:	f45ff06f          	j	1182c <_malloc_r+0x600>
   118ec:	00812c03          	lw	s8,8(sp)
   118f0:	00100793          	li	a5,1
   118f4:	00fba223          	sw	a5,4(s7)
   118f8:	00090513          	mv	a0,s2
   118fc:	0fc000ef          	jal	119f8 <__malloc_unlock>
   11900:	00000513          	li	a0,0
   11904:	01812a03          	lw	s4,24(sp)
   11908:	01412a83          	lw	s5,20(sp)
   1190c:	01012b03          	lw	s6,16(sp)
   11910:	00c12b83          	lw	s7,12(sp)
   11914:	00412c83          	lw	s9,4(sp)
   11918:	9a1ff06f          	j	112b8 <_malloc_r+0x8c>
   1191c:	05400693          	li	a3,84
   11920:	04e6e463          	bltu	a3,a4,11968 <_malloc_r+0x73c>
   11924:	00c7d713          	srli	a4,a5,0xc
   11928:	06f70613          	addi	a2,a4,111
   1192c:	06e70693          	addi	a3,a4,110
   11930:	00361613          	slli	a2,a2,0x3
   11934:	cd1ff06f          	j	11604 <_malloc_r+0x3d8>
   11938:	15400713          	li	a4,340
   1193c:	04f76463          	bltu	a4,a5,11984 <_malloc_r+0x758>
   11940:	00f4d793          	srli	a5,s1,0xf
   11944:	07878593          	addi	a1,a5,120
   11948:	07778813          	addi	a6,a5,119
   1194c:	00359613          	slli	a2,a1,0x3
   11950:	9bdff06f          	j	1130c <_malloc_r+0xe0>
   11954:	0089a403          	lw	s0,8(s3)
   11958:	00442783          	lw	a5,4(s0)
   1195c:	c01ff06f          	j	1155c <_malloc_r+0x330>
   11960:	017ca023          	sw	s7,0(s9)
   11964:	e81ff06f          	j	117e4 <_malloc_r+0x5b8>
   11968:	15400693          	li	a3,340
   1196c:	04e6e463          	bltu	a3,a4,119b4 <_malloc_r+0x788>
   11970:	00f7d713          	srli	a4,a5,0xf
   11974:	07870613          	addi	a2,a4,120
   11978:	07770693          	addi	a3,a4,119
   1197c:	00361613          	slli	a2,a2,0x3
   11980:	c85ff06f          	j	11604 <_malloc_r+0x3d8>
   11984:	55400713          	li	a4,1364
   11988:	04f76463          	bltu	a4,a5,119d0 <_malloc_r+0x7a4>
   1198c:	0124d793          	srli	a5,s1,0x12
   11990:	07d78593          	addi	a1,a5,125
   11994:	07c78813          	addi	a6,a5,124
   11998:	00359613          	slli	a2,a1,0x3
   1199c:	971ff06f          	j	1130c <_malloc_r+0xe0>
   119a0:	ff8c8c93          	addi	s9,s9,-8
   119a4:	019a8ab3          	add	s5,s5,s9
   119a8:	417a8ab3          	sub	s5,s5,s7
   119ac:	00000a13          	li	s4,0
   119b0:	e7dff06f          	j	1182c <_malloc_r+0x600>
   119b4:	55400693          	li	a3,1364
   119b8:	02e6e463          	bltu	a3,a4,119e0 <_malloc_r+0x7b4>
   119bc:	0127d713          	srli	a4,a5,0x12
   119c0:	07d70613          	addi	a2,a4,125
   119c4:	07c70693          	addi	a3,a4,124
   119c8:	00361613          	slli	a2,a2,0x3
   119cc:	c39ff06f          	j	11604 <_malloc_r+0x3d8>
   119d0:	3f800613          	li	a2,1016
   119d4:	07f00593          	li	a1,127
   119d8:	07e00813          	li	a6,126
   119dc:	931ff06f          	j	1130c <_malloc_r+0xe0>
   119e0:	3f800613          	li	a2,1016
   119e4:	07e00693          	li	a3,126
   119e8:	c1dff06f          	j	11604 <_malloc_r+0x3d8>
   119ec:	0049a783          	lw	a5,4(s3)
   119f0:	da1ff06f          	j	11790 <_malloc_r+0x564>

000119f4 <__malloc_lock>:
   119f4:	00008067          	ret

000119f8 <__malloc_unlock>:
   119f8:	00008067          	ret

000119fc <_fclose_r>:
   119fc:	ff010113          	addi	sp,sp,-16
   11a00:	00112623          	sw	ra,12(sp)
   11a04:	01212023          	sw	s2,0(sp)
   11a08:	02058863          	beqz	a1,11a38 <_fclose_r+0x3c>
   11a0c:	00812423          	sw	s0,8(sp)
   11a10:	00912223          	sw	s1,4(sp)
   11a14:	00058413          	mv	s0,a1
   11a18:	00050493          	mv	s1,a0
   11a1c:	00050663          	beqz	a0,11a28 <_fclose_r+0x2c>
   11a20:	03452783          	lw	a5,52(a0)
   11a24:	0c078c63          	beqz	a5,11afc <_fclose_r+0x100>
   11a28:	00c41783          	lh	a5,12(s0)
   11a2c:	02079263          	bnez	a5,11a50 <_fclose_r+0x54>
   11a30:	00812403          	lw	s0,8(sp)
   11a34:	00412483          	lw	s1,4(sp)
   11a38:	00c12083          	lw	ra,12(sp)
   11a3c:	00000913          	li	s2,0
   11a40:	00090513          	mv	a0,s2
   11a44:	00012903          	lw	s2,0(sp)
   11a48:	01010113          	addi	sp,sp,16
   11a4c:	00008067          	ret
   11a50:	00040593          	mv	a1,s0
   11a54:	00048513          	mv	a0,s1
   11a58:	0b8000ef          	jal	11b10 <__sflush_r>
   11a5c:	02c42783          	lw	a5,44(s0)
   11a60:	00050913          	mv	s2,a0
   11a64:	00078a63          	beqz	a5,11a78 <_fclose_r+0x7c>
   11a68:	01c42583          	lw	a1,28(s0)
   11a6c:	00048513          	mv	a0,s1
   11a70:	000780e7          	jalr	a5
   11a74:	06054463          	bltz	a0,11adc <_fclose_r+0xe0>
   11a78:	00c45783          	lhu	a5,12(s0)
   11a7c:	0807f793          	andi	a5,a5,128
   11a80:	06079663          	bnez	a5,11aec <_fclose_r+0xf0>
   11a84:	03042583          	lw	a1,48(s0)
   11a88:	00058c63          	beqz	a1,11aa0 <_fclose_r+0xa4>
   11a8c:	04040793          	addi	a5,s0,64
   11a90:	00f58663          	beq	a1,a5,11a9c <_fclose_r+0xa0>
   11a94:	00048513          	mv	a0,s1
   11a98:	c90ff0ef          	jal	10f28 <_free_r>
   11a9c:	02042823          	sw	zero,48(s0)
   11aa0:	04442583          	lw	a1,68(s0)
   11aa4:	00058863          	beqz	a1,11ab4 <_fclose_r+0xb8>
   11aa8:	00048513          	mv	a0,s1
   11aac:	c7cff0ef          	jal	10f28 <_free_r>
   11ab0:	04042223          	sw	zero,68(s0)
   11ab4:	c01fe0ef          	jal	106b4 <__sfp_lock_acquire>
   11ab8:	00041623          	sh	zero,12(s0)
   11abc:	bfdfe0ef          	jal	106b8 <__sfp_lock_release>
   11ac0:	00c12083          	lw	ra,12(sp)
   11ac4:	00812403          	lw	s0,8(sp)
   11ac8:	00412483          	lw	s1,4(sp)
   11acc:	00090513          	mv	a0,s2
   11ad0:	00012903          	lw	s2,0(sp)
   11ad4:	01010113          	addi	sp,sp,16
   11ad8:	00008067          	ret
   11adc:	00c45783          	lhu	a5,12(s0)
   11ae0:	fff00913          	li	s2,-1
   11ae4:	0807f793          	andi	a5,a5,128
   11ae8:	f8078ee3          	beqz	a5,11a84 <_fclose_r+0x88>
   11aec:	01042583          	lw	a1,16(s0)
   11af0:	00048513          	mv	a0,s1
   11af4:	c34ff0ef          	jal	10f28 <_free_r>
   11af8:	f8dff06f          	j	11a84 <_fclose_r+0x88>
   11afc:	b95fe0ef          	jal	10690 <__sinit>
   11b00:	f29ff06f          	j	11a28 <_fclose_r+0x2c>

00011b04 <fclose>:
   11b04:	00050593          	mv	a1,a0
   11b08:	d8c1a503          	lw	a0,-628(gp) # 1368c <_impure_ptr>
   11b0c:	ef1ff06f          	j	119fc <_fclose_r>

00011b10 <__sflush_r>:
   11b10:	00c59703          	lh	a4,12(a1)
   11b14:	fe010113          	addi	sp,sp,-32
   11b18:	00812c23          	sw	s0,24(sp)
   11b1c:	01312623          	sw	s3,12(sp)
   11b20:	00112e23          	sw	ra,28(sp)
   11b24:	00877793          	andi	a5,a4,8
   11b28:	00058413          	mv	s0,a1
   11b2c:	00050993          	mv	s3,a0
   11b30:	12079063          	bnez	a5,11c50 <__sflush_r+0x140>
   11b34:	000017b7          	lui	a5,0x1
   11b38:	80078793          	addi	a5,a5,-2048 # 800 <exit-0xf8b4>
   11b3c:	0045a683          	lw	a3,4(a1)
   11b40:	00f767b3          	or	a5,a4,a5
   11b44:	00f59623          	sh	a5,12(a1)
   11b48:	18d05263          	blez	a3,11ccc <__sflush_r+0x1bc>
   11b4c:	02842803          	lw	a6,40(s0)
   11b50:	0e080463          	beqz	a6,11c38 <__sflush_r+0x128>
   11b54:	00912a23          	sw	s1,20(sp)
   11b58:	01371693          	slli	a3,a4,0x13
   11b5c:	0009a483          	lw	s1,0(s3)
   11b60:	0009a023          	sw	zero,0(s3)
   11b64:	01c42583          	lw	a1,28(s0)
   11b68:	1606ce63          	bltz	a3,11ce4 <__sflush_r+0x1d4>
   11b6c:	00000613          	li	a2,0
   11b70:	00100693          	li	a3,1
   11b74:	00098513          	mv	a0,s3
   11b78:	000800e7          	jalr	a6
   11b7c:	fff00793          	li	a5,-1
   11b80:	00050613          	mv	a2,a0
   11b84:	1af50463          	beq	a0,a5,11d2c <__sflush_r+0x21c>
   11b88:	00c41783          	lh	a5,12(s0)
   11b8c:	02842803          	lw	a6,40(s0)
   11b90:	01c42583          	lw	a1,28(s0)
   11b94:	0047f793          	andi	a5,a5,4
   11b98:	00078e63          	beqz	a5,11bb4 <__sflush_r+0xa4>
   11b9c:	00442703          	lw	a4,4(s0)
   11ba0:	03042783          	lw	a5,48(s0)
   11ba4:	40e60633          	sub	a2,a2,a4
   11ba8:	00078663          	beqz	a5,11bb4 <__sflush_r+0xa4>
   11bac:	03c42783          	lw	a5,60(s0)
   11bb0:	40f60633          	sub	a2,a2,a5
   11bb4:	00000693          	li	a3,0
   11bb8:	00098513          	mv	a0,s3
   11bbc:	000800e7          	jalr	a6
   11bc0:	fff00793          	li	a5,-1
   11bc4:	12f51463          	bne	a0,a5,11cec <__sflush_r+0x1dc>
   11bc8:	0009a683          	lw	a3,0(s3)
   11bcc:	01d00793          	li	a5,29
   11bd0:	00c41703          	lh	a4,12(s0)
   11bd4:	16d7ea63          	bltu	a5,a3,11d48 <__sflush_r+0x238>
   11bd8:	204007b7          	lui	a5,0x20400
   11bdc:	00178793          	addi	a5,a5,1 # 20400001 <__BSS_END__+0x203ec641>
   11be0:	00d7d7b3          	srl	a5,a5,a3
   11be4:	0017f793          	andi	a5,a5,1
   11be8:	16078063          	beqz	a5,11d48 <__sflush_r+0x238>
   11bec:	01042603          	lw	a2,16(s0)
   11bf0:	fffff7b7          	lui	a5,0xfffff
   11bf4:	7ff78793          	addi	a5,a5,2047 # fffff7ff <__BSS_END__+0xfffebe3f>
   11bf8:	00f777b3          	and	a5,a4,a5
   11bfc:	00f41623          	sh	a5,12(s0)
   11c00:	00042223          	sw	zero,4(s0)
   11c04:	00c42023          	sw	a2,0(s0)
   11c08:	01371793          	slli	a5,a4,0x13
   11c0c:	0007d463          	bgez	a5,11c14 <__sflush_r+0x104>
   11c10:	10068263          	beqz	a3,11d14 <__sflush_r+0x204>
   11c14:	03042583          	lw	a1,48(s0)
   11c18:	0099a023          	sw	s1,0(s3)
   11c1c:	10058463          	beqz	a1,11d24 <__sflush_r+0x214>
   11c20:	04040793          	addi	a5,s0,64
   11c24:	00f58663          	beq	a1,a5,11c30 <__sflush_r+0x120>
   11c28:	00098513          	mv	a0,s3
   11c2c:	afcff0ef          	jal	10f28 <_free_r>
   11c30:	01412483          	lw	s1,20(sp)
   11c34:	02042823          	sw	zero,48(s0)
   11c38:	00000513          	li	a0,0
   11c3c:	01c12083          	lw	ra,28(sp)
   11c40:	01812403          	lw	s0,24(sp)
   11c44:	00c12983          	lw	s3,12(sp)
   11c48:	02010113          	addi	sp,sp,32
   11c4c:	00008067          	ret
   11c50:	01212823          	sw	s2,16(sp)
   11c54:	0105a903          	lw	s2,16(a1)
   11c58:	08090263          	beqz	s2,11cdc <__sflush_r+0x1cc>
   11c5c:	00912a23          	sw	s1,20(sp)
   11c60:	0005a483          	lw	s1,0(a1)
   11c64:	00377713          	andi	a4,a4,3
   11c68:	0125a023          	sw	s2,0(a1)
   11c6c:	412484b3          	sub	s1,s1,s2
   11c70:	00000793          	li	a5,0
   11c74:	00071463          	bnez	a4,11c7c <__sflush_r+0x16c>
   11c78:	0145a783          	lw	a5,20(a1)
   11c7c:	00f42423          	sw	a5,8(s0)
   11c80:	00904863          	bgtz	s1,11c90 <__sflush_r+0x180>
   11c84:	0540006f          	j	11cd8 <__sflush_r+0x1c8>
   11c88:	00a90933          	add	s2,s2,a0
   11c8c:	04905663          	blez	s1,11cd8 <__sflush_r+0x1c8>
   11c90:	02442783          	lw	a5,36(s0)
   11c94:	01c42583          	lw	a1,28(s0)
   11c98:	00048693          	mv	a3,s1
   11c9c:	00090613          	mv	a2,s2
   11ca0:	00098513          	mv	a0,s3
   11ca4:	000780e7          	jalr	a5
   11ca8:	40a484b3          	sub	s1,s1,a0
   11cac:	fca04ee3          	bgtz	a0,11c88 <__sflush_r+0x178>
   11cb0:	00c41703          	lh	a4,12(s0)
   11cb4:	01012903          	lw	s2,16(sp)
   11cb8:	04076713          	ori	a4,a4,64
   11cbc:	01412483          	lw	s1,20(sp)
   11cc0:	00e41623          	sh	a4,12(s0)
   11cc4:	fff00513          	li	a0,-1
   11cc8:	f75ff06f          	j	11c3c <__sflush_r+0x12c>
   11ccc:	03c5a683          	lw	a3,60(a1)
   11cd0:	e6d04ee3          	bgtz	a3,11b4c <__sflush_r+0x3c>
   11cd4:	f65ff06f          	j	11c38 <__sflush_r+0x128>
   11cd8:	01412483          	lw	s1,20(sp)
   11cdc:	01012903          	lw	s2,16(sp)
   11ce0:	f59ff06f          	j	11c38 <__sflush_r+0x128>
   11ce4:	05042603          	lw	a2,80(s0)
   11ce8:	eadff06f          	j	11b94 <__sflush_r+0x84>
   11cec:	00c41703          	lh	a4,12(s0)
   11cf0:	01042683          	lw	a3,16(s0)
   11cf4:	fffff7b7          	lui	a5,0xfffff
   11cf8:	7ff78793          	addi	a5,a5,2047 # fffff7ff <__BSS_END__+0xfffebe3f>
   11cfc:	00f777b3          	and	a5,a4,a5
   11d00:	00f41623          	sh	a5,12(s0)
   11d04:	00042223          	sw	zero,4(s0)
   11d08:	00d42023          	sw	a3,0(s0)
   11d0c:	01371793          	slli	a5,a4,0x13
   11d10:	f007d2e3          	bgez	a5,11c14 <__sflush_r+0x104>
   11d14:	03042583          	lw	a1,48(s0)
   11d18:	04a42823          	sw	a0,80(s0)
   11d1c:	0099a023          	sw	s1,0(s3)
   11d20:	f00590e3          	bnez	a1,11c20 <__sflush_r+0x110>
   11d24:	01412483          	lw	s1,20(sp)
   11d28:	f11ff06f          	j	11c38 <__sflush_r+0x128>
   11d2c:	0009a783          	lw	a5,0(s3)
   11d30:	e4078ce3          	beqz	a5,11b88 <__sflush_r+0x78>
   11d34:	01d00713          	li	a4,29
   11d38:	00e78c63          	beq	a5,a4,11d50 <__sflush_r+0x240>
   11d3c:	01600713          	li	a4,22
   11d40:	00e78863          	beq	a5,a4,11d50 <__sflush_r+0x240>
   11d44:	00c41703          	lh	a4,12(s0)
   11d48:	04076713          	ori	a4,a4,64
   11d4c:	f71ff06f          	j	11cbc <__sflush_r+0x1ac>
   11d50:	0099a023          	sw	s1,0(s3)
   11d54:	01412483          	lw	s1,20(sp)
   11d58:	ee1ff06f          	j	11c38 <__sflush_r+0x128>

00011d5c <_fflush_r>:
   11d5c:	fe010113          	addi	sp,sp,-32
   11d60:	00812c23          	sw	s0,24(sp)
   11d64:	00112e23          	sw	ra,28(sp)
   11d68:	00050413          	mv	s0,a0
   11d6c:	00050663          	beqz	a0,11d78 <_fflush_r+0x1c>
   11d70:	03452783          	lw	a5,52(a0)
   11d74:	02078a63          	beqz	a5,11da8 <_fflush_r+0x4c>
   11d78:	00c59783          	lh	a5,12(a1)
   11d7c:	00079c63          	bnez	a5,11d94 <_fflush_r+0x38>
   11d80:	01c12083          	lw	ra,28(sp)
   11d84:	01812403          	lw	s0,24(sp)
   11d88:	00000513          	li	a0,0
   11d8c:	02010113          	addi	sp,sp,32
   11d90:	00008067          	ret
   11d94:	00040513          	mv	a0,s0
   11d98:	01812403          	lw	s0,24(sp)
   11d9c:	01c12083          	lw	ra,28(sp)
   11da0:	02010113          	addi	sp,sp,32
   11da4:	d6dff06f          	j	11b10 <__sflush_r>
   11da8:	00b12623          	sw	a1,12(sp)
   11dac:	8e5fe0ef          	jal	10690 <__sinit>
   11db0:	00c12583          	lw	a1,12(sp)
   11db4:	fc5ff06f          	j	11d78 <_fflush_r+0x1c>

00011db8 <fflush>:
   11db8:	06050063          	beqz	a0,11e18 <fflush+0x60>
   11dbc:	00050593          	mv	a1,a0
   11dc0:	d8c1a503          	lw	a0,-628(gp) # 1368c <_impure_ptr>
   11dc4:	00050663          	beqz	a0,11dd0 <fflush+0x18>
   11dc8:	03452783          	lw	a5,52(a0)
   11dcc:	00078c63          	beqz	a5,11de4 <fflush+0x2c>
   11dd0:	00c59783          	lh	a5,12(a1)
   11dd4:	00079663          	bnez	a5,11de0 <fflush+0x28>
   11dd8:	00000513          	li	a0,0
   11ddc:	00008067          	ret
   11de0:	d31ff06f          	j	11b10 <__sflush_r>
   11de4:	fe010113          	addi	sp,sp,-32
   11de8:	00b12623          	sw	a1,12(sp)
   11dec:	00a12423          	sw	a0,8(sp)
   11df0:	00112e23          	sw	ra,28(sp)
   11df4:	89dfe0ef          	jal	10690 <__sinit>
   11df8:	00c12583          	lw	a1,12(sp)
   11dfc:	00812503          	lw	a0,8(sp)
   11e00:	00c59783          	lh	a5,12(a1)
   11e04:	02079863          	bnez	a5,11e34 <fflush+0x7c>
   11e08:	01c12083          	lw	ra,28(sp)
   11e0c:	00000513          	li	a0,0
   11e10:	02010113          	addi	sp,sp,32
   11e14:	00008067          	ret
   11e18:	00001617          	auipc	a2,0x1
   11e1c:	33860613          	addi	a2,a2,824 # 13150 <__sglue>
   11e20:	00000597          	auipc	a1,0x0
   11e24:	f3c58593          	addi	a1,a1,-196 # 11d5c <_fflush_r>
   11e28:	00001517          	auipc	a0,0x1
   11e2c:	33850513          	addi	a0,a0,824 # 13160 <_impure_data>
   11e30:	8bdfe06f          	j	106ec <_fwalk_sglue>
   11e34:	01c12083          	lw	ra,28(sp)
   11e38:	02010113          	addi	sp,sp,32
   11e3c:	cd5ff06f          	j	11b10 <__sflush_r>

00011e40 <_sbrk_r>:
   11e40:	ff010113          	addi	sp,sp,-16
   11e44:	00812423          	sw	s0,8(sp)
   11e48:	00050413          	mv	s0,a0
   11e4c:	00058513          	mv	a0,a1
   11e50:	d801ae23          	sw	zero,-612(gp) # 1369c <errno>
   11e54:	00112623          	sw	ra,12(sp)
   11e58:	15c000ef          	jal	11fb4 <_sbrk>
   11e5c:	fff00793          	li	a5,-1
   11e60:	00f50a63          	beq	a0,a5,11e74 <_sbrk_r+0x34>
   11e64:	00c12083          	lw	ra,12(sp)
   11e68:	00812403          	lw	s0,8(sp)
   11e6c:	01010113          	addi	sp,sp,16
   11e70:	00008067          	ret
   11e74:	d9c1a783          	lw	a5,-612(gp) # 1369c <errno>
   11e78:	fe0786e3          	beqz	a5,11e64 <_sbrk_r+0x24>
   11e7c:	00c12083          	lw	ra,12(sp)
   11e80:	00f42023          	sw	a5,0(s0)
   11e84:	00812403          	lw	s0,8(sp)
   11e88:	01010113          	addi	sp,sp,16
   11e8c:	00008067          	ret

00011e90 <__libc_fini_array>:
   11e90:	ff010113          	addi	sp,sp,-16
   11e94:	00812423          	sw	s0,8(sp)
   11e98:	00001797          	auipc	a5,0x1
   11e9c:	20478793          	addi	a5,a5,516 # 1309c <__do_global_dtors_aux_fini_array_entry>
   11ea0:	00001417          	auipc	s0,0x1
   11ea4:	20040413          	addi	s0,s0,512 # 130a0 <__fini_array_end>
   11ea8:	40f40433          	sub	s0,s0,a5
   11eac:	00912223          	sw	s1,4(sp)
   11eb0:	00112623          	sw	ra,12(sp)
   11eb4:	40245493          	srai	s1,s0,0x2
   11eb8:	02048063          	beqz	s1,11ed8 <__libc_fini_array+0x48>
   11ebc:	ffc40413          	addi	s0,s0,-4
   11ec0:	00f40433          	add	s0,s0,a5
   11ec4:	00042783          	lw	a5,0(s0)
   11ec8:	fff48493          	addi	s1,s1,-1
   11ecc:	ffc40413          	addi	s0,s0,-4
   11ed0:	000780e7          	jalr	a5
   11ed4:	fe0498e3          	bnez	s1,11ec4 <__libc_fini_array+0x34>
   11ed8:	00c12083          	lw	ra,12(sp)
   11edc:	00812403          	lw	s0,8(sp)
   11ee0:	00412483          	lw	s1,4(sp)
   11ee4:	01010113          	addi	sp,sp,16
   11ee8:	00008067          	ret

00011eec <__register_exitproc>:
   11eec:	da018713          	addi	a4,gp,-608 # 136a0 <__atexit>
   11ef0:	00072783          	lw	a5,0(a4)
   11ef4:	04078c63          	beqz	a5,11f4c <__register_exitproc+0x60>
   11ef8:	0047a703          	lw	a4,4(a5)
   11efc:	01f00813          	li	a6,31
   11f00:	06e84e63          	blt	a6,a4,11f7c <__register_exitproc+0x90>
   11f04:	00271813          	slli	a6,a4,0x2
   11f08:	02050663          	beqz	a0,11f34 <__register_exitproc+0x48>
   11f0c:	01078333          	add	t1,a5,a6
   11f10:	08c32423          	sw	a2,136(t1)
   11f14:	1887a883          	lw	a7,392(a5)
   11f18:	00100613          	li	a2,1
   11f1c:	00e61633          	sll	a2,a2,a4
   11f20:	00c8e8b3          	or	a7,a7,a2
   11f24:	1917a423          	sw	a7,392(a5)
   11f28:	10d32423          	sw	a3,264(t1)
   11f2c:	00200693          	li	a3,2
   11f30:	02d50463          	beq	a0,a3,11f58 <__register_exitproc+0x6c>
   11f34:	00170713          	addi	a4,a4,1
   11f38:	00e7a223          	sw	a4,4(a5)
   11f3c:	010787b3          	add	a5,a5,a6
   11f40:	00b7a423          	sw	a1,8(a5)
   11f44:	00000513          	li	a0,0
   11f48:	00008067          	ret
   11f4c:	f3018793          	addi	a5,gp,-208 # 13830 <__atexit0>
   11f50:	00f72023          	sw	a5,0(a4)
   11f54:	fa5ff06f          	j	11ef8 <__register_exitproc+0xc>
   11f58:	18c7a683          	lw	a3,396(a5)
   11f5c:	00170713          	addi	a4,a4,1
   11f60:	00e7a223          	sw	a4,4(a5)
   11f64:	00c6e6b3          	or	a3,a3,a2
   11f68:	18d7a623          	sw	a3,396(a5)
   11f6c:	010787b3          	add	a5,a5,a6
   11f70:	00b7a423          	sw	a1,8(a5)
   11f74:	00000513          	li	a0,0
   11f78:	00008067          	ret
   11f7c:	fff00513          	li	a0,-1
   11f80:	00008067          	ret

00011f84 <_close>:
   11f84:	05800793          	li	a5,88
   11f88:	d8f1ae23          	sw	a5,-612(gp) # 1369c <errno>
   11f8c:	fff00513          	li	a0,-1
   11f90:	00008067          	ret

00011f94 <_lseek>:
   11f94:	05800793          	li	a5,88
   11f98:	d8f1ae23          	sw	a5,-612(gp) # 1369c <errno>
   11f9c:	fff00513          	li	a0,-1
   11fa0:	00008067          	ret

00011fa4 <_read>:
   11fa4:	05800793          	li	a5,88
   11fa8:	d8f1ae23          	sw	a5,-612(gp) # 1369c <errno>
   11fac:	fff00513          	li	a0,-1
   11fb0:	00008067          	ret

00011fb4 <_sbrk>:
   11fb4:	db018713          	addi	a4,gp,-592 # 136b0 <heap_end.0>
   11fb8:	00072783          	lw	a5,0(a4)
   11fbc:	00078a63          	beqz	a5,11fd0 <_sbrk+0x1c>
   11fc0:	00a78533          	add	a0,a5,a0
   11fc4:	00a72023          	sw	a0,0(a4)
   11fc8:	00078513          	mv	a0,a5
   11fcc:	00008067          	ret
   11fd0:	0c018793          	addi	a5,gp,192 # 139c0 <__BSS_END__>
   11fd4:	00a78533          	add	a0,a5,a0
   11fd8:	00a72023          	sw	a0,0(a4)
   11fdc:	00078513          	mv	a0,a5
   11fe0:	00008067          	ret

00011fe4 <_write>:
   11fe4:	05800793          	li	a5,88
   11fe8:	d8f1ae23          	sw	a5,-612(gp) # 1369c <errno>
   11fec:	fff00513          	li	a0,-1
   11ff0:	00008067          	ret

00011ff4 <_exit>:
   11ff4:	0000006f          	j	11ff4 <_exit>

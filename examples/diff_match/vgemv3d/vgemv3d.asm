
/home/cloud/aps-mlir/examples/diff_match/vgemv3d/vgemv3d.out:     file format elf32-littleriscv


Disassembly of section .text:

000100b4 <exit>:
   100b4:	ff010113          	addi	sp,sp,-16
   100b8:	00000593          	li	a1,0
   100bc:	00812423          	sw	s0,8(sp)
   100c0:	00112623          	sw	ra,12(sp)
   100c4:	00050413          	mv	s0,a0
   100c8:	5a5000ef          	jal	10e6c <__call_exitprocs>
   100cc:	f681a783          	lw	a5,-152(gp) # 229e8 <__stdio_exit_handler>
   100d0:	00078463          	beqz	a5,100d8 <exit+0x24>
   100d4:	000780e7          	jalr	a5
   100d8:	00040513          	mv	a0,s0
   100dc:	75c100ef          	jal	20838 <_exit>

000100e0 <register_fini>:
   100e0:	00000793          	li	a5,0
   100e4:	00078863          	beqz	a5,100f4 <register_fini+0x14>
   100e8:	00007517          	auipc	a0,0x7
   100ec:	92850513          	addi	a0,a0,-1752 # 16a10 <__libc_fini_array>
   100f0:	6b50006f          	j	10fa4 <atexit>
   100f4:	00008067          	ret

000100f8 <_start>:
   100f8:	00013197          	auipc	gp,0x13
   100fc:	98818193          	addi	gp,gp,-1656 # 22a80 <completed.1>
   10100:	f6818513          	addi	a0,gp,-152 # 229e8 <__stdio_exit_handler>
   10104:	38018613          	addi	a2,gp,896 # 22e00 <__BSS_END__>
   10108:	40a60633          	sub	a2,a2,a0
   1010c:	00000593          	li	a1,0
   10110:	3f5000ef          	jal	10d04 <memset>
   10114:	00001517          	auipc	a0,0x1
   10118:	e9050513          	addi	a0,a0,-368 # 10fa4 <atexit>
   1011c:	00050863          	beqz	a0,1012c <_start+0x34>
   10120:	00007517          	auipc	a0,0x7
   10124:	8f050513          	addi	a0,a0,-1808 # 16a10 <__libc_fini_array>
   10128:	67d000ef          	jal	10fa4 <atexit>
   1012c:	345000ef          	jal	10c70 <__libc_init_array>
   10130:	00012503          	lw	a0,0(sp)
   10134:	00410593          	addi	a1,sp,4
   10138:	00000613          	li	a2,0
   1013c:	0ac000ef          	jal	101e8 <main>
   10140:	f75ff06f          	j	100b4 <exit>

00010144 <__do_global_dtors_aux>:
   10144:	ff010113          	addi	sp,sp,-16
   10148:	00812423          	sw	s0,8(sp)
   1014c:	00018413          	mv	s0,gp
   10150:	00044783          	lbu	a5,0(s0)
   10154:	00112623          	sw	ra,12(sp)
   10158:	02079263          	bnez	a5,1017c <__do_global_dtors_aux+0x38>
   1015c:	00000793          	li	a5,0
   10160:	00078a63          	beqz	a5,10174 <__do_global_dtors_aux+0x30>
   10164:	00012517          	auipc	a0,0x12
   10168:	09450513          	addi	a0,a0,148 # 221f8 <__EH_FRAME_BEGIN__>
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
   10194:	00418593          	addi	a1,gp,4 # 22a84 <object.0>
   10198:	00012517          	auipc	a0,0x12
   1019c:	06050513          	addi	a0,a0,96 # 221f8 <__EH_FRAME_BEGIN__>
   101a0:	00000317          	auipc	t1,0x0
   101a4:	00000067          	jr	zero # 0 <exit-0x100b4>
   101a8:	00008067          	ret

000101ac <vgemv3d_vv>:
   101ac:	62b5752b          	.insn	4, 0x62b5752b
   101b0:	00000513          	li	a0,0
   101b4:	00008067          	ret

000101b8 <get_march>:
   101b8:	fff50513          	addi	a0,a0,-1
   101bc:	00400593          	li	a1,4
   101c0:	00a5ee63          	bltu	a1,a0,101dc <get_march+0x24>
   101c4:	00251513          	slli	a0,a0,0x2
   101c8:	00010597          	auipc	a1,0x10
   101cc:	73c58593          	addi	a1,a1,1852 # 20904 <_exit+0xcc>
   101d0:	00a58533          	add	a0,a1,a0
   101d4:	00052503          	lw	a0,0(a0)
   101d8:	00008067          	ret
   101dc:	00010517          	auipc	a0,0x10
   101e0:	67b50513          	addi	a0,a0,1659 # 20857 <_exit+0x1f>
   101e4:	00008067          	ret

000101e8 <main>:
   101e8:	ff010113          	addi	sp,sp,-16
   101ec:	00112623          	sw	ra,12(sp)
   101f0:	00812423          	sw	s0,8(sp)
   101f4:	00912223          	sw	s1,4(sp)
   101f8:	00010517          	auipc	a0,0x10
   101fc:	6dc50513          	addi	a0,a0,1756 # 208d4 <_exit+0x9c>
   10200:	6d0000ef          	jal	108d0 <puts>
   10204:	00010517          	auipc	a0,0x10
   10208:	00012597          	auipc	a1,0x12
   1020c:	65b50513          	addi	a0,a0,1627 # 2085f <_exit+0x27>
   10210:	07858593          	addi	a1,a1,120 # 22280 <input_data>
   10214:	5a0000ef          	jal	107b4 <printf>
   10218:	00010517          	auipc	a0,0x10
   1021c:	66250513          	addi	a0,a0,1634 # 2087a <_exit+0x42>
   10220:	08018413          	addi	s0,gp,128 # 22b00 <output_data>
   10224:	00040593          	mv	a1,s0
   10228:	58c000ef          	jal	107b4 <printf>
   1022c:	f1202573          	.insn	4, 0xf1202573
   10230:	fff50513          	addi	a0,a0,-1
   10234:	00400593          	li	a1,4
   10238:	00a5ee63          	bltu	a1,a0,10254 <main+0x6c>
   1023c:	00251513          	slli	a0,a0,0x2
   10240:	00010597          	auipc	a1,0x10
   10244:	6c458593          	addi	a1,a1,1732 # 20904 <_exit+0xcc>
   10248:	00a58533          	add	a0,a1,a0
   1024c:	00052583          	lw	a1,0(a0)
   10250:	00c0006f          	j	1025c <main+0x74>
   10254:	00010597          	auipc	a1,0x10
   10258:	60358593          	addi	a1,a1,1539 # 20857 <_exit+0x1f>
   1025c:	00010517          	auipc	a0,0x10
   10260:	63a50513          	addi	a0,a0,1594 # 20896 <_exit+0x5e>
   10264:	550000ef          	jal	107b4 <printf>
   10268:	00012023          	sw	zero,0(sp)
   1026c:	00012497          	auipc	s1,0x12
   10270:	01448493          	addi	s1,s1,20 # 22280 <input_data>
   10274:	00048513          	mv	a0,s1
   10278:	00040593          	mv	a1,s0
   1027c:	f31ff0ef          	jal	101ac <vgemv3d_vv>
   10280:	00a12023          	sw	a0,0(sp)
   10284:	00048513          	mv	a0,s1
   10288:	00040593          	mv	a1,s0
   1028c:	f21ff0ef          	jal	101ac <vgemv3d_vv>
   10290:	00a12023          	sw	a0,0(sp)
   10294:	00048513          	mv	a0,s1
   10298:	00040593          	mv	a1,s0
   1029c:	f11ff0ef          	jal	101ac <vgemv3d_vv>
   102a0:	00a12023          	sw	a0,0(sp)
   102a4:	00048513          	mv	a0,s1
   102a8:	00040593          	mv	a1,s0
   102ac:	f01ff0ef          	jal	101ac <vgemv3d_vv>
   102b0:	00a12023          	sw	a0,0(sp)
   102b4:	00048513          	mv	a0,s1
   102b8:	00040593          	mv	a1,s0
   102bc:	ef1ff0ef          	jal	101ac <vgemv3d_vv>
   102c0:	00a12023          	sw	a0,0(sp)
   102c4:	00048513          	mv	a0,s1
   102c8:	00040593          	mv	a1,s0
   102cc:	ee1ff0ef          	jal	101ac <vgemv3d_vv>
   102d0:	00a12023          	sw	a0,0(sp)
   102d4:	00048513          	mv	a0,s1
   102d8:	00040593          	mv	a1,s0
   102dc:	ed1ff0ef          	jal	101ac <vgemv3d_vv>
   102e0:	00a12023          	sw	a0,0(sp)
   102e4:	00048513          	mv	a0,s1
   102e8:	00040593          	mv	a1,s0
   102ec:	ec1ff0ef          	jal	101ac <vgemv3d_vv>
   102f0:	00a12023          	sw	a0,0(sp)
   102f4:	00048513          	mv	a0,s1
   102f8:	00040593          	mv	a1,s0
   102fc:	eb1ff0ef          	jal	101ac <vgemv3d_vv>
   10300:	00a12023          	sw	a0,0(sp)
   10304:	00048513          	mv	a0,s1
   10308:	00040593          	mv	a1,s0
   1030c:	ea1ff0ef          	jal	101ac <vgemv3d_vv>
   10310:	00a12023          	sw	a0,0(sp)
   10314:	00012583          	lw	a1,0(sp)
   10318:	00010517          	auipc	a0,0x10
   1031c:	58f50513          	addi	a0,a0,1423 # 208a7 <_exit+0x6f>
   10320:	494000ef          	jal	107b4 <printf>
   10324:	00042583          	lw	a1,0(s0)
   10328:	00442603          	lw	a2,4(s0)
   1032c:	00842683          	lw	a3,8(s0)
   10330:	00c42703          	lw	a4,12(s0)
   10334:	00010517          	auipc	a0,0x10
   10338:	57f50513          	addi	a0,a0,1407 # 208b3 <_exit+0x7b>
   1033c:	478000ef          	jal	107b4 <printf>
   10340:	00000513          	li	a0,0
   10344:	00c12083          	lw	ra,12(sp)
   10348:	00812403          	lw	s0,8(sp)
   1034c:	00412483          	lw	s1,4(sp)
   10350:	01010113          	addi	sp,sp,16
   10354:	00008067          	ret

00010358 <__fp_lock>:
   10358:	00000513          	li	a0,0
   1035c:	00008067          	ret

00010360 <stdio_exit_handler>:
   10360:	00012617          	auipc	a2,0x12
   10364:	f7060613          	addi	a2,a2,-144 # 222d0 <__sglue>
   10368:	00005597          	auipc	a1,0x5
   1036c:	7a058593          	addi	a1,a1,1952 # 15b08 <_fclose_r>
   10370:	00012517          	auipc	a0,0x12
   10374:	f7050513          	addi	a0,a0,-144 # 222e0 <_impure_data>
   10378:	3480006f          	j	106c0 <_fwalk_sglue>

0001037c <cleanup_stdio>:
   1037c:	00452583          	lw	a1,4(a0)
   10380:	ff010113          	addi	sp,sp,-16
   10384:	00812423          	sw	s0,8(sp)
   10388:	00112623          	sw	ra,12(sp)
   1038c:	09018793          	addi	a5,gp,144 # 22b10 <__sf>
   10390:	00050413          	mv	s0,a0
   10394:	00f58463          	beq	a1,a5,1039c <cleanup_stdio+0x20>
   10398:	770050ef          	jal	15b08 <_fclose_r>
   1039c:	00842583          	lw	a1,8(s0)
   103a0:	0f818793          	addi	a5,gp,248 # 22b78 <__sf+0x68>
   103a4:	00f58663          	beq	a1,a5,103b0 <cleanup_stdio+0x34>
   103a8:	00040513          	mv	a0,s0
   103ac:	75c050ef          	jal	15b08 <_fclose_r>
   103b0:	00c42583          	lw	a1,12(s0)
   103b4:	16018793          	addi	a5,gp,352 # 22be0 <__sf+0xd0>
   103b8:	00f58c63          	beq	a1,a5,103d0 <cleanup_stdio+0x54>
   103bc:	00040513          	mv	a0,s0
   103c0:	00812403          	lw	s0,8(sp)
   103c4:	00c12083          	lw	ra,12(sp)
   103c8:	01010113          	addi	sp,sp,16
   103cc:	73c0506f          	j	15b08 <_fclose_r>
   103d0:	00c12083          	lw	ra,12(sp)
   103d4:	00812403          	lw	s0,8(sp)
   103d8:	01010113          	addi	sp,sp,16
   103dc:	00008067          	ret

000103e0 <__fp_unlock>:
   103e0:	00000513          	li	a0,0
   103e4:	00008067          	ret

000103e8 <global_stdio_init.part.0>:
   103e8:	fe010113          	addi	sp,sp,-32
   103ec:	00000797          	auipc	a5,0x0
   103f0:	f7478793          	addi	a5,a5,-140 # 10360 <stdio_exit_handler>
   103f4:	00112e23          	sw	ra,28(sp)
   103f8:	00812c23          	sw	s0,24(sp)
   103fc:	00912a23          	sw	s1,20(sp)
   10400:	09018413          	addi	s0,gp,144 # 22b10 <__sf>
   10404:	01212823          	sw	s2,16(sp)
   10408:	01312623          	sw	s3,12(sp)
   1040c:	01412423          	sw	s4,8(sp)
   10410:	f6f1a423          	sw	a5,-152(gp) # 229e8 <__stdio_exit_handler>
   10414:	00800613          	li	a2,8
   10418:	00400793          	li	a5,4
   1041c:	00000593          	li	a1,0
   10420:	0ec18513          	addi	a0,gp,236 # 22b6c <__sf+0x5c>
   10424:	00f42623          	sw	a5,12(s0)
   10428:	00042023          	sw	zero,0(s0)
   1042c:	00042223          	sw	zero,4(s0)
   10430:	00042423          	sw	zero,8(s0)
   10434:	06042223          	sw	zero,100(s0)
   10438:	00042823          	sw	zero,16(s0)
   1043c:	00042a23          	sw	zero,20(s0)
   10440:	00042c23          	sw	zero,24(s0)
   10444:	0c1000ef          	jal	10d04 <memset>
   10448:	000107b7          	lui	a5,0x10
   1044c:	00000a17          	auipc	s4,0x0
   10450:	490a0a13          	addi	s4,s4,1168 # 108dc <__sread>
   10454:	00000997          	auipc	s3,0x0
   10458:	4ec98993          	addi	s3,s3,1260 # 10940 <__swrite>
   1045c:	00000917          	auipc	s2,0x0
   10460:	56c90913          	addi	s2,s2,1388 # 109c8 <__sseek>
   10464:	00000497          	auipc	s1,0x0
   10468:	5dc48493          	addi	s1,s1,1500 # 10a40 <__sclose>
   1046c:	00978793          	addi	a5,a5,9 # 10009 <exit-0xab>
   10470:	00800613          	li	a2,8
   10474:	00000593          	li	a1,0
   10478:	15418513          	addi	a0,gp,340 # 22bd4 <__sf+0xc4>
   1047c:	03442023          	sw	s4,32(s0)
   10480:	03342223          	sw	s3,36(s0)
   10484:	03242423          	sw	s2,40(s0)
   10488:	02942623          	sw	s1,44(s0)
   1048c:	06f42a23          	sw	a5,116(s0)
   10490:	00842e23          	sw	s0,28(s0)
   10494:	06042423          	sw	zero,104(s0)
   10498:	06042623          	sw	zero,108(s0)
   1049c:	06042823          	sw	zero,112(s0)
   104a0:	0c042623          	sw	zero,204(s0)
   104a4:	06042c23          	sw	zero,120(s0)
   104a8:	06042e23          	sw	zero,124(s0)
   104ac:	08042023          	sw	zero,128(s0)
   104b0:	055000ef          	jal	10d04 <memset>
   104b4:	000207b7          	lui	a5,0x20
   104b8:	01278793          	addi	a5,a5,18 # 20012 <__subtf3+0x124a>
   104bc:	0f818713          	addi	a4,gp,248 # 22b78 <__sf+0x68>
   104c0:	00800613          	li	a2,8
   104c4:	00000593          	li	a1,0
   104c8:	1bc18513          	addi	a0,gp,444 # 22c3c <__sf+0x12c>
   104cc:	09442423          	sw	s4,136(s0)
   104d0:	09342623          	sw	s3,140(s0)
   104d4:	09242823          	sw	s2,144(s0)
   104d8:	08942a23          	sw	s1,148(s0)
   104dc:	0cf42e23          	sw	a5,220(s0)
   104e0:	08e42223          	sw	a4,132(s0)
   104e4:	0c042823          	sw	zero,208(s0)
   104e8:	0c042a23          	sw	zero,212(s0)
   104ec:	0c042c23          	sw	zero,216(s0)
   104f0:	12042a23          	sw	zero,308(s0)
   104f4:	0e042023          	sw	zero,224(s0)
   104f8:	0e042223          	sw	zero,228(s0)
   104fc:	0e042423          	sw	zero,232(s0)
   10500:	005000ef          	jal	10d04 <memset>
   10504:	16018793          	addi	a5,gp,352 # 22be0 <__sf+0xd0>
   10508:	0f442823          	sw	s4,240(s0)
   1050c:	0f342a23          	sw	s3,244(s0)
   10510:	0f242c23          	sw	s2,248(s0)
   10514:	0e942e23          	sw	s1,252(s0)
   10518:	01c12083          	lw	ra,28(sp)
   1051c:	0ef42623          	sw	a5,236(s0)
   10520:	01812403          	lw	s0,24(sp)
   10524:	01412483          	lw	s1,20(sp)
   10528:	01012903          	lw	s2,16(sp)
   1052c:	00c12983          	lw	s3,12(sp)
   10530:	00812a03          	lw	s4,8(sp)
   10534:	02010113          	addi	sp,sp,32
   10538:	00008067          	ret

0001053c <__sfp>:
   1053c:	fe010113          	addi	sp,sp,-32
   10540:	01312623          	sw	s3,12(sp)
   10544:	00112e23          	sw	ra,28(sp)
   10548:	00812c23          	sw	s0,24(sp)
   1054c:	00912a23          	sw	s1,20(sp)
   10550:	01212823          	sw	s2,16(sp)
   10554:	f681a783          	lw	a5,-152(gp) # 229e8 <__stdio_exit_handler>
   10558:	00050993          	mv	s3,a0
   1055c:	0e078863          	beqz	a5,1064c <__sfp+0x110>
   10560:	00012917          	auipc	s2,0x12
   10564:	d7090913          	addi	s2,s2,-656 # 222d0 <__sglue>
   10568:	fff00493          	li	s1,-1
   1056c:	00492783          	lw	a5,4(s2)
   10570:	00892403          	lw	s0,8(s2)
   10574:	fff78793          	addi	a5,a5,-1
   10578:	0007d863          	bgez	a5,10588 <__sfp+0x4c>
   1057c:	0800006f          	j	105fc <__sfp+0xc0>
   10580:	06840413          	addi	s0,s0,104
   10584:	06978c63          	beq	a5,s1,105fc <__sfp+0xc0>
   10588:	00c41703          	lh	a4,12(s0)
   1058c:	fff78793          	addi	a5,a5,-1
   10590:	fe0718e3          	bnez	a4,10580 <__sfp+0x44>
   10594:	ffff07b7          	lui	a5,0xffff0
   10598:	00178793          	addi	a5,a5,1 # ffff0001 <__BSS_END__+0xfffcd201>
   1059c:	00f42623          	sw	a5,12(s0)
   105a0:	06042223          	sw	zero,100(s0)
   105a4:	00042023          	sw	zero,0(s0)
   105a8:	00042423          	sw	zero,8(s0)
   105ac:	00042223          	sw	zero,4(s0)
   105b0:	00042823          	sw	zero,16(s0)
   105b4:	00042a23          	sw	zero,20(s0)
   105b8:	00042c23          	sw	zero,24(s0)
   105bc:	00800613          	li	a2,8
   105c0:	00000593          	li	a1,0
   105c4:	05c40513          	addi	a0,s0,92
   105c8:	73c000ef          	jal	10d04 <memset>
   105cc:	02042823          	sw	zero,48(s0)
   105d0:	02042a23          	sw	zero,52(s0)
   105d4:	04042223          	sw	zero,68(s0)
   105d8:	04042423          	sw	zero,72(s0)
   105dc:	01c12083          	lw	ra,28(sp)
   105e0:	00040513          	mv	a0,s0
   105e4:	01812403          	lw	s0,24(sp)
   105e8:	01412483          	lw	s1,20(sp)
   105ec:	01012903          	lw	s2,16(sp)
   105f0:	00c12983          	lw	s3,12(sp)
   105f4:	02010113          	addi	sp,sp,32
   105f8:	00008067          	ret
   105fc:	00092403          	lw	s0,0(s2)
   10600:	00040663          	beqz	s0,1060c <__sfp+0xd0>
   10604:	00040913          	mv	s2,s0
   10608:	f65ff06f          	j	1056c <__sfp+0x30>
   1060c:	1ac00593          	li	a1,428
   10610:	00098513          	mv	a0,s3
   10614:	5e1000ef          	jal	113f4 <_malloc_r>
   10618:	00050413          	mv	s0,a0
   1061c:	02050c63          	beqz	a0,10654 <__sfp+0x118>
   10620:	00c50513          	addi	a0,a0,12
   10624:	00400793          	li	a5,4
   10628:	00042023          	sw	zero,0(s0)
   1062c:	00f42223          	sw	a5,4(s0)
   10630:	00a42423          	sw	a0,8(s0)
   10634:	1a000613          	li	a2,416
   10638:	00000593          	li	a1,0
   1063c:	6c8000ef          	jal	10d04 <memset>
   10640:	00892023          	sw	s0,0(s2)
   10644:	00040913          	mv	s2,s0
   10648:	f25ff06f          	j	1056c <__sfp+0x30>
   1064c:	d9dff0ef          	jal	103e8 <global_stdio_init.part.0>
   10650:	f11ff06f          	j	10560 <__sfp+0x24>
   10654:	00092023          	sw	zero,0(s2)
   10658:	00c00793          	li	a5,12
   1065c:	00f9a023          	sw	a5,0(s3)
   10660:	f7dff06f          	j	105dc <__sfp+0xa0>

00010664 <__sinit>:
   10664:	03452783          	lw	a5,52(a0)
   10668:	00078463          	beqz	a5,10670 <__sinit+0xc>
   1066c:	00008067          	ret
   10670:	00000797          	auipc	a5,0x0
   10674:	d0c78793          	addi	a5,a5,-756 # 1037c <cleanup_stdio>
   10678:	02f52a23          	sw	a5,52(a0)
   1067c:	f681a783          	lw	a5,-152(gp) # 229e8 <__stdio_exit_handler>
   10680:	fe0796e3          	bnez	a5,1066c <__sinit+0x8>
   10684:	d65ff06f          	j	103e8 <global_stdio_init.part.0>

00010688 <__sfp_lock_acquire>:
   10688:	00008067          	ret

0001068c <__sfp_lock_release>:
   1068c:	00008067          	ret

00010690 <__fp_lock_all>:
   10690:	00012617          	auipc	a2,0x12
   10694:	c4060613          	addi	a2,a2,-960 # 222d0 <__sglue>
   10698:	00000597          	auipc	a1,0x0
   1069c:	cc058593          	addi	a1,a1,-832 # 10358 <__fp_lock>
   106a0:	00000513          	li	a0,0
   106a4:	01c0006f          	j	106c0 <_fwalk_sglue>

000106a8 <__fp_unlock_all>:
   106a8:	00012617          	auipc	a2,0x12
   106ac:	c2860613          	addi	a2,a2,-984 # 222d0 <__sglue>
   106b0:	00000597          	auipc	a1,0x0
   106b4:	d3058593          	addi	a1,a1,-720 # 103e0 <__fp_unlock>
   106b8:	00000513          	li	a0,0
   106bc:	0040006f          	j	106c0 <_fwalk_sglue>

000106c0 <_fwalk_sglue>:
   106c0:	fd010113          	addi	sp,sp,-48
   106c4:	03212023          	sw	s2,32(sp)
   106c8:	01312e23          	sw	s3,28(sp)
   106cc:	01412c23          	sw	s4,24(sp)
   106d0:	01512a23          	sw	s5,20(sp)
   106d4:	01612823          	sw	s6,16(sp)
   106d8:	01712623          	sw	s7,12(sp)
   106dc:	02112623          	sw	ra,44(sp)
   106e0:	02812423          	sw	s0,40(sp)
   106e4:	02912223          	sw	s1,36(sp)
   106e8:	00050b13          	mv	s6,a0
   106ec:	00058b93          	mv	s7,a1
   106f0:	00060a93          	mv	s5,a2
   106f4:	00000a13          	li	s4,0
   106f8:	00100993          	li	s3,1
   106fc:	fff00913          	li	s2,-1
   10700:	004aa483          	lw	s1,4(s5)
   10704:	008aa403          	lw	s0,8(s5)
   10708:	fff48493          	addi	s1,s1,-1
   1070c:	0204c863          	bltz	s1,1073c <_fwalk_sglue+0x7c>
   10710:	00c45783          	lhu	a5,12(s0)
   10714:	00f9fe63          	bgeu	s3,a5,10730 <_fwalk_sglue+0x70>
   10718:	00e41783          	lh	a5,14(s0)
   1071c:	00040593          	mv	a1,s0
   10720:	000b0513          	mv	a0,s6
   10724:	01278663          	beq	a5,s2,10730 <_fwalk_sglue+0x70>
   10728:	000b80e7          	jalr	s7
   1072c:	00aa6a33          	or	s4,s4,a0
   10730:	fff48493          	addi	s1,s1,-1
   10734:	06840413          	addi	s0,s0,104
   10738:	fd249ce3          	bne	s1,s2,10710 <_fwalk_sglue+0x50>
   1073c:	000aaa83          	lw	s5,0(s5)
   10740:	fc0a90e3          	bnez	s5,10700 <_fwalk_sglue+0x40>
   10744:	02c12083          	lw	ra,44(sp)
   10748:	02812403          	lw	s0,40(sp)
   1074c:	02412483          	lw	s1,36(sp)
   10750:	02012903          	lw	s2,32(sp)
   10754:	01c12983          	lw	s3,28(sp)
   10758:	01412a83          	lw	s5,20(sp)
   1075c:	01012b03          	lw	s6,16(sp)
   10760:	00c12b83          	lw	s7,12(sp)
   10764:	000a0513          	mv	a0,s4
   10768:	01812a03          	lw	s4,24(sp)
   1076c:	03010113          	addi	sp,sp,48
   10770:	00008067          	ret

00010774 <_printf_r>:
   10774:	fc010113          	addi	sp,sp,-64
   10778:	02c12423          	sw	a2,40(sp)
   1077c:	02d12623          	sw	a3,44(sp)
   10780:	02e12823          	sw	a4,48(sp)
   10784:	02f12a23          	sw	a5,52(sp)
   10788:	03012c23          	sw	a6,56(sp)
   1078c:	03112e23          	sw	a7,60(sp)
   10790:	00058613          	mv	a2,a1
   10794:	00852583          	lw	a1,8(a0)
   10798:	02810693          	addi	a3,sp,40
   1079c:	00112e23          	sw	ra,28(sp)
   107a0:	00d12623          	sw	a3,12(sp)
   107a4:	420010ef          	jal	11bc4 <_vfprintf_r>
   107a8:	01c12083          	lw	ra,28(sp)
   107ac:	04010113          	addi	sp,sp,64
   107b0:	00008067          	ret

000107b4 <printf>:
   107b4:	fc010113          	addi	sp,sp,-64
   107b8:	02c12423          	sw	a2,40(sp)
   107bc:	02d12623          	sw	a3,44(sp)
   107c0:	f5c1a303          	lw	t1,-164(gp) # 229dc <_impure_ptr>
   107c4:	02b12223          	sw	a1,36(sp)
   107c8:	02e12823          	sw	a4,48(sp)
   107cc:	02f12a23          	sw	a5,52(sp)
   107d0:	03012c23          	sw	a6,56(sp)
   107d4:	03112e23          	sw	a7,60(sp)
   107d8:	00832583          	lw	a1,8(t1) # 101a8 <frame_dummy+0x1c>
   107dc:	02410693          	addi	a3,sp,36
   107e0:	00050613          	mv	a2,a0
   107e4:	00030513          	mv	a0,t1
   107e8:	00112e23          	sw	ra,28(sp)
   107ec:	00d12623          	sw	a3,12(sp)
   107f0:	3d4010ef          	jal	11bc4 <_vfprintf_r>
   107f4:	01c12083          	lw	ra,28(sp)
   107f8:	04010113          	addi	sp,sp,64
   107fc:	00008067          	ret

00010800 <_puts_r>:
   10800:	fc010113          	addi	sp,sp,-64
   10804:	02812c23          	sw	s0,56(sp)
   10808:	00050413          	mv	s0,a0
   1080c:	00058513          	mv	a0,a1
   10810:	02912a23          	sw	s1,52(sp)
   10814:	02112e23          	sw	ra,60(sp)
   10818:	00058493          	mv	s1,a1
   1081c:	5c4000ef          	jal	10de0 <strlen>
   10820:	00150713          	addi	a4,a0,1
   10824:	00010697          	auipc	a3,0x10
   10828:	0f468693          	addi	a3,a3,244 # 20918 <_exit+0xe0>
   1082c:	00e12e23          	sw	a4,28(sp)
   10830:	03442783          	lw	a5,52(s0)
   10834:	02010713          	addi	a4,sp,32
   10838:	02d12423          	sw	a3,40(sp)
   1083c:	00e12a23          	sw	a4,20(sp)
   10840:	00100693          	li	a3,1
   10844:	00200713          	li	a4,2
   10848:	02912023          	sw	s1,32(sp)
   1084c:	02a12223          	sw	a0,36(sp)
   10850:	02d12623          	sw	a3,44(sp)
   10854:	00e12c23          	sw	a4,24(sp)
   10858:	00842583          	lw	a1,8(s0)
   1085c:	06078063          	beqz	a5,108bc <_puts_r+0xbc>
   10860:	00c59783          	lh	a5,12(a1)
   10864:	01279713          	slli	a4,a5,0x12
   10868:	02074263          	bltz	a4,1088c <_puts_r+0x8c>
   1086c:	0645a703          	lw	a4,100(a1)
   10870:	ffffe6b7          	lui	a3,0xffffe
   10874:	fff68693          	addi	a3,a3,-1 # ffffdfff <__BSS_END__+0xfffdb1ff>
   10878:	00002637          	lui	a2,0x2
   1087c:	00c7e7b3          	or	a5,a5,a2
   10880:	00d77733          	and	a4,a4,a3
   10884:	00f59623          	sh	a5,12(a1)
   10888:	06e5a223          	sw	a4,100(a1)
   1088c:	01410613          	addi	a2,sp,20
   10890:	00040513          	mv	a0,s0
   10894:	6b8050ef          	jal	15f4c <__sfvwrite_r>
   10898:	03c12083          	lw	ra,60(sp)
   1089c:	03812403          	lw	s0,56(sp)
   108a0:	00153513          	seqz	a0,a0
   108a4:	40a00533          	neg	a0,a0
   108a8:	00b57513          	andi	a0,a0,11
   108ac:	03412483          	lw	s1,52(sp)
   108b0:	fff50513          	addi	a0,a0,-1
   108b4:	04010113          	addi	sp,sp,64
   108b8:	00008067          	ret
   108bc:	00040513          	mv	a0,s0
   108c0:	00b12623          	sw	a1,12(sp)
   108c4:	da1ff0ef          	jal	10664 <__sinit>
   108c8:	00c12583          	lw	a1,12(sp)
   108cc:	f95ff06f          	j	10860 <_puts_r+0x60>

000108d0 <puts>:
   108d0:	00050593          	mv	a1,a0
   108d4:	f5c1a503          	lw	a0,-164(gp) # 229dc <_impure_ptr>
   108d8:	f29ff06f          	j	10800 <_puts_r>

000108dc <__sread>:
   108dc:	ff010113          	addi	sp,sp,-16
   108e0:	00812423          	sw	s0,8(sp)
   108e4:	00058413          	mv	s0,a1
   108e8:	00e59583          	lh	a1,14(a1)
   108ec:	00112623          	sw	ra,12(sp)
   108f0:	2c8000ef          	jal	10bb8 <_read_r>
   108f4:	02054063          	bltz	a0,10914 <__sread+0x38>
   108f8:	05042783          	lw	a5,80(s0)
   108fc:	00c12083          	lw	ra,12(sp)
   10900:	00a787b3          	add	a5,a5,a0
   10904:	04f42823          	sw	a5,80(s0)
   10908:	00812403          	lw	s0,8(sp)
   1090c:	01010113          	addi	sp,sp,16
   10910:	00008067          	ret
   10914:	00c45783          	lhu	a5,12(s0)
   10918:	fffff737          	lui	a4,0xfffff
   1091c:	fff70713          	addi	a4,a4,-1 # ffffefff <__BSS_END__+0xfffdc1ff>
   10920:	00e7f7b3          	and	a5,a5,a4
   10924:	00c12083          	lw	ra,12(sp)
   10928:	00f41623          	sh	a5,12(s0)
   1092c:	00812403          	lw	s0,8(sp)
   10930:	01010113          	addi	sp,sp,16
   10934:	00008067          	ret

00010938 <__seofread>:
   10938:	00000513          	li	a0,0
   1093c:	00008067          	ret

00010940 <__swrite>:
   10940:	00c59783          	lh	a5,12(a1)
   10944:	fe010113          	addi	sp,sp,-32
   10948:	00812c23          	sw	s0,24(sp)
   1094c:	00912a23          	sw	s1,20(sp)
   10950:	01212823          	sw	s2,16(sp)
   10954:	01312623          	sw	s3,12(sp)
   10958:	00112e23          	sw	ra,28(sp)
   1095c:	1007f713          	andi	a4,a5,256
   10960:	00058413          	mv	s0,a1
   10964:	00050493          	mv	s1,a0
   10968:	00060913          	mv	s2,a2
   1096c:	00068993          	mv	s3,a3
   10970:	04071063          	bnez	a4,109b0 <__swrite+0x70>
   10974:	fffff737          	lui	a4,0xfffff
   10978:	fff70713          	addi	a4,a4,-1 # ffffefff <__BSS_END__+0xfffdc1ff>
   1097c:	00e7f7b3          	and	a5,a5,a4
   10980:	00e41583          	lh	a1,14(s0)
   10984:	00f41623          	sh	a5,12(s0)
   10988:	01812403          	lw	s0,24(sp)
   1098c:	01c12083          	lw	ra,28(sp)
   10990:	00098693          	mv	a3,s3
   10994:	00090613          	mv	a2,s2
   10998:	00c12983          	lw	s3,12(sp)
   1099c:	01012903          	lw	s2,16(sp)
   109a0:	00048513          	mv	a0,s1
   109a4:	01412483          	lw	s1,20(sp)
   109a8:	02010113          	addi	sp,sp,32
   109ac:	2680006f          	j	10c14 <_write_r>
   109b0:	00e59583          	lh	a1,14(a1)
   109b4:	00200693          	li	a3,2
   109b8:	00000613          	li	a2,0
   109bc:	1a0000ef          	jal	10b5c <_lseek_r>
   109c0:	00c41783          	lh	a5,12(s0)
   109c4:	fb1ff06f          	j	10974 <__swrite+0x34>

000109c8 <__sseek>:
   109c8:	ff010113          	addi	sp,sp,-16
   109cc:	00812423          	sw	s0,8(sp)
   109d0:	00058413          	mv	s0,a1
   109d4:	00e59583          	lh	a1,14(a1)
   109d8:	00112623          	sw	ra,12(sp)
   109dc:	180000ef          	jal	10b5c <_lseek_r>
   109e0:	fff00793          	li	a5,-1
   109e4:	02f50863          	beq	a0,a5,10a14 <__sseek+0x4c>
   109e8:	00c45783          	lhu	a5,12(s0)
   109ec:	00001737          	lui	a4,0x1
   109f0:	00c12083          	lw	ra,12(sp)
   109f4:	00e7e7b3          	or	a5,a5,a4
   109f8:	01079793          	slli	a5,a5,0x10
   109fc:	4107d793          	srai	a5,a5,0x10
   10a00:	04a42823          	sw	a0,80(s0)
   10a04:	00f41623          	sh	a5,12(s0)
   10a08:	00812403          	lw	s0,8(sp)
   10a0c:	01010113          	addi	sp,sp,16
   10a10:	00008067          	ret
   10a14:	00c45783          	lhu	a5,12(s0)
   10a18:	fffff737          	lui	a4,0xfffff
   10a1c:	fff70713          	addi	a4,a4,-1 # ffffefff <__BSS_END__+0xfffdc1ff>
   10a20:	00e7f7b3          	and	a5,a5,a4
   10a24:	01079793          	slli	a5,a5,0x10
   10a28:	4107d793          	srai	a5,a5,0x10
   10a2c:	00c12083          	lw	ra,12(sp)
   10a30:	00f41623          	sh	a5,12(s0)
   10a34:	00812403          	lw	s0,8(sp)
   10a38:	01010113          	addi	sp,sp,16
   10a3c:	00008067          	ret

00010a40 <__sclose>:
   10a40:	00e59583          	lh	a1,14(a1)
   10a44:	0040006f          	j	10a48 <_close_r>

00010a48 <_close_r>:
   10a48:	ff010113          	addi	sp,sp,-16
   10a4c:	00812423          	sw	s0,8(sp)
   10a50:	00050413          	mv	s0,a0
   10a54:	00058513          	mv	a0,a1
   10a58:	f601a623          	sw	zero,-148(gp) # 229ec <errno>
   10a5c:	00112623          	sw	ra,12(sp)
   10a60:	5290f0ef          	jal	20788 <_close>
   10a64:	fff00793          	li	a5,-1
   10a68:	00f50a63          	beq	a0,a5,10a7c <_close_r+0x34>
   10a6c:	00c12083          	lw	ra,12(sp)
   10a70:	00812403          	lw	s0,8(sp)
   10a74:	01010113          	addi	sp,sp,16
   10a78:	00008067          	ret
   10a7c:	f6c1a783          	lw	a5,-148(gp) # 229ec <errno>
   10a80:	fe0786e3          	beqz	a5,10a6c <_close_r+0x24>
   10a84:	00c12083          	lw	ra,12(sp)
   10a88:	00f42023          	sw	a5,0(s0)
   10a8c:	00812403          	lw	s0,8(sp)
   10a90:	01010113          	addi	sp,sp,16
   10a94:	00008067          	ret

00010a98 <_reclaim_reent>:
   10a98:	f5c1a783          	lw	a5,-164(gp) # 229dc <_impure_ptr>
   10a9c:	0aa78e63          	beq	a5,a0,10b58 <_reclaim_reent+0xc0>
   10aa0:	04452583          	lw	a1,68(a0)
   10aa4:	fe010113          	addi	sp,sp,-32
   10aa8:	00912a23          	sw	s1,20(sp)
   10aac:	00112e23          	sw	ra,28(sp)
   10ab0:	00050493          	mv	s1,a0
   10ab4:	04058c63          	beqz	a1,10b0c <_reclaim_reent+0x74>
   10ab8:	01212823          	sw	s2,16(sp)
   10abc:	01312623          	sw	s3,12(sp)
   10ac0:	00812c23          	sw	s0,24(sp)
   10ac4:	00000913          	li	s2,0
   10ac8:	08000993          	li	s3,128
   10acc:	012587b3          	add	a5,a1,s2
   10ad0:	0007a403          	lw	s0,0(a5)
   10ad4:	00040e63          	beqz	s0,10af0 <_reclaim_reent+0x58>
   10ad8:	00040593          	mv	a1,s0
   10adc:	00042403          	lw	s0,0(s0)
   10ae0:	00048513          	mv	a0,s1
   10ae4:	60c000ef          	jal	110f0 <_free_r>
   10ae8:	fe0418e3          	bnez	s0,10ad8 <_reclaim_reent+0x40>
   10aec:	0444a583          	lw	a1,68(s1)
   10af0:	00490913          	addi	s2,s2,4
   10af4:	fd391ce3          	bne	s2,s3,10acc <_reclaim_reent+0x34>
   10af8:	00048513          	mv	a0,s1
   10afc:	5f4000ef          	jal	110f0 <_free_r>
   10b00:	01812403          	lw	s0,24(sp)
   10b04:	01012903          	lw	s2,16(sp)
   10b08:	00c12983          	lw	s3,12(sp)
   10b0c:	0384a583          	lw	a1,56(s1)
   10b10:	00058663          	beqz	a1,10b1c <_reclaim_reent+0x84>
   10b14:	00048513          	mv	a0,s1
   10b18:	5d8000ef          	jal	110f0 <_free_r>
   10b1c:	04c4a583          	lw	a1,76(s1)
   10b20:	00058663          	beqz	a1,10b2c <_reclaim_reent+0x94>
   10b24:	00048513          	mv	a0,s1
   10b28:	5c8000ef          	jal	110f0 <_free_r>
   10b2c:	0344a783          	lw	a5,52(s1)
   10b30:	00078c63          	beqz	a5,10b48 <_reclaim_reent+0xb0>
   10b34:	01c12083          	lw	ra,28(sp)
   10b38:	00048513          	mv	a0,s1
   10b3c:	01412483          	lw	s1,20(sp)
   10b40:	02010113          	addi	sp,sp,32
   10b44:	00078067          	jr	a5
   10b48:	01c12083          	lw	ra,28(sp)
   10b4c:	01412483          	lw	s1,20(sp)
   10b50:	02010113          	addi	sp,sp,32
   10b54:	00008067          	ret
   10b58:	00008067          	ret

00010b5c <_lseek_r>:
   10b5c:	ff010113          	addi	sp,sp,-16
   10b60:	00058713          	mv	a4,a1
   10b64:	00812423          	sw	s0,8(sp)
   10b68:	00060593          	mv	a1,a2
   10b6c:	00050413          	mv	s0,a0
   10b70:	00068613          	mv	a2,a3
   10b74:	00070513          	mv	a0,a4
   10b78:	f601a623          	sw	zero,-148(gp) # 229ec <errno>
   10b7c:	00112623          	sw	ra,12(sp)
   10b80:	4590f0ef          	jal	207d8 <_lseek>
   10b84:	fff00793          	li	a5,-1
   10b88:	00f50a63          	beq	a0,a5,10b9c <_lseek_r+0x40>
   10b8c:	00c12083          	lw	ra,12(sp)
   10b90:	00812403          	lw	s0,8(sp)
   10b94:	01010113          	addi	sp,sp,16
   10b98:	00008067          	ret
   10b9c:	f6c1a783          	lw	a5,-148(gp) # 229ec <errno>
   10ba0:	fe0786e3          	beqz	a5,10b8c <_lseek_r+0x30>
   10ba4:	00c12083          	lw	ra,12(sp)
   10ba8:	00f42023          	sw	a5,0(s0)
   10bac:	00812403          	lw	s0,8(sp)
   10bb0:	01010113          	addi	sp,sp,16
   10bb4:	00008067          	ret

00010bb8 <_read_r>:
   10bb8:	ff010113          	addi	sp,sp,-16
   10bbc:	00058713          	mv	a4,a1
   10bc0:	00812423          	sw	s0,8(sp)
   10bc4:	00060593          	mv	a1,a2
   10bc8:	00050413          	mv	s0,a0
   10bcc:	00068613          	mv	a2,a3
   10bd0:	00070513          	mv	a0,a4
   10bd4:	f601a623          	sw	zero,-148(gp) # 229ec <errno>
   10bd8:	00112623          	sw	ra,12(sp)
   10bdc:	40d0f0ef          	jal	207e8 <_read>
   10be0:	fff00793          	li	a5,-1
   10be4:	00f50a63          	beq	a0,a5,10bf8 <_read_r+0x40>
   10be8:	00c12083          	lw	ra,12(sp)
   10bec:	00812403          	lw	s0,8(sp)
   10bf0:	01010113          	addi	sp,sp,16
   10bf4:	00008067          	ret
   10bf8:	f6c1a783          	lw	a5,-148(gp) # 229ec <errno>
   10bfc:	fe0786e3          	beqz	a5,10be8 <_read_r+0x30>
   10c00:	00c12083          	lw	ra,12(sp)
   10c04:	00f42023          	sw	a5,0(s0)
   10c08:	00812403          	lw	s0,8(sp)
   10c0c:	01010113          	addi	sp,sp,16
   10c10:	00008067          	ret

00010c14 <_write_r>:
   10c14:	ff010113          	addi	sp,sp,-16
   10c18:	00058713          	mv	a4,a1
   10c1c:	00812423          	sw	s0,8(sp)
   10c20:	00060593          	mv	a1,a2
   10c24:	00050413          	mv	s0,a0
   10c28:	00068613          	mv	a2,a3
   10c2c:	00070513          	mv	a0,a4
   10c30:	f601a623          	sw	zero,-148(gp) # 229ec <errno>
   10c34:	00112623          	sw	ra,12(sp)
   10c38:	3f10f0ef          	jal	20828 <_write>
   10c3c:	fff00793          	li	a5,-1
   10c40:	00f50a63          	beq	a0,a5,10c54 <_write_r+0x40>
   10c44:	00c12083          	lw	ra,12(sp)
   10c48:	00812403          	lw	s0,8(sp)
   10c4c:	01010113          	addi	sp,sp,16
   10c50:	00008067          	ret
   10c54:	f6c1a783          	lw	a5,-148(gp) # 229ec <errno>
   10c58:	fe0786e3          	beqz	a5,10c44 <_write_r+0x30>
   10c5c:	00c12083          	lw	ra,12(sp)
   10c60:	00f42023          	sw	a5,0(s0)
   10c64:	00812403          	lw	s0,8(sp)
   10c68:	01010113          	addi	sp,sp,16
   10c6c:	00008067          	ret

00010c70 <__libc_init_array>:
   10c70:	ff010113          	addi	sp,sp,-16
   10c74:	00812423          	sw	s0,8(sp)
   10c78:	01212023          	sw	s2,0(sp)
   10c7c:	00011797          	auipc	a5,0x11
   10c80:	5e078793          	addi	a5,a5,1504 # 2225c <__init_array_start>
   10c84:	00011417          	auipc	s0,0x11
   10c88:	5d840413          	addi	s0,s0,1496 # 2225c <__init_array_start>
   10c8c:	00112623          	sw	ra,12(sp)
   10c90:	00912223          	sw	s1,4(sp)
   10c94:	40878933          	sub	s2,a5,s0
   10c98:	02878063          	beq	a5,s0,10cb8 <__libc_init_array+0x48>
   10c9c:	40295913          	srai	s2,s2,0x2
   10ca0:	00000493          	li	s1,0
   10ca4:	00042783          	lw	a5,0(s0)
   10ca8:	00148493          	addi	s1,s1,1
   10cac:	00440413          	addi	s0,s0,4
   10cb0:	000780e7          	jalr	a5
   10cb4:	ff24e8e3          	bltu	s1,s2,10ca4 <__libc_init_array+0x34>
   10cb8:	00011797          	auipc	a5,0x11
   10cbc:	5ac78793          	addi	a5,a5,1452 # 22264 <__do_global_dtors_aux_fini_array_entry>
   10cc0:	00011417          	auipc	s0,0x11
   10cc4:	59c40413          	addi	s0,s0,1436 # 2225c <__init_array_start>
   10cc8:	40878933          	sub	s2,a5,s0
   10ccc:	40295913          	srai	s2,s2,0x2
   10cd0:	00878e63          	beq	a5,s0,10cec <__libc_init_array+0x7c>
   10cd4:	00000493          	li	s1,0
   10cd8:	00042783          	lw	a5,0(s0)
   10cdc:	00148493          	addi	s1,s1,1
   10ce0:	00440413          	addi	s0,s0,4
   10ce4:	000780e7          	jalr	a5
   10ce8:	ff24e8e3          	bltu	s1,s2,10cd8 <__libc_init_array+0x68>
   10cec:	00c12083          	lw	ra,12(sp)
   10cf0:	00812403          	lw	s0,8(sp)
   10cf4:	00412483          	lw	s1,4(sp)
   10cf8:	00012903          	lw	s2,0(sp)
   10cfc:	01010113          	addi	sp,sp,16
   10d00:	00008067          	ret

00010d04 <memset>:
   10d04:	00f00313          	li	t1,15
   10d08:	00050713          	mv	a4,a0
   10d0c:	02c37e63          	bgeu	t1,a2,10d48 <memset+0x44>
   10d10:	00f77793          	andi	a5,a4,15
   10d14:	0a079063          	bnez	a5,10db4 <memset+0xb0>
   10d18:	08059263          	bnez	a1,10d9c <memset+0x98>
   10d1c:	ff067693          	andi	a3,a2,-16
   10d20:	00f67613          	andi	a2,a2,15
   10d24:	00e686b3          	add	a3,a3,a4
   10d28:	00b72023          	sw	a1,0(a4)
   10d2c:	00b72223          	sw	a1,4(a4)
   10d30:	00b72423          	sw	a1,8(a4)
   10d34:	00b72623          	sw	a1,12(a4)
   10d38:	01070713          	addi	a4,a4,16
   10d3c:	fed766e3          	bltu	a4,a3,10d28 <memset+0x24>
   10d40:	00061463          	bnez	a2,10d48 <memset+0x44>
   10d44:	00008067          	ret
   10d48:	40c306b3          	sub	a3,t1,a2
   10d4c:	00269693          	slli	a3,a3,0x2
   10d50:	00000297          	auipc	t0,0x0
   10d54:	005686b3          	add	a3,a3,t0
   10d58:	00c68067          	jr	12(a3)
   10d5c:	00b70723          	sb	a1,14(a4)
   10d60:	00b706a3          	sb	a1,13(a4)
   10d64:	00b70623          	sb	a1,12(a4)
   10d68:	00b705a3          	sb	a1,11(a4)
   10d6c:	00b70523          	sb	a1,10(a4)
   10d70:	00b704a3          	sb	a1,9(a4)
   10d74:	00b70423          	sb	a1,8(a4)
   10d78:	00b703a3          	sb	a1,7(a4)
   10d7c:	00b70323          	sb	a1,6(a4)
   10d80:	00b702a3          	sb	a1,5(a4)
   10d84:	00b70223          	sb	a1,4(a4)
   10d88:	00b701a3          	sb	a1,3(a4)
   10d8c:	00b70123          	sb	a1,2(a4)
   10d90:	00b700a3          	sb	a1,1(a4)
   10d94:	00b70023          	sb	a1,0(a4)
   10d98:	00008067          	ret
   10d9c:	0ff5f593          	zext.b	a1,a1
   10da0:	00859693          	slli	a3,a1,0x8
   10da4:	00d5e5b3          	or	a1,a1,a3
   10da8:	01059693          	slli	a3,a1,0x10
   10dac:	00d5e5b3          	or	a1,a1,a3
   10db0:	f6dff06f          	j	10d1c <memset+0x18>
   10db4:	00279693          	slli	a3,a5,0x2
   10db8:	00000297          	auipc	t0,0x0
   10dbc:	005686b3          	add	a3,a3,t0
   10dc0:	00008293          	mv	t0,ra
   10dc4:	fa0680e7          	jalr	-96(a3)
   10dc8:	00028093          	mv	ra,t0
   10dcc:	ff078793          	addi	a5,a5,-16
   10dd0:	40f70733          	sub	a4,a4,a5
   10dd4:	00f60633          	add	a2,a2,a5
   10dd8:	f6c378e3          	bgeu	t1,a2,10d48 <memset+0x44>
   10ddc:	f3dff06f          	j	10d18 <memset+0x14>

00010de0 <strlen>:
   10de0:	00357793          	andi	a5,a0,3
   10de4:	00050713          	mv	a4,a0
   10de8:	04079c63          	bnez	a5,10e40 <strlen+0x60>
   10dec:	7f7f86b7          	lui	a3,0x7f7f8
   10df0:	f7f68693          	addi	a3,a3,-129 # 7f7f7f7f <__BSS_END__+0x7f7d517f>
   10df4:	fff00593          	li	a1,-1
   10df8:	00072603          	lw	a2,0(a4)
   10dfc:	00470713          	addi	a4,a4,4
   10e00:	00d677b3          	and	a5,a2,a3
   10e04:	00d787b3          	add	a5,a5,a3
   10e08:	00c7e7b3          	or	a5,a5,a2
   10e0c:	00d7e7b3          	or	a5,a5,a3
   10e10:	feb784e3          	beq	a5,a1,10df8 <strlen+0x18>
   10e14:	ffc74683          	lbu	a3,-4(a4)
   10e18:	40a707b3          	sub	a5,a4,a0
   10e1c:	04068463          	beqz	a3,10e64 <strlen+0x84>
   10e20:	ffd74683          	lbu	a3,-3(a4)
   10e24:	02068c63          	beqz	a3,10e5c <strlen+0x7c>
   10e28:	ffe74503          	lbu	a0,-2(a4)
   10e2c:	00a03533          	snez	a0,a0
   10e30:	00f50533          	add	a0,a0,a5
   10e34:	ffe50513          	addi	a0,a0,-2
   10e38:	00008067          	ret
   10e3c:	fa0688e3          	beqz	a3,10dec <strlen+0xc>
   10e40:	00074783          	lbu	a5,0(a4)
   10e44:	00170713          	addi	a4,a4,1
   10e48:	00377693          	andi	a3,a4,3
   10e4c:	fe0798e3          	bnez	a5,10e3c <strlen+0x5c>
   10e50:	40a70733          	sub	a4,a4,a0
   10e54:	fff70513          	addi	a0,a4,-1
   10e58:	00008067          	ret
   10e5c:	ffd78513          	addi	a0,a5,-3
   10e60:	00008067          	ret
   10e64:	ffc78513          	addi	a0,a5,-4
   10e68:	00008067          	ret

00010e6c <__call_exitprocs>:
   10e6c:	fd010113          	addi	sp,sp,-48
   10e70:	01412c23          	sw	s4,24(sp)
   10e74:	f7018a13          	addi	s4,gp,-144 # 229f0 <__atexit>
   10e78:	03212023          	sw	s2,32(sp)
   10e7c:	000a2903          	lw	s2,0(s4)
   10e80:	02112623          	sw	ra,44(sp)
   10e84:	0a090863          	beqz	s2,10f34 <__call_exitprocs+0xc8>
   10e88:	01312e23          	sw	s3,28(sp)
   10e8c:	01512a23          	sw	s5,20(sp)
   10e90:	01612823          	sw	s6,16(sp)
   10e94:	01712623          	sw	s7,12(sp)
   10e98:	02812423          	sw	s0,40(sp)
   10e9c:	02912223          	sw	s1,36(sp)
   10ea0:	01812423          	sw	s8,8(sp)
   10ea4:	00050b13          	mv	s6,a0
   10ea8:	00058b93          	mv	s7,a1
   10eac:	fff00993          	li	s3,-1
   10eb0:	00100a93          	li	s5,1
   10eb4:	00492483          	lw	s1,4(s2)
   10eb8:	fff48413          	addi	s0,s1,-1
   10ebc:	04044e63          	bltz	s0,10f18 <__call_exitprocs+0xac>
   10ec0:	00249493          	slli	s1,s1,0x2
   10ec4:	009904b3          	add	s1,s2,s1
   10ec8:	080b9063          	bnez	s7,10f48 <__call_exitprocs+0xdc>
   10ecc:	00492783          	lw	a5,4(s2)
   10ed0:	0044a683          	lw	a3,4(s1)
   10ed4:	fff78793          	addi	a5,a5,-1
   10ed8:	0a878c63          	beq	a5,s0,10f90 <__call_exitprocs+0x124>
   10edc:	0004a223          	sw	zero,4(s1)
   10ee0:	02068663          	beqz	a3,10f0c <__call_exitprocs+0xa0>
   10ee4:	18892783          	lw	a5,392(s2)
   10ee8:	008a9733          	sll	a4,s5,s0
   10eec:	00492c03          	lw	s8,4(s2)
   10ef0:	00f777b3          	and	a5,a4,a5
   10ef4:	06079663          	bnez	a5,10f60 <__call_exitprocs+0xf4>
   10ef8:	000680e7          	jalr	a3
   10efc:	00492703          	lw	a4,4(s2)
   10f00:	000a2783          	lw	a5,0(s4)
   10f04:	09871063          	bne	a4,s8,10f84 <__call_exitprocs+0x118>
   10f08:	07279e63          	bne	a5,s2,10f84 <__call_exitprocs+0x118>
   10f0c:	fff40413          	addi	s0,s0,-1
   10f10:	ffc48493          	addi	s1,s1,-4
   10f14:	fb341ae3          	bne	s0,s3,10ec8 <__call_exitprocs+0x5c>
   10f18:	02812403          	lw	s0,40(sp)
   10f1c:	02412483          	lw	s1,36(sp)
   10f20:	01c12983          	lw	s3,28(sp)
   10f24:	01412a83          	lw	s5,20(sp)
   10f28:	01012b03          	lw	s6,16(sp)
   10f2c:	00c12b83          	lw	s7,12(sp)
   10f30:	00812c03          	lw	s8,8(sp)
   10f34:	02c12083          	lw	ra,44(sp)
   10f38:	02012903          	lw	s2,32(sp)
   10f3c:	01812a03          	lw	s4,24(sp)
   10f40:	03010113          	addi	sp,sp,48
   10f44:	00008067          	ret
   10f48:	1044a783          	lw	a5,260(s1)
   10f4c:	f97780e3          	beq	a5,s7,10ecc <__call_exitprocs+0x60>
   10f50:	fff40413          	addi	s0,s0,-1
   10f54:	ffc48493          	addi	s1,s1,-4
   10f58:	ff3418e3          	bne	s0,s3,10f48 <__call_exitprocs+0xdc>
   10f5c:	fbdff06f          	j	10f18 <__call_exitprocs+0xac>
   10f60:	18c92783          	lw	a5,396(s2)
   10f64:	0844a583          	lw	a1,132(s1)
   10f68:	00f77733          	and	a4,a4,a5
   10f6c:	02071663          	bnez	a4,10f98 <__call_exitprocs+0x12c>
   10f70:	000b0513          	mv	a0,s6
   10f74:	000680e7          	jalr	a3
   10f78:	00492703          	lw	a4,4(s2)
   10f7c:	000a2783          	lw	a5,0(s4)
   10f80:	f98704e3          	beq	a4,s8,10f08 <__call_exitprocs+0x9c>
   10f84:	f8078ae3          	beqz	a5,10f18 <__call_exitprocs+0xac>
   10f88:	00078913          	mv	s2,a5
   10f8c:	f29ff06f          	j	10eb4 <__call_exitprocs+0x48>
   10f90:	00892223          	sw	s0,4(s2)
   10f94:	f4dff06f          	j	10ee0 <__call_exitprocs+0x74>
   10f98:	00058513          	mv	a0,a1
   10f9c:	000680e7          	jalr	a3
   10fa0:	f5dff06f          	j	10efc <__call_exitprocs+0x90>

00010fa4 <atexit>:
   10fa4:	00050593          	mv	a1,a0
   10fa8:	00000693          	li	a3,0
   10fac:	00000613          	li	a2,0
   10fb0:	00000513          	li	a0,0
   10fb4:	07c0606f          	j	17030 <__register_exitproc>

00010fb8 <_malloc_trim_r>:
   10fb8:	fe010113          	addi	sp,sp,-32
   10fbc:	00812c23          	sw	s0,24(sp)
   10fc0:	00912a23          	sw	s1,20(sp)
   10fc4:	01212823          	sw	s2,16(sp)
   10fc8:	01312623          	sw	s3,12(sp)
   10fcc:	01412423          	sw	s4,8(sp)
   10fd0:	00058993          	mv	s3,a1
   10fd4:	00112e23          	sw	ra,28(sp)
   10fd8:	00050913          	mv	s2,a0
   10fdc:	00011a17          	auipc	s4,0x11
   10fe0:	424a0a13          	addi	s4,s4,1060 # 22400 <__malloc_av_>
   10fe4:	3d9000ef          	jal	11bbc <__malloc_lock>
   10fe8:	008a2703          	lw	a4,8(s4)
   10fec:	000017b7          	lui	a5,0x1
   10ff0:	fef78793          	addi	a5,a5,-17 # fef <exit-0xf0c5>
   10ff4:	00472483          	lw	s1,4(a4)
   10ff8:	00001737          	lui	a4,0x1
   10ffc:	ffc4f493          	andi	s1,s1,-4
   11000:	00f48433          	add	s0,s1,a5
   11004:	41340433          	sub	s0,s0,s3
   11008:	00c45413          	srli	s0,s0,0xc
   1100c:	fff40413          	addi	s0,s0,-1
   11010:	00c41413          	slli	s0,s0,0xc
   11014:	00e44e63          	blt	s0,a4,11030 <_malloc_trim_r+0x78>
   11018:	00000593          	li	a1,0
   1101c:	00090513          	mv	a0,s2
   11020:	1a1050ef          	jal	169c0 <_sbrk_r>
   11024:	008a2783          	lw	a5,8(s4)
   11028:	009787b3          	add	a5,a5,s1
   1102c:	02f50863          	beq	a0,a5,1105c <_malloc_trim_r+0xa4>
   11030:	00090513          	mv	a0,s2
   11034:	38d000ef          	jal	11bc0 <__malloc_unlock>
   11038:	01c12083          	lw	ra,28(sp)
   1103c:	01812403          	lw	s0,24(sp)
   11040:	01412483          	lw	s1,20(sp)
   11044:	01012903          	lw	s2,16(sp)
   11048:	00c12983          	lw	s3,12(sp)
   1104c:	00812a03          	lw	s4,8(sp)
   11050:	00000513          	li	a0,0
   11054:	02010113          	addi	sp,sp,32
   11058:	00008067          	ret
   1105c:	408005b3          	neg	a1,s0
   11060:	00090513          	mv	a0,s2
   11064:	15d050ef          	jal	169c0 <_sbrk_r>
   11068:	fff00793          	li	a5,-1
   1106c:	04f50863          	beq	a0,a5,110bc <_malloc_trim_r+0x104>
   11070:	1c818713          	addi	a4,gp,456 # 22c48 <__malloc_current_mallinfo>
   11074:	00072783          	lw	a5,0(a4) # 1000 <exit-0xf0b4>
   11078:	008a2683          	lw	a3,8(s4)
   1107c:	408484b3          	sub	s1,s1,s0
   11080:	0014e493          	ori	s1,s1,1
   11084:	408787b3          	sub	a5,a5,s0
   11088:	00090513          	mv	a0,s2
   1108c:	0096a223          	sw	s1,4(a3)
   11090:	00f72023          	sw	a5,0(a4)
   11094:	32d000ef          	jal	11bc0 <__malloc_unlock>
   11098:	01c12083          	lw	ra,28(sp)
   1109c:	01812403          	lw	s0,24(sp)
   110a0:	01412483          	lw	s1,20(sp)
   110a4:	01012903          	lw	s2,16(sp)
   110a8:	00c12983          	lw	s3,12(sp)
   110ac:	00812a03          	lw	s4,8(sp)
   110b0:	00100513          	li	a0,1
   110b4:	02010113          	addi	sp,sp,32
   110b8:	00008067          	ret
   110bc:	00000593          	li	a1,0
   110c0:	00090513          	mv	a0,s2
   110c4:	0fd050ef          	jal	169c0 <_sbrk_r>
   110c8:	008a2703          	lw	a4,8(s4)
   110cc:	00f00693          	li	a3,15
   110d0:	40e507b3          	sub	a5,a0,a4
   110d4:	f4f6dee3          	bge	a3,a5,11030 <_malloc_trim_r+0x78>
   110d8:	f601a683          	lw	a3,-160(gp) # 229e0 <__malloc_sbrk_base>
   110dc:	40d50533          	sub	a0,a0,a3
   110e0:	0017e793          	ori	a5,a5,1
   110e4:	1ca1a423          	sw	a0,456(gp) # 22c48 <__malloc_current_mallinfo>
   110e8:	00f72223          	sw	a5,4(a4)
   110ec:	f45ff06f          	j	11030 <_malloc_trim_r+0x78>

000110f0 <_free_r>:
   110f0:	18058263          	beqz	a1,11274 <_free_r+0x184>
   110f4:	ff010113          	addi	sp,sp,-16
   110f8:	00812423          	sw	s0,8(sp)
   110fc:	00912223          	sw	s1,4(sp)
   11100:	00058413          	mv	s0,a1
   11104:	00050493          	mv	s1,a0
   11108:	00112623          	sw	ra,12(sp)
   1110c:	2b1000ef          	jal	11bbc <__malloc_lock>
   11110:	ffc42583          	lw	a1,-4(s0)
   11114:	ff840713          	addi	a4,s0,-8
   11118:	00011517          	auipc	a0,0x11
   1111c:	2e850513          	addi	a0,a0,744 # 22400 <__malloc_av_>
   11120:	ffe5f793          	andi	a5,a1,-2
   11124:	00f70633          	add	a2,a4,a5
   11128:	00462683          	lw	a3,4(a2) # 2004 <exit-0xe0b0>
   1112c:	00852803          	lw	a6,8(a0)
   11130:	ffc6f693          	andi	a3,a3,-4
   11134:	1ac80263          	beq	a6,a2,112d8 <_free_r+0x1e8>
   11138:	00d62223          	sw	a3,4(a2)
   1113c:	0015f593          	andi	a1,a1,1
   11140:	00d60833          	add	a6,a2,a3
   11144:	0a059063          	bnez	a1,111e4 <_free_r+0xf4>
   11148:	ff842303          	lw	t1,-8(s0)
   1114c:	00482583          	lw	a1,4(a6)
   11150:	00011897          	auipc	a7,0x11
   11154:	2b888893          	addi	a7,a7,696 # 22408 <__malloc_av_+0x8>
   11158:	40670733          	sub	a4,a4,t1
   1115c:	00872803          	lw	a6,8(a4)
   11160:	006787b3          	add	a5,a5,t1
   11164:	0015f593          	andi	a1,a1,1
   11168:	15180263          	beq	a6,a7,112ac <_free_r+0x1bc>
   1116c:	00c72303          	lw	t1,12(a4)
   11170:	00682623          	sw	t1,12(a6)
   11174:	01032423          	sw	a6,8(t1)
   11178:	1a058663          	beqz	a1,11324 <_free_r+0x234>
   1117c:	0017e693          	ori	a3,a5,1
   11180:	00d72223          	sw	a3,4(a4)
   11184:	00f62023          	sw	a5,0(a2)
   11188:	1ff00693          	li	a3,511
   1118c:	06f6ec63          	bltu	a3,a5,11204 <_free_r+0x114>
   11190:	ff87f693          	andi	a3,a5,-8
   11194:	00868693          	addi	a3,a3,8
   11198:	00452583          	lw	a1,4(a0)
   1119c:	00d506b3          	add	a3,a0,a3
   111a0:	0006a603          	lw	a2,0(a3)
   111a4:	0057d813          	srli	a6,a5,0x5
   111a8:	00100793          	li	a5,1
   111ac:	010797b3          	sll	a5,a5,a6
   111b0:	00b7e7b3          	or	a5,a5,a1
   111b4:	ff868593          	addi	a1,a3,-8
   111b8:	00b72623          	sw	a1,12(a4)
   111bc:	00c72423          	sw	a2,8(a4)
   111c0:	00f52223          	sw	a5,4(a0)
   111c4:	00e6a023          	sw	a4,0(a3)
   111c8:	00e62623          	sw	a4,12(a2)
   111cc:	00812403          	lw	s0,8(sp)
   111d0:	00c12083          	lw	ra,12(sp)
   111d4:	00048513          	mv	a0,s1
   111d8:	00412483          	lw	s1,4(sp)
   111dc:	01010113          	addi	sp,sp,16
   111e0:	1e10006f          	j	11bc0 <__malloc_unlock>
   111e4:	00482583          	lw	a1,4(a6)
   111e8:	0015f593          	andi	a1,a1,1
   111ec:	08058663          	beqz	a1,11278 <_free_r+0x188>
   111f0:	0017e693          	ori	a3,a5,1
   111f4:	fed42e23          	sw	a3,-4(s0)
   111f8:	00f62023          	sw	a5,0(a2)
   111fc:	1ff00693          	li	a3,511
   11200:	f8f6f8e3          	bgeu	a3,a5,11190 <_free_r+0xa0>
   11204:	0097d693          	srli	a3,a5,0x9
   11208:	00400613          	li	a2,4
   1120c:	12d66063          	bltu	a2,a3,1132c <_free_r+0x23c>
   11210:	0067d693          	srli	a3,a5,0x6
   11214:	03968593          	addi	a1,a3,57
   11218:	03868613          	addi	a2,a3,56
   1121c:	00359593          	slli	a1,a1,0x3
   11220:	00b505b3          	add	a1,a0,a1
   11224:	0005a683          	lw	a3,0(a1)
   11228:	ff858593          	addi	a1,a1,-8
   1122c:	00d59863          	bne	a1,a3,1123c <_free_r+0x14c>
   11230:	1540006f          	j	11384 <_free_r+0x294>
   11234:	0086a683          	lw	a3,8(a3)
   11238:	00d58863          	beq	a1,a3,11248 <_free_r+0x158>
   1123c:	0046a603          	lw	a2,4(a3)
   11240:	ffc67613          	andi	a2,a2,-4
   11244:	fec7e8e3          	bltu	a5,a2,11234 <_free_r+0x144>
   11248:	00c6a583          	lw	a1,12(a3)
   1124c:	00b72623          	sw	a1,12(a4)
   11250:	00d72423          	sw	a3,8(a4)
   11254:	00812403          	lw	s0,8(sp)
   11258:	00c12083          	lw	ra,12(sp)
   1125c:	00e5a423          	sw	a4,8(a1)
   11260:	00048513          	mv	a0,s1
   11264:	00412483          	lw	s1,4(sp)
   11268:	00e6a623          	sw	a4,12(a3)
   1126c:	01010113          	addi	sp,sp,16
   11270:	1510006f          	j	11bc0 <__malloc_unlock>
   11274:	00008067          	ret
   11278:	00d787b3          	add	a5,a5,a3
   1127c:	00011897          	auipc	a7,0x11
   11280:	18c88893          	addi	a7,a7,396 # 22408 <__malloc_av_+0x8>
   11284:	00862683          	lw	a3,8(a2)
   11288:	0d168c63          	beq	a3,a7,11360 <_free_r+0x270>
   1128c:	00c62803          	lw	a6,12(a2)
   11290:	0017e593          	ori	a1,a5,1
   11294:	00f70633          	add	a2,a4,a5
   11298:	0106a623          	sw	a6,12(a3)
   1129c:	00d82423          	sw	a3,8(a6)
   112a0:	00b72223          	sw	a1,4(a4)
   112a4:	00f62023          	sw	a5,0(a2)
   112a8:	ee1ff06f          	j	11188 <_free_r+0x98>
   112ac:	12059c63          	bnez	a1,113e4 <_free_r+0x2f4>
   112b0:	00862583          	lw	a1,8(a2)
   112b4:	00c62603          	lw	a2,12(a2)
   112b8:	00f686b3          	add	a3,a3,a5
   112bc:	0016e793          	ori	a5,a3,1
   112c0:	00c5a623          	sw	a2,12(a1)
   112c4:	00b62423          	sw	a1,8(a2)
   112c8:	00f72223          	sw	a5,4(a4)
   112cc:	00d70733          	add	a4,a4,a3
   112d0:	00d72023          	sw	a3,0(a4)
   112d4:	ef9ff06f          	j	111cc <_free_r+0xdc>
   112d8:	0015f593          	andi	a1,a1,1
   112dc:	00d786b3          	add	a3,a5,a3
   112e0:	02059063          	bnez	a1,11300 <_free_r+0x210>
   112e4:	ff842583          	lw	a1,-8(s0)
   112e8:	40b70733          	sub	a4,a4,a1
   112ec:	00c72783          	lw	a5,12(a4)
   112f0:	00872603          	lw	a2,8(a4)
   112f4:	00b686b3          	add	a3,a3,a1
   112f8:	00f62623          	sw	a5,12(a2)
   112fc:	00c7a423          	sw	a2,8(a5)
   11300:	0016e793          	ori	a5,a3,1
   11304:	00f72223          	sw	a5,4(a4)
   11308:	00e52423          	sw	a4,8(a0)
   1130c:	f641a783          	lw	a5,-156(gp) # 229e4 <__malloc_trim_threshold>
   11310:	eaf6eee3          	bltu	a3,a5,111cc <_free_r+0xdc>
   11314:	f7c1a583          	lw	a1,-132(gp) # 229fc <__malloc_top_pad>
   11318:	00048513          	mv	a0,s1
   1131c:	c9dff0ef          	jal	10fb8 <_malloc_trim_r>
   11320:	eadff06f          	j	111cc <_free_r+0xdc>
   11324:	00d787b3          	add	a5,a5,a3
   11328:	f5dff06f          	j	11284 <_free_r+0x194>
   1132c:	01400613          	li	a2,20
   11330:	02d67063          	bgeu	a2,a3,11350 <_free_r+0x260>
   11334:	05400613          	li	a2,84
   11338:	06d66463          	bltu	a2,a3,113a0 <_free_r+0x2b0>
   1133c:	00c7d693          	srli	a3,a5,0xc
   11340:	06f68593          	addi	a1,a3,111
   11344:	06e68613          	addi	a2,a3,110
   11348:	00359593          	slli	a1,a1,0x3
   1134c:	ed5ff06f          	j	11220 <_free_r+0x130>
   11350:	05c68593          	addi	a1,a3,92
   11354:	05b68613          	addi	a2,a3,91
   11358:	00359593          	slli	a1,a1,0x3
   1135c:	ec5ff06f          	j	11220 <_free_r+0x130>
   11360:	00e52a23          	sw	a4,20(a0)
   11364:	00e52823          	sw	a4,16(a0)
   11368:	0017e693          	ori	a3,a5,1
   1136c:	01172623          	sw	a7,12(a4)
   11370:	01172423          	sw	a7,8(a4)
   11374:	00d72223          	sw	a3,4(a4)
   11378:	00f70733          	add	a4,a4,a5
   1137c:	00f72023          	sw	a5,0(a4)
   11380:	e4dff06f          	j	111cc <_free_r+0xdc>
   11384:	00452803          	lw	a6,4(a0)
   11388:	40265613          	srai	a2,a2,0x2
   1138c:	00100793          	li	a5,1
   11390:	00c797b3          	sll	a5,a5,a2
   11394:	0107e7b3          	or	a5,a5,a6
   11398:	00f52223          	sw	a5,4(a0)
   1139c:	eb1ff06f          	j	1124c <_free_r+0x15c>
   113a0:	15400613          	li	a2,340
   113a4:	00d66c63          	bltu	a2,a3,113bc <_free_r+0x2cc>
   113a8:	00f7d693          	srli	a3,a5,0xf
   113ac:	07868593          	addi	a1,a3,120
   113b0:	07768613          	addi	a2,a3,119
   113b4:	00359593          	slli	a1,a1,0x3
   113b8:	e69ff06f          	j	11220 <_free_r+0x130>
   113bc:	55400613          	li	a2,1364
   113c0:	00d66c63          	bltu	a2,a3,113d8 <_free_r+0x2e8>
   113c4:	0127d693          	srli	a3,a5,0x12
   113c8:	07d68593          	addi	a1,a3,125
   113cc:	07c68613          	addi	a2,a3,124
   113d0:	00359593          	slli	a1,a1,0x3
   113d4:	e4dff06f          	j	11220 <_free_r+0x130>
   113d8:	3f800593          	li	a1,1016
   113dc:	07e00613          	li	a2,126
   113e0:	e41ff06f          	j	11220 <_free_r+0x130>
   113e4:	0017e693          	ori	a3,a5,1
   113e8:	00d72223          	sw	a3,4(a4)
   113ec:	00f62023          	sw	a5,0(a2)
   113f0:	dddff06f          	j	111cc <_free_r+0xdc>

000113f4 <_malloc_r>:
   113f4:	fd010113          	addi	sp,sp,-48
   113f8:	03212023          	sw	s2,32(sp)
   113fc:	02112623          	sw	ra,44(sp)
   11400:	02812423          	sw	s0,40(sp)
   11404:	02912223          	sw	s1,36(sp)
   11408:	01312e23          	sw	s3,28(sp)
   1140c:	00b58793          	addi	a5,a1,11
   11410:	01600713          	li	a4,22
   11414:	00050913          	mv	s2,a0
   11418:	08f76263          	bltu	a4,a5,1149c <_malloc_r+0xa8>
   1141c:	01000793          	li	a5,16
   11420:	20b7e663          	bltu	a5,a1,1162c <_malloc_r+0x238>
   11424:	798000ef          	jal	11bbc <__malloc_lock>
   11428:	01800793          	li	a5,24
   1142c:	00200593          	li	a1,2
   11430:	01000493          	li	s1,16
   11434:	00011997          	auipc	s3,0x11
   11438:	fcc98993          	addi	s3,s3,-52 # 22400 <__malloc_av_>
   1143c:	00f987b3          	add	a5,s3,a5
   11440:	0047a403          	lw	s0,4(a5)
   11444:	ff878713          	addi	a4,a5,-8
   11448:	34e40a63          	beq	s0,a4,1179c <_malloc_r+0x3a8>
   1144c:	00442783          	lw	a5,4(s0)
   11450:	00c42683          	lw	a3,12(s0)
   11454:	00842603          	lw	a2,8(s0)
   11458:	ffc7f793          	andi	a5,a5,-4
   1145c:	00f407b3          	add	a5,s0,a5
   11460:	0047a703          	lw	a4,4(a5)
   11464:	00d62623          	sw	a3,12(a2)
   11468:	00c6a423          	sw	a2,8(a3)
   1146c:	00176713          	ori	a4,a4,1
   11470:	00090513          	mv	a0,s2
   11474:	00e7a223          	sw	a4,4(a5)
   11478:	748000ef          	jal	11bc0 <__malloc_unlock>
   1147c:	00840513          	addi	a0,s0,8
   11480:	02c12083          	lw	ra,44(sp)
   11484:	02812403          	lw	s0,40(sp)
   11488:	02412483          	lw	s1,36(sp)
   1148c:	02012903          	lw	s2,32(sp)
   11490:	01c12983          	lw	s3,28(sp)
   11494:	03010113          	addi	sp,sp,48
   11498:	00008067          	ret
   1149c:	ff87f493          	andi	s1,a5,-8
   114a0:	1807c663          	bltz	a5,1162c <_malloc_r+0x238>
   114a4:	18b4e463          	bltu	s1,a1,1162c <_malloc_r+0x238>
   114a8:	714000ef          	jal	11bbc <__malloc_lock>
   114ac:	1f700793          	li	a5,503
   114b0:	4097f063          	bgeu	a5,s1,118b0 <_malloc_r+0x4bc>
   114b4:	0094d793          	srli	a5,s1,0x9
   114b8:	18078263          	beqz	a5,1163c <_malloc_r+0x248>
   114bc:	00400713          	li	a4,4
   114c0:	34f76663          	bltu	a4,a5,1180c <_malloc_r+0x418>
   114c4:	0064d793          	srli	a5,s1,0x6
   114c8:	03978593          	addi	a1,a5,57
   114cc:	03878813          	addi	a6,a5,56
   114d0:	00359613          	slli	a2,a1,0x3
   114d4:	00011997          	auipc	s3,0x11
   114d8:	f2c98993          	addi	s3,s3,-212 # 22400 <__malloc_av_>
   114dc:	00c98633          	add	a2,s3,a2
   114e0:	00462403          	lw	s0,4(a2)
   114e4:	ff860613          	addi	a2,a2,-8
   114e8:	02860863          	beq	a2,s0,11518 <_malloc_r+0x124>
   114ec:	00f00513          	li	a0,15
   114f0:	0140006f          	j	11504 <_malloc_r+0x110>
   114f4:	00c42683          	lw	a3,12(s0)
   114f8:	28075e63          	bgez	a4,11794 <_malloc_r+0x3a0>
   114fc:	00d60e63          	beq	a2,a3,11518 <_malloc_r+0x124>
   11500:	00068413          	mv	s0,a3
   11504:	00442783          	lw	a5,4(s0)
   11508:	ffc7f793          	andi	a5,a5,-4
   1150c:	40978733          	sub	a4,a5,s1
   11510:	fee552e3          	bge	a0,a4,114f4 <_malloc_r+0x100>
   11514:	00080593          	mv	a1,a6
   11518:	0109a403          	lw	s0,16(s3)
   1151c:	00011897          	auipc	a7,0x11
   11520:	eec88893          	addi	a7,a7,-276 # 22408 <__malloc_av_+0x8>
   11524:	27140463          	beq	s0,a7,1178c <_malloc_r+0x398>
   11528:	00442783          	lw	a5,4(s0)
   1152c:	00f00693          	li	a3,15
   11530:	ffc7f793          	andi	a5,a5,-4
   11534:	40978733          	sub	a4,a5,s1
   11538:	38e6c263          	blt	a3,a4,118bc <_malloc_r+0x4c8>
   1153c:	0119aa23          	sw	a7,20(s3)
   11540:	0119a823          	sw	a7,16(s3)
   11544:	34075663          	bgez	a4,11890 <_malloc_r+0x49c>
   11548:	1ff00713          	li	a4,511
   1154c:	0049a503          	lw	a0,4(s3)
   11550:	24f76e63          	bltu	a4,a5,117ac <_malloc_r+0x3b8>
   11554:	ff87f713          	andi	a4,a5,-8
   11558:	00870713          	addi	a4,a4,8
   1155c:	00e98733          	add	a4,s3,a4
   11560:	00072683          	lw	a3,0(a4)
   11564:	0057d613          	srli	a2,a5,0x5
   11568:	00100793          	li	a5,1
   1156c:	00c797b3          	sll	a5,a5,a2
   11570:	00f56533          	or	a0,a0,a5
   11574:	ff870793          	addi	a5,a4,-8
   11578:	00f42623          	sw	a5,12(s0)
   1157c:	00d42423          	sw	a3,8(s0)
   11580:	00a9a223          	sw	a0,4(s3)
   11584:	00872023          	sw	s0,0(a4)
   11588:	0086a623          	sw	s0,12(a3)
   1158c:	4025d793          	srai	a5,a1,0x2
   11590:	00100613          	li	a2,1
   11594:	00f61633          	sll	a2,a2,a5
   11598:	0ac56a63          	bltu	a0,a2,1164c <_malloc_r+0x258>
   1159c:	00a677b3          	and	a5,a2,a0
   115a0:	02079463          	bnez	a5,115c8 <_malloc_r+0x1d4>
   115a4:	00161613          	slli	a2,a2,0x1
   115a8:	ffc5f593          	andi	a1,a1,-4
   115ac:	00a677b3          	and	a5,a2,a0
   115b0:	00458593          	addi	a1,a1,4
   115b4:	00079a63          	bnez	a5,115c8 <_malloc_r+0x1d4>
   115b8:	00161613          	slli	a2,a2,0x1
   115bc:	00a677b3          	and	a5,a2,a0
   115c0:	00458593          	addi	a1,a1,4
   115c4:	fe078ae3          	beqz	a5,115b8 <_malloc_r+0x1c4>
   115c8:	00f00813          	li	a6,15
   115cc:	00359313          	slli	t1,a1,0x3
   115d0:	00698333          	add	t1,s3,t1
   115d4:	00030513          	mv	a0,t1
   115d8:	00c52783          	lw	a5,12(a0)
   115dc:	00058e13          	mv	t3,a1
   115e0:	24f50863          	beq	a0,a5,11830 <_malloc_r+0x43c>
   115e4:	0047a703          	lw	a4,4(a5)
   115e8:	00078413          	mv	s0,a5
   115ec:	00c7a783          	lw	a5,12(a5)
   115f0:	ffc77713          	andi	a4,a4,-4
   115f4:	409706b3          	sub	a3,a4,s1
   115f8:	24d84863          	blt	a6,a3,11848 <_malloc_r+0x454>
   115fc:	fe06c2e3          	bltz	a3,115e0 <_malloc_r+0x1ec>
   11600:	00e40733          	add	a4,s0,a4
   11604:	00472683          	lw	a3,4(a4)
   11608:	00842603          	lw	a2,8(s0)
   1160c:	00090513          	mv	a0,s2
   11610:	0016e693          	ori	a3,a3,1
   11614:	00d72223          	sw	a3,4(a4)
   11618:	00f62623          	sw	a5,12(a2)
   1161c:	00c7a423          	sw	a2,8(a5)
   11620:	5a0000ef          	jal	11bc0 <__malloc_unlock>
   11624:	00840513          	addi	a0,s0,8
   11628:	e59ff06f          	j	11480 <_malloc_r+0x8c>
   1162c:	00c00793          	li	a5,12
   11630:	00f92023          	sw	a5,0(s2)
   11634:	00000513          	li	a0,0
   11638:	e49ff06f          	j	11480 <_malloc_r+0x8c>
   1163c:	20000613          	li	a2,512
   11640:	04000593          	li	a1,64
   11644:	03f00813          	li	a6,63
   11648:	e8dff06f          	j	114d4 <_malloc_r+0xe0>
   1164c:	0089a403          	lw	s0,8(s3)
   11650:	01612823          	sw	s6,16(sp)
   11654:	00442783          	lw	a5,4(s0)
   11658:	ffc7fb13          	andi	s6,a5,-4
   1165c:	009b6863          	bltu	s6,s1,1166c <_malloc_r+0x278>
   11660:	409b0733          	sub	a4,s6,s1
   11664:	00f00793          	li	a5,15
   11668:	0ee7c063          	blt	a5,a4,11748 <_malloc_r+0x354>
   1166c:	01912223          	sw	s9,4(sp)
   11670:	f6018c93          	addi	s9,gp,-160 # 229e0 <__malloc_sbrk_base>
   11674:	000ca703          	lw	a4,0(s9)
   11678:	01412c23          	sw	s4,24(sp)
   1167c:	01512a23          	sw	s5,20(sp)
   11680:	01712623          	sw	s7,12(sp)
   11684:	f7c1aa83          	lw	s5,-132(gp) # 229fc <__malloc_top_pad>
   11688:	fff00793          	li	a5,-1
   1168c:	01640a33          	add	s4,s0,s6
   11690:	01548ab3          	add	s5,s1,s5
   11694:	3cf70a63          	beq	a4,a5,11a68 <_malloc_r+0x674>
   11698:	000017b7          	lui	a5,0x1
   1169c:	00f78793          	addi	a5,a5,15 # 100f <exit-0xf0a5>
   116a0:	00fa8ab3          	add	s5,s5,a5
   116a4:	fffff7b7          	lui	a5,0xfffff
   116a8:	00fafab3          	and	s5,s5,a5
   116ac:	000a8593          	mv	a1,s5
   116b0:	00090513          	mv	a0,s2
   116b4:	30c050ef          	jal	169c0 <_sbrk_r>
   116b8:	fff00793          	li	a5,-1
   116bc:	00050b93          	mv	s7,a0
   116c0:	44f50e63          	beq	a0,a5,11b1c <_malloc_r+0x728>
   116c4:	01812423          	sw	s8,8(sp)
   116c8:	25456263          	bltu	a0,s4,1190c <_malloc_r+0x518>
   116cc:	1c818c13          	addi	s8,gp,456 # 22c48 <__malloc_current_mallinfo>
   116d0:	000c2583          	lw	a1,0(s8)
   116d4:	00ba85b3          	add	a1,s5,a1
   116d8:	00bc2023          	sw	a1,0(s8)
   116dc:	00058713          	mv	a4,a1
   116e0:	2aaa1a63          	bne	s4,a0,11994 <_malloc_r+0x5a0>
   116e4:	01451793          	slli	a5,a0,0x14
   116e8:	2a079663          	bnez	a5,11994 <_malloc_r+0x5a0>
   116ec:	0089ab83          	lw	s7,8(s3)
   116f0:	015b07b3          	add	a5,s6,s5
   116f4:	0017e793          	ori	a5,a5,1
   116f8:	00fba223          	sw	a5,4(s7)
   116fc:	f7818713          	addi	a4,gp,-136 # 229f8 <__malloc_max_sbrked_mem>
   11700:	00072683          	lw	a3,0(a4)
   11704:	00b6f463          	bgeu	a3,a1,1170c <_malloc_r+0x318>
   11708:	00b72023          	sw	a1,0(a4)
   1170c:	f7418713          	addi	a4,gp,-140 # 229f4 <__malloc_max_total_mem>
   11710:	00072683          	lw	a3,0(a4)
   11714:	00b6f463          	bgeu	a3,a1,1171c <_malloc_r+0x328>
   11718:	00b72023          	sw	a1,0(a4)
   1171c:	00812c03          	lw	s8,8(sp)
   11720:	000b8413          	mv	s0,s7
   11724:	ffc7f793          	andi	a5,a5,-4
   11728:	40978733          	sub	a4,a5,s1
   1172c:	3897ea63          	bltu	a5,s1,11ac0 <_malloc_r+0x6cc>
   11730:	00f00793          	li	a5,15
   11734:	38e7d663          	bge	a5,a4,11ac0 <_malloc_r+0x6cc>
   11738:	01812a03          	lw	s4,24(sp)
   1173c:	01412a83          	lw	s5,20(sp)
   11740:	00c12b83          	lw	s7,12(sp)
   11744:	00412c83          	lw	s9,4(sp)
   11748:	0014e793          	ori	a5,s1,1
   1174c:	00f42223          	sw	a5,4(s0)
   11750:	009404b3          	add	s1,s0,s1
   11754:	0099a423          	sw	s1,8(s3)
   11758:	00176713          	ori	a4,a4,1
   1175c:	00090513          	mv	a0,s2
   11760:	00e4a223          	sw	a4,4(s1)
   11764:	45c000ef          	jal	11bc0 <__malloc_unlock>
   11768:	02c12083          	lw	ra,44(sp)
   1176c:	00840513          	addi	a0,s0,8
   11770:	02812403          	lw	s0,40(sp)
   11774:	01012b03          	lw	s6,16(sp)
   11778:	02412483          	lw	s1,36(sp)
   1177c:	02012903          	lw	s2,32(sp)
   11780:	01c12983          	lw	s3,28(sp)
   11784:	03010113          	addi	sp,sp,48
   11788:	00008067          	ret
   1178c:	0049a503          	lw	a0,4(s3)
   11790:	dfdff06f          	j	1158c <_malloc_r+0x198>
   11794:	00842603          	lw	a2,8(s0)
   11798:	cc5ff06f          	j	1145c <_malloc_r+0x68>
   1179c:	00c7a403          	lw	s0,12(a5) # fffff00c <__BSS_END__+0xfffdc20c>
   117a0:	00258593          	addi	a1,a1,2
   117a4:	d6878ae3          	beq	a5,s0,11518 <_malloc_r+0x124>
   117a8:	ca5ff06f          	j	1144c <_malloc_r+0x58>
   117ac:	0097d713          	srli	a4,a5,0x9
   117b0:	00400693          	li	a3,4
   117b4:	14e6f263          	bgeu	a3,a4,118f8 <_malloc_r+0x504>
   117b8:	01400693          	li	a3,20
   117bc:	32e6e463          	bltu	a3,a4,11ae4 <_malloc_r+0x6f0>
   117c0:	05c70613          	addi	a2,a4,92
   117c4:	05b70693          	addi	a3,a4,91
   117c8:	00361613          	slli	a2,a2,0x3
   117cc:	00c98633          	add	a2,s3,a2
   117d0:	00062703          	lw	a4,0(a2)
   117d4:	ff860613          	addi	a2,a2,-8
   117d8:	00e61863          	bne	a2,a4,117e8 <_malloc_r+0x3f4>
   117dc:	2940006f          	j	11a70 <_malloc_r+0x67c>
   117e0:	00872703          	lw	a4,8(a4)
   117e4:	00e60863          	beq	a2,a4,117f4 <_malloc_r+0x400>
   117e8:	00472683          	lw	a3,4(a4)
   117ec:	ffc6f693          	andi	a3,a3,-4
   117f0:	fed7e8e3          	bltu	a5,a3,117e0 <_malloc_r+0x3ec>
   117f4:	00c72603          	lw	a2,12(a4)
   117f8:	00c42623          	sw	a2,12(s0)
   117fc:	00e42423          	sw	a4,8(s0)
   11800:	00862423          	sw	s0,8(a2)
   11804:	00872623          	sw	s0,12(a4)
   11808:	d85ff06f          	j	1158c <_malloc_r+0x198>
   1180c:	01400713          	li	a4,20
   11810:	10f77863          	bgeu	a4,a5,11920 <_malloc_r+0x52c>
   11814:	05400713          	li	a4,84
   11818:	2ef76463          	bltu	a4,a5,11b00 <_malloc_r+0x70c>
   1181c:	00c4d793          	srli	a5,s1,0xc
   11820:	06f78593          	addi	a1,a5,111
   11824:	06e78813          	addi	a6,a5,110
   11828:	00359613          	slli	a2,a1,0x3
   1182c:	ca9ff06f          	j	114d4 <_malloc_r+0xe0>
   11830:	001e0e13          	addi	t3,t3,1
   11834:	003e7793          	andi	a5,t3,3
   11838:	00850513          	addi	a0,a0,8
   1183c:	10078063          	beqz	a5,1193c <_malloc_r+0x548>
   11840:	00c52783          	lw	a5,12(a0)
   11844:	d9dff06f          	j	115e0 <_malloc_r+0x1ec>
   11848:	00842603          	lw	a2,8(s0)
   1184c:	0014e593          	ori	a1,s1,1
   11850:	00b42223          	sw	a1,4(s0)
   11854:	00f62623          	sw	a5,12(a2)
   11858:	00c7a423          	sw	a2,8(a5)
   1185c:	009404b3          	add	s1,s0,s1
   11860:	0099aa23          	sw	s1,20(s3)
   11864:	0099a823          	sw	s1,16(s3)
   11868:	0016e793          	ori	a5,a3,1
   1186c:	0114a623          	sw	a7,12(s1)
   11870:	0114a423          	sw	a7,8(s1)
   11874:	00f4a223          	sw	a5,4(s1)
   11878:	00e40733          	add	a4,s0,a4
   1187c:	00090513          	mv	a0,s2
   11880:	00d72023          	sw	a3,0(a4)
   11884:	33c000ef          	jal	11bc0 <__malloc_unlock>
   11888:	00840513          	addi	a0,s0,8
   1188c:	bf5ff06f          	j	11480 <_malloc_r+0x8c>
   11890:	00f407b3          	add	a5,s0,a5
   11894:	0047a703          	lw	a4,4(a5)
   11898:	00090513          	mv	a0,s2
   1189c:	00176713          	ori	a4,a4,1
   118a0:	00e7a223          	sw	a4,4(a5)
   118a4:	31c000ef          	jal	11bc0 <__malloc_unlock>
   118a8:	00840513          	addi	a0,s0,8
   118ac:	bd5ff06f          	j	11480 <_malloc_r+0x8c>
   118b0:	0034d593          	srli	a1,s1,0x3
   118b4:	00848793          	addi	a5,s1,8
   118b8:	b7dff06f          	j	11434 <_malloc_r+0x40>
   118bc:	0014e693          	ori	a3,s1,1
   118c0:	00d42223          	sw	a3,4(s0)
   118c4:	009404b3          	add	s1,s0,s1
   118c8:	0099aa23          	sw	s1,20(s3)
   118cc:	0099a823          	sw	s1,16(s3)
   118d0:	00176693          	ori	a3,a4,1
   118d4:	0114a623          	sw	a7,12(s1)
   118d8:	0114a423          	sw	a7,8(s1)
   118dc:	00d4a223          	sw	a3,4(s1)
   118e0:	00f407b3          	add	a5,s0,a5
   118e4:	00090513          	mv	a0,s2
   118e8:	00e7a023          	sw	a4,0(a5)
   118ec:	2d4000ef          	jal	11bc0 <__malloc_unlock>
   118f0:	00840513          	addi	a0,s0,8
   118f4:	b8dff06f          	j	11480 <_malloc_r+0x8c>
   118f8:	0067d713          	srli	a4,a5,0x6
   118fc:	03970613          	addi	a2,a4,57
   11900:	03870693          	addi	a3,a4,56
   11904:	00361613          	slli	a2,a2,0x3
   11908:	ec5ff06f          	j	117cc <_malloc_r+0x3d8>
   1190c:	07340c63          	beq	s0,s3,11984 <_malloc_r+0x590>
   11910:	0089a403          	lw	s0,8(s3)
   11914:	00812c03          	lw	s8,8(sp)
   11918:	00442783          	lw	a5,4(s0)
   1191c:	e09ff06f          	j	11724 <_malloc_r+0x330>
   11920:	05c78593          	addi	a1,a5,92
   11924:	05b78813          	addi	a6,a5,91
   11928:	00359613          	slli	a2,a1,0x3
   1192c:	ba9ff06f          	j	114d4 <_malloc_r+0xe0>
   11930:	00832783          	lw	a5,8(t1)
   11934:	fff58593          	addi	a1,a1,-1
   11938:	26679e63          	bne	a5,t1,11bb4 <_malloc_r+0x7c0>
   1193c:	0035f793          	andi	a5,a1,3
   11940:	ff830313          	addi	t1,t1,-8
   11944:	fe0796e3          	bnez	a5,11930 <_malloc_r+0x53c>
   11948:	0049a703          	lw	a4,4(s3)
   1194c:	fff64793          	not	a5,a2
   11950:	00e7f7b3          	and	a5,a5,a4
   11954:	00f9a223          	sw	a5,4(s3)
   11958:	00161613          	slli	a2,a2,0x1
   1195c:	cec7e8e3          	bltu	a5,a2,1164c <_malloc_r+0x258>
   11960:	ce0606e3          	beqz	a2,1164c <_malloc_r+0x258>
   11964:	00f67733          	and	a4,a2,a5
   11968:	00071a63          	bnez	a4,1197c <_malloc_r+0x588>
   1196c:	00161613          	slli	a2,a2,0x1
   11970:	00f67733          	and	a4,a2,a5
   11974:	004e0e13          	addi	t3,t3,4
   11978:	fe070ae3          	beqz	a4,1196c <_malloc_r+0x578>
   1197c:	000e0593          	mv	a1,t3
   11980:	c4dff06f          	j	115cc <_malloc_r+0x1d8>
   11984:	1c818c13          	addi	s8,gp,456 # 22c48 <__malloc_current_mallinfo>
   11988:	000c2703          	lw	a4,0(s8)
   1198c:	00ea8733          	add	a4,s5,a4
   11990:	00ec2023          	sw	a4,0(s8)
   11994:	000ca683          	lw	a3,0(s9)
   11998:	fff00793          	li	a5,-1
   1199c:	18f68663          	beq	a3,a5,11b28 <_malloc_r+0x734>
   119a0:	414b87b3          	sub	a5,s7,s4
   119a4:	00e787b3          	add	a5,a5,a4
   119a8:	00fc2023          	sw	a5,0(s8)
   119ac:	007bfc93          	andi	s9,s7,7
   119b0:	0c0c8c63          	beqz	s9,11a88 <_malloc_r+0x694>
   119b4:	419b8bb3          	sub	s7,s7,s9
   119b8:	000017b7          	lui	a5,0x1
   119bc:	00878793          	addi	a5,a5,8 # 1008 <exit-0xf0ac>
   119c0:	008b8b93          	addi	s7,s7,8
   119c4:	419785b3          	sub	a1,a5,s9
   119c8:	015b8ab3          	add	s5,s7,s5
   119cc:	415585b3          	sub	a1,a1,s5
   119d0:	01459593          	slli	a1,a1,0x14
   119d4:	0145da13          	srli	s4,a1,0x14
   119d8:	000a0593          	mv	a1,s4
   119dc:	00090513          	mv	a0,s2
   119e0:	7e1040ef          	jal	169c0 <_sbrk_r>
   119e4:	fff00793          	li	a5,-1
   119e8:	18f50063          	beq	a0,a5,11b68 <_malloc_r+0x774>
   119ec:	41750533          	sub	a0,a0,s7
   119f0:	01450ab3          	add	s5,a0,s4
   119f4:	000c2703          	lw	a4,0(s8)
   119f8:	0179a423          	sw	s7,8(s3)
   119fc:	001ae793          	ori	a5,s5,1
   11a00:	00ea05b3          	add	a1,s4,a4
   11a04:	00bc2023          	sw	a1,0(s8)
   11a08:	00fba223          	sw	a5,4(s7)
   11a0c:	cf3408e3          	beq	s0,s3,116fc <_malloc_r+0x308>
   11a10:	00f00693          	li	a3,15
   11a14:	0b66f063          	bgeu	a3,s6,11ab4 <_malloc_r+0x6c0>
   11a18:	00442703          	lw	a4,4(s0)
   11a1c:	ff4b0793          	addi	a5,s6,-12
   11a20:	ff87f793          	andi	a5,a5,-8
   11a24:	00177713          	andi	a4,a4,1
   11a28:	00f76733          	or	a4,a4,a5
   11a2c:	00e42223          	sw	a4,4(s0)
   11a30:	00500613          	li	a2,5
   11a34:	00f40733          	add	a4,s0,a5
   11a38:	00c72223          	sw	a2,4(a4)
   11a3c:	00c72423          	sw	a2,8(a4)
   11a40:	00f6e663          	bltu	a3,a5,11a4c <_malloc_r+0x658>
   11a44:	004ba783          	lw	a5,4(s7)
   11a48:	cb5ff06f          	j	116fc <_malloc_r+0x308>
   11a4c:	00840593          	addi	a1,s0,8
   11a50:	00090513          	mv	a0,s2
   11a54:	e9cff0ef          	jal	110f0 <_free_r>
   11a58:	0089ab83          	lw	s7,8(s3)
   11a5c:	000c2583          	lw	a1,0(s8)
   11a60:	004ba783          	lw	a5,4(s7)
   11a64:	c99ff06f          	j	116fc <_malloc_r+0x308>
   11a68:	010a8a93          	addi	s5,s5,16
   11a6c:	c41ff06f          	j	116ac <_malloc_r+0x2b8>
   11a70:	4026d693          	srai	a3,a3,0x2
   11a74:	00100793          	li	a5,1
   11a78:	00d797b3          	sll	a5,a5,a3
   11a7c:	00f56533          	or	a0,a0,a5
   11a80:	00a9a223          	sw	a0,4(s3)
   11a84:	d75ff06f          	j	117f8 <_malloc_r+0x404>
   11a88:	015b85b3          	add	a1,s7,s5
   11a8c:	40b005b3          	neg	a1,a1
   11a90:	01459593          	slli	a1,a1,0x14
   11a94:	0145da13          	srli	s4,a1,0x14
   11a98:	000a0593          	mv	a1,s4
   11a9c:	00090513          	mv	a0,s2
   11aa0:	721040ef          	jal	169c0 <_sbrk_r>
   11aa4:	fff00793          	li	a5,-1
   11aa8:	f4f512e3          	bne	a0,a5,119ec <_malloc_r+0x5f8>
   11aac:	00000a13          	li	s4,0
   11ab0:	f45ff06f          	j	119f4 <_malloc_r+0x600>
   11ab4:	00812c03          	lw	s8,8(sp)
   11ab8:	00100793          	li	a5,1
   11abc:	00fba223          	sw	a5,4(s7)
   11ac0:	00090513          	mv	a0,s2
   11ac4:	0fc000ef          	jal	11bc0 <__malloc_unlock>
   11ac8:	00000513          	li	a0,0
   11acc:	01812a03          	lw	s4,24(sp)
   11ad0:	01412a83          	lw	s5,20(sp)
   11ad4:	01012b03          	lw	s6,16(sp)
   11ad8:	00c12b83          	lw	s7,12(sp)
   11adc:	00412c83          	lw	s9,4(sp)
   11ae0:	9a1ff06f          	j	11480 <_malloc_r+0x8c>
   11ae4:	05400693          	li	a3,84
   11ae8:	04e6e463          	bltu	a3,a4,11b30 <_malloc_r+0x73c>
   11aec:	00c7d713          	srli	a4,a5,0xc
   11af0:	06f70613          	addi	a2,a4,111
   11af4:	06e70693          	addi	a3,a4,110
   11af8:	00361613          	slli	a2,a2,0x3
   11afc:	cd1ff06f          	j	117cc <_malloc_r+0x3d8>
   11b00:	15400713          	li	a4,340
   11b04:	04f76463          	bltu	a4,a5,11b4c <_malloc_r+0x758>
   11b08:	00f4d793          	srli	a5,s1,0xf
   11b0c:	07878593          	addi	a1,a5,120
   11b10:	07778813          	addi	a6,a5,119
   11b14:	00359613          	slli	a2,a1,0x3
   11b18:	9bdff06f          	j	114d4 <_malloc_r+0xe0>
   11b1c:	0089a403          	lw	s0,8(s3)
   11b20:	00442783          	lw	a5,4(s0)
   11b24:	c01ff06f          	j	11724 <_malloc_r+0x330>
   11b28:	017ca023          	sw	s7,0(s9)
   11b2c:	e81ff06f          	j	119ac <_malloc_r+0x5b8>
   11b30:	15400693          	li	a3,340
   11b34:	04e6e463          	bltu	a3,a4,11b7c <_malloc_r+0x788>
   11b38:	00f7d713          	srli	a4,a5,0xf
   11b3c:	07870613          	addi	a2,a4,120
   11b40:	07770693          	addi	a3,a4,119
   11b44:	00361613          	slli	a2,a2,0x3
   11b48:	c85ff06f          	j	117cc <_malloc_r+0x3d8>
   11b4c:	55400713          	li	a4,1364
   11b50:	04f76463          	bltu	a4,a5,11b98 <_malloc_r+0x7a4>
   11b54:	0124d793          	srli	a5,s1,0x12
   11b58:	07d78593          	addi	a1,a5,125
   11b5c:	07c78813          	addi	a6,a5,124
   11b60:	00359613          	slli	a2,a1,0x3
   11b64:	971ff06f          	j	114d4 <_malloc_r+0xe0>
   11b68:	ff8c8c93          	addi	s9,s9,-8
   11b6c:	019a8ab3          	add	s5,s5,s9
   11b70:	417a8ab3          	sub	s5,s5,s7
   11b74:	00000a13          	li	s4,0
   11b78:	e7dff06f          	j	119f4 <_malloc_r+0x600>
   11b7c:	55400693          	li	a3,1364
   11b80:	02e6e463          	bltu	a3,a4,11ba8 <_malloc_r+0x7b4>
   11b84:	0127d713          	srli	a4,a5,0x12
   11b88:	07d70613          	addi	a2,a4,125
   11b8c:	07c70693          	addi	a3,a4,124
   11b90:	00361613          	slli	a2,a2,0x3
   11b94:	c39ff06f          	j	117cc <_malloc_r+0x3d8>
   11b98:	3f800613          	li	a2,1016
   11b9c:	07f00593          	li	a1,127
   11ba0:	07e00813          	li	a6,126
   11ba4:	931ff06f          	j	114d4 <_malloc_r+0xe0>
   11ba8:	3f800613          	li	a2,1016
   11bac:	07e00693          	li	a3,126
   11bb0:	c1dff06f          	j	117cc <_malloc_r+0x3d8>
   11bb4:	0049a783          	lw	a5,4(s3)
   11bb8:	da1ff06f          	j	11958 <_malloc_r+0x564>

00011bbc <__malloc_lock>:
   11bbc:	00008067          	ret

00011bc0 <__malloc_unlock>:
   11bc0:	00008067          	ret

00011bc4 <_vfprintf_r>:
   11bc4:	e3010113          	addi	sp,sp,-464
   11bc8:	1c112623          	sw	ra,460(sp)
   11bcc:	1c812423          	sw	s0,456(sp)
   11bd0:	1c912223          	sw	s1,452(sp)
   11bd4:	1d212023          	sw	s2,448(sp)
   11bd8:	00058493          	mv	s1,a1
   11bdc:	00060913          	mv	s2,a2
   11be0:	00d12a23          	sw	a3,20(sp)
   11be4:	00050413          	mv	s0,a0
   11be8:	00a12423          	sw	a0,8(sp)
   11bec:	5c5040ef          	jal	169b0 <_localeconv_r>
   11bf0:	00052703          	lw	a4,0(a0)
   11bf4:	00070513          	mv	a0,a4
   11bf8:	02e12623          	sw	a4,44(sp)
   11bfc:	9e4ff0ef          	jal	10de0 <strlen>
   11c00:	02a12423          	sw	a0,40(sp)
   11c04:	0c012823          	sw	zero,208(sp)
   11c08:	0c012a23          	sw	zero,212(sp)
   11c0c:	0c012c23          	sw	zero,216(sp)
   11c10:	0c012e23          	sw	zero,220(sp)
   11c14:	00040863          	beqz	s0,11c24 <_vfprintf_r+0x60>
   11c18:	03442703          	lw	a4,52(s0)
   11c1c:	00071463          	bnez	a4,11c24 <_vfprintf_r+0x60>
   11c20:	10d0106f          	j	1352c <_vfprintf_r+0x1968>
   11c24:	00c49703          	lh	a4,12(s1)
   11c28:	01271693          	slli	a3,a4,0x12
   11c2c:	0206c663          	bltz	a3,11c58 <_vfprintf_r+0x94>
   11c30:	0644a683          	lw	a3,100(s1)
   11c34:	000025b7          	lui	a1,0x2
   11c38:	ffffe637          	lui	a2,0xffffe
   11c3c:	00b76733          	or	a4,a4,a1
   11c40:	fff60613          	addi	a2,a2,-1 # ffffdfff <__BSS_END__+0xfffdb1ff>
   11c44:	01071713          	slli	a4,a4,0x10
   11c48:	41075713          	srai	a4,a4,0x10
   11c4c:	00c6f6b3          	and	a3,a3,a2
   11c50:	00e49623          	sh	a4,12(s1)
   11c54:	06d4a223          	sw	a3,100(s1)
   11c58:	00877693          	andi	a3,a4,8
   11c5c:	2e068e63          	beqz	a3,11f58 <_vfprintf_r+0x394>
   11c60:	0104a683          	lw	a3,16(s1)
   11c64:	2e068a63          	beqz	a3,11f58 <_vfprintf_r+0x394>
   11c68:	01a77713          	andi	a4,a4,26
   11c6c:	00a00693          	li	a3,10
   11c70:	30d70663          	beq	a4,a3,11f7c <_vfprintf_r+0x3b8>
   11c74:	1b312e23          	sw	s3,444(sp)
   11c78:	1b412c23          	sw	s4,440(sp)
   11c7c:	1ba12023          	sw	s10,416(sp)
   11c80:	1b512a23          	sw	s5,436(sp)
   11c84:	1b612823          	sw	s6,432(sp)
   11c88:	1b712623          	sw	s7,428(sp)
   11c8c:	1b812423          	sw	s8,424(sp)
   11c90:	1b912223          	sw	s9,420(sp)
   11c94:	19b12e23          	sw	s11,412(sp)
   11c98:	00090d13          	mv	s10,s2
   11c9c:	000d4703          	lbu	a4,0(s10)
   11ca0:	0ec10993          	addi	s3,sp,236
   11ca4:	0d312223          	sw	s3,196(sp)
   11ca8:	0c012623          	sw	zero,204(sp)
   11cac:	0c012423          	sw	zero,200(sp)
   11cb0:	00012e23          	sw	zero,28(sp)
   11cb4:	02012823          	sw	zero,48(sp)
   11cb8:	02012e23          	sw	zero,60(sp)
   11cbc:	02012c23          	sw	zero,56(sp)
   11cc0:	04012223          	sw	zero,68(sp)
   11cc4:	04012023          	sw	zero,64(sp)
   11cc8:	00012623          	sw	zero,12(sp)
   11ccc:	01000413          	li	s0,16
   11cd0:	00098a13          	mv	s4,s3
   11cd4:	22070463          	beqz	a4,11efc <_vfprintf_r+0x338>
   11cd8:	000d0a93          	mv	s5,s10
   11cdc:	02500693          	li	a3,37
   11ce0:	3ed70e63          	beq	a4,a3,120dc <_vfprintf_r+0x518>
   11ce4:	001ac703          	lbu	a4,1(s5)
   11ce8:	001a8a93          	addi	s5,s5,1
   11cec:	fe071ae3          	bnez	a4,11ce0 <_vfprintf_r+0x11c>
   11cf0:	41aa8933          	sub	s2,s5,s10
   11cf4:	21aa8463          	beq	s5,s10,11efc <_vfprintf_r+0x338>
   11cf8:	0cc12683          	lw	a3,204(sp)
   11cfc:	0c812703          	lw	a4,200(sp)
   11d00:	01aa2023          	sw	s10,0(s4)
   11d04:	012686b3          	add	a3,a3,s2
   11d08:	00170713          	addi	a4,a4,1
   11d0c:	012a2223          	sw	s2,4(s4)
   11d10:	0cd12623          	sw	a3,204(sp)
   11d14:	0ce12423          	sw	a4,200(sp)
   11d18:	00700693          	li	a3,7
   11d1c:	008a0a13          	addi	s4,s4,8
   11d20:	3ce6c663          	blt	a3,a4,120ec <_vfprintf_r+0x528>
   11d24:	00c12783          	lw	a5,12(sp)
   11d28:	000ac703          	lbu	a4,0(s5)
   11d2c:	012787b3          	add	a5,a5,s2
   11d30:	00f12623          	sw	a5,12(sp)
   11d34:	1c070463          	beqz	a4,11efc <_vfprintf_r+0x338>
   11d38:	001ac883          	lbu	a7,1(s5)
   11d3c:	0a0103a3          	sb	zero,167(sp)
   11d40:	001a8a93          	addi	s5,s5,1
   11d44:	fff00b13          	li	s6,-1
   11d48:	00000b93          	li	s7,0
   11d4c:	00000c93          	li	s9,0
   11d50:	05a00913          	li	s2,90
   11d54:	001a8a93          	addi	s5,s5,1
   11d58:	fe088793          	addi	a5,a7,-32
   11d5c:	04f96a63          	bltu	s2,a5,11db0 <_vfprintf_r+0x1ec>
   11d60:	0000f717          	auipc	a4,0xf
   11d64:	d0c70713          	addi	a4,a4,-756 # 20a6c <_exit+0x234>
   11d68:	00279793          	slli	a5,a5,0x2
   11d6c:	00e787b3          	add	a5,a5,a4
   11d70:	0007a783          	lw	a5,0(a5)
   11d74:	00e787b3          	add	a5,a5,a4
   11d78:	00078067          	jr	a5
   11d7c:	00000b93          	li	s7,0
   11d80:	fd088793          	addi	a5,a7,-48
   11d84:	00900693          	li	a3,9
   11d88:	000ac883          	lbu	a7,0(s5)
   11d8c:	002b9713          	slli	a4,s7,0x2
   11d90:	01770bb3          	add	s7,a4,s7
   11d94:	001b9b93          	slli	s7,s7,0x1
   11d98:	01778bb3          	add	s7,a5,s7
   11d9c:	fd088793          	addi	a5,a7,-48
   11da0:	001a8a93          	addi	s5,s5,1
   11da4:	fef6f2e3          	bgeu	a3,a5,11d88 <_vfprintf_r+0x1c4>
   11da8:	fe088793          	addi	a5,a7,-32
   11dac:	faf97ae3          	bgeu	s2,a5,11d60 <_vfprintf_r+0x19c>
   11db0:	14088663          	beqz	a7,11efc <_vfprintf_r+0x338>
   11db4:	13110623          	sb	a7,300(sp)
   11db8:	0a0103a3          	sb	zero,167(sp)
   11dbc:	00100d93          	li	s11,1
   11dc0:	00100913          	li	s2,1
   11dc4:	12c10d13          	addi	s10,sp,300
   11dc8:	00012823          	sw	zero,16(sp)
   11dcc:	00000b13          	li	s6,0
   11dd0:	02012223          	sw	zero,36(sp)
   11dd4:	02012023          	sw	zero,32(sp)
   11dd8:	00012c23          	sw	zero,24(sp)
   11ddc:	002cf293          	andi	t0,s9,2
   11de0:	00028463          	beqz	t0,11de8 <_vfprintf_r+0x224>
   11de4:	002d8d93          	addi	s11,s11,2
   11de8:	084cff93          	andi	t6,s9,132
   11dec:	0cc12603          	lw	a2,204(sp)
   11df0:	000f9663          	bnez	t6,11dfc <_vfprintf_r+0x238>
   11df4:	41bb8733          	sub	a4,s7,s11
   11df8:	46e04ae3          	bgtz	a4,12a6c <_vfprintf_r+0xea8>
   11dfc:	0a714703          	lbu	a4,167(sp)
   11e00:	02070a63          	beqz	a4,11e34 <_vfprintf_r+0x270>
   11e04:	0c812703          	lw	a4,200(sp)
   11e08:	0a710593          	addi	a1,sp,167
   11e0c:	00ba2023          	sw	a1,0(s4)
   11e10:	00160613          	addi	a2,a2,1
   11e14:	00100593          	li	a1,1
   11e18:	00170713          	addi	a4,a4,1
   11e1c:	00ba2223          	sw	a1,4(s4)
   11e20:	0cc12623          	sw	a2,204(sp)
   11e24:	0ce12423          	sw	a4,200(sp)
   11e28:	00700593          	li	a1,7
   11e2c:	008a0a13          	addi	s4,s4,8
   11e30:	3ee5c663          	blt	a1,a4,1221c <_vfprintf_r+0x658>
   11e34:	02028a63          	beqz	t0,11e68 <_vfprintf_r+0x2a4>
   11e38:	0c812703          	lw	a4,200(sp)
   11e3c:	00200593          	li	a1,2
   11e40:	00260613          	addi	a2,a2,2
   11e44:	00170713          	addi	a4,a4,1
   11e48:	0a810793          	addi	a5,sp,168
   11e4c:	00ba2223          	sw	a1,4(s4)
   11e50:	00fa2023          	sw	a5,0(s4)
   11e54:	0cc12623          	sw	a2,204(sp)
   11e58:	0ce12423          	sw	a4,200(sp)
   11e5c:	00700593          	li	a1,7
   11e60:	008a0a13          	addi	s4,s4,8
   11e64:	50e5cae3          	blt	a1,a4,12b78 <_vfprintf_r+0xfb4>
   11e68:	08000713          	li	a4,128
   11e6c:	1eef86e3          	beq	t6,a4,12858 <_vfprintf_r+0xc94>
   11e70:	412b0b33          	sub	s6,s6,s2
   11e74:	2f6040e3          	bgtz	s6,12954 <_vfprintf_r+0xd90>
   11e78:	100cf713          	andi	a4,s9,256
   11e7c:	040712e3          	bnez	a4,126c0 <_vfprintf_r+0xafc>
   11e80:	0c812783          	lw	a5,200(sp)
   11e84:	01260633          	add	a2,a2,s2
   11e88:	01aa2023          	sw	s10,0(s4)
   11e8c:	00178793          	addi	a5,a5,1
   11e90:	012a2223          	sw	s2,4(s4)
   11e94:	0cc12623          	sw	a2,204(sp)
   11e98:	0cf12423          	sw	a5,200(sp)
   11e9c:	00700713          	li	a4,7
   11ea0:	4af74e63          	blt	a4,a5,1235c <_vfprintf_r+0x798>
   11ea4:	008a0a13          	addi	s4,s4,8
   11ea8:	004cfe13          	andi	t3,s9,4
   11eac:	000e0663          	beqz	t3,11eb8 <_vfprintf_r+0x2f4>
   11eb0:	41bb8933          	sub	s2,s7,s11
   11eb4:	4f204ae3          	bgtz	s2,12ba8 <_vfprintf_r+0xfe4>
   11eb8:	000b8313          	mv	t1,s7
   11ebc:	01bbd463          	bge	s7,s11,11ec4 <_vfprintf_r+0x300>
   11ec0:	000d8313          	mv	t1,s11
   11ec4:	00c12783          	lw	a5,12(sp)
   11ec8:	006787b3          	add	a5,a5,t1
   11ecc:	00f12623          	sw	a5,12(sp)
   11ed0:	360614e3          	bnez	a2,12a38 <_vfprintf_r+0xe74>
   11ed4:	01012783          	lw	a5,16(sp)
   11ed8:	0c012423          	sw	zero,200(sp)
   11edc:	00078863          	beqz	a5,11eec <_vfprintf_r+0x328>
   11ee0:	01012583          	lw	a1,16(sp)
   11ee4:	00812503          	lw	a0,8(sp)
   11ee8:	a08ff0ef          	jal	110f0 <_free_r>
   11eec:	00098a13          	mv	s4,s3
   11ef0:	000a8d13          	mv	s10,s5
   11ef4:	000d4703          	lbu	a4,0(s10)
   11ef8:	de0710e3          	bnez	a4,11cd8 <_vfprintf_r+0x114>
   11efc:	0cc12783          	lw	a5,204(sp)
   11f00:	00078463          	beqz	a5,11f08 <_vfprintf_r+0x344>
   11f04:	3a10106f          	j	13aa4 <_vfprintf_r+0x1ee0>
   11f08:	00c4d783          	lhu	a5,12(s1)
   11f0c:	1bc12983          	lw	s3,444(sp)
   11f10:	1b812a03          	lw	s4,440(sp)
   11f14:	0407f793          	andi	a5,a5,64
   11f18:	1b412a83          	lw	s5,436(sp)
   11f1c:	1b012b03          	lw	s6,432(sp)
   11f20:	1ac12b83          	lw	s7,428(sp)
   11f24:	1a812c03          	lw	s8,424(sp)
   11f28:	1a412c83          	lw	s9,420(sp)
   11f2c:	1a012d03          	lw	s10,416(sp)
   11f30:	19c12d83          	lw	s11,412(sp)
   11f34:	00078463          	beqz	a5,11f3c <_vfprintf_r+0x378>
   11f38:	29c0206f          	j	141d4 <_vfprintf_r+0x2610>
   11f3c:	1cc12083          	lw	ra,460(sp)
   11f40:	1c812403          	lw	s0,456(sp)
   11f44:	00c12503          	lw	a0,12(sp)
   11f48:	1c412483          	lw	s1,452(sp)
   11f4c:	1c012903          	lw	s2,448(sp)
   11f50:	1d010113          	addi	sp,sp,464
   11f54:	00008067          	ret
   11f58:	00812503          	lw	a0,8(sp)
   11f5c:	00048593          	mv	a1,s1
   11f60:	4fc040ef          	jal	1645c <__swsetup_r>
   11f64:	00050463          	beqz	a0,11f6c <_vfprintf_r+0x3a8>
   11f68:	26c0206f          	j	141d4 <_vfprintf_r+0x2610>
   11f6c:	00c49703          	lh	a4,12(s1)
   11f70:	00a00693          	li	a3,10
   11f74:	01a77713          	andi	a4,a4,26
   11f78:	ced71ee3          	bne	a4,a3,11c74 <_vfprintf_r+0xb0>
   11f7c:	00e49703          	lh	a4,14(s1)
   11f80:	ce074ae3          	bltz	a4,11c74 <_vfprintf_r+0xb0>
   11f84:	01412683          	lw	a3,20(sp)
   11f88:	00812503          	lw	a0,8(sp)
   11f8c:	00090613          	mv	a2,s2
   11f90:	00048593          	mv	a1,s1
   11f94:	52c020ef          	jal	144c0 <__sbprintf>
   11f98:	00a12623          	sw	a0,12(sp)
   11f9c:	fa1ff06f          	j	11f3c <_vfprintf_r+0x378>
   11fa0:	00812c03          	lw	s8,8(sp)
   11fa4:	000c0513          	mv	a0,s8
   11fa8:	209040ef          	jal	169b0 <_localeconv_r>
   11fac:	00452783          	lw	a5,4(a0)
   11fb0:	00078513          	mv	a0,a5
   11fb4:	04f12023          	sw	a5,64(sp)
   11fb8:	e29fe0ef          	jal	10de0 <strlen>
   11fbc:	00050793          	mv	a5,a0
   11fc0:	000c0513          	mv	a0,s8
   11fc4:	04f12223          	sw	a5,68(sp)
   11fc8:	00078c13          	mv	s8,a5
   11fcc:	1e5040ef          	jal	169b0 <_localeconv_r>
   11fd0:	00852703          	lw	a4,8(a0)
   11fd4:	02e12c23          	sw	a4,56(sp)
   11fd8:	740c1ce3          	bnez	s8,12f30 <_vfprintf_r+0x136c>
   11fdc:	000ac883          	lbu	a7,0(s5)
   11fe0:	d75ff06f          	j	11d54 <_vfprintf_r+0x190>
   11fe4:	000ac883          	lbu	a7,0(s5)
   11fe8:	020cec93          	ori	s9,s9,32
   11fec:	d69ff06f          	j	11d54 <_vfprintf_r+0x190>
   11ff0:	010cec93          	ori	s9,s9,16
   11ff4:	020cf793          	andi	a5,s9,32
   11ff8:	3a078a63          	beqz	a5,123ac <_vfprintf_r+0x7e8>
   11ffc:	01412783          	lw	a5,20(sp)
   12000:	00778c13          	addi	s8,a5,7
   12004:	ff8c7c13          	andi	s8,s8,-8
   12008:	004c2783          	lw	a5,4(s8)
   1200c:	000c2903          	lw	s2,0(s8)
   12010:	008c0713          	addi	a4,s8,8
   12014:	00e12a23          	sw	a4,20(sp)
   12018:	00078d93          	mv	s11,a5
   1201c:	3c07c263          	bltz	a5,123e0 <_vfprintf_r+0x81c>
   12020:	000c8e93          	mv	t4,s9
   12024:	4e0b4663          	bltz	s6,12510 <_vfprintf_r+0x94c>
   12028:	01b967b3          	or	a5,s2,s11
   1202c:	f7fcfe93          	andi	t4,s9,-129
   12030:	4e079063          	bnez	a5,12510 <_vfprintf_r+0x94c>
   12034:	4e0b1463          	bnez	s6,1251c <_vfprintf_r+0x958>
   12038:	00000913          	li	s2,0
   1203c:	000e8c93          	mv	s9,t4
   12040:	19010d13          	addi	s10,sp,400
   12044:	0a714703          	lbu	a4,167(sp)
   12048:	000b0d93          	mv	s11,s6
   1204c:	012b5463          	bge	s6,s2,12054 <_vfprintf_r+0x490>
   12050:	00090d93          	mv	s11,s2
   12054:	00012823          	sw	zero,16(sp)
   12058:	02012223          	sw	zero,36(sp)
   1205c:	02012023          	sw	zero,32(sp)
   12060:	00012c23          	sw	zero,24(sp)
   12064:	d6070ce3          	beqz	a4,11ddc <_vfprintf_r+0x218>
   12068:	001d8d93          	addi	s11,s11,1
   1206c:	d71ff06f          	j	11ddc <_vfprintf_r+0x218>
   12070:	010cec93          	ori	s9,s9,16
   12074:	020cf793          	andi	a5,s9,32
   12078:	30078263          	beqz	a5,1237c <_vfprintf_r+0x7b8>
   1207c:	01412783          	lw	a5,20(sp)
   12080:	00778c13          	addi	s8,a5,7
   12084:	ff8c7c13          	andi	s8,s8,-8
   12088:	000c2903          	lw	s2,0(s8)
   1208c:	004c2d83          	lw	s11,4(s8)
   12090:	008c0793          	addi	a5,s8,8
   12094:	00f12a23          	sw	a5,20(sp)
   12098:	bffcfe93          	andi	t4,s9,-1025
   1209c:	00000793          	li	a5,0
   120a0:	00000713          	li	a4,0
   120a4:	0ae103a3          	sb	a4,167(sp)
   120a8:	340b4e63          	bltz	s6,12404 <_vfprintf_r+0x840>
   120ac:	01b96733          	or	a4,s2,s11
   120b0:	f7fefc93          	andi	s9,t4,-129
   120b4:	1a0718e3          	bnez	a4,12a64 <_vfprintf_r+0xea0>
   120b8:	740b1263          	bnez	s6,127fc <_vfprintf_r+0xc38>
   120bc:	5e0796e3          	bnez	a5,12ea8 <_vfprintf_r+0x12e4>
   120c0:	001ef913          	andi	s2,t4,1
   120c4:	19010d13          	addi	s10,sp,400
   120c8:	f6090ee3          	beqz	s2,12044 <_vfprintf_r+0x480>
   120cc:	03000793          	li	a5,48
   120d0:	18f107a3          	sb	a5,399(sp)
   120d4:	18f10d13          	addi	s10,sp,399
   120d8:	f6dff06f          	j	12044 <_vfprintf_r+0x480>
   120dc:	41aa8933          	sub	s2,s5,s10
   120e0:	c1aa9ce3          	bne	s5,s10,11cf8 <_vfprintf_r+0x134>
   120e4:	000ac703          	lbu	a4,0(s5)
   120e8:	c4dff06f          	j	11d34 <_vfprintf_r+0x170>
   120ec:	00812503          	lw	a0,8(sp)
   120f0:	0c410613          	addi	a2,sp,196
   120f4:	00048593          	mv	a1,s1
   120f8:	5b8020ef          	jal	146b0 <__sprint_r>
   120fc:	e00516e3          	bnez	a0,11f08 <_vfprintf_r+0x344>
   12100:	00098a13          	mv	s4,s3
   12104:	c21ff06f          	j	11d24 <_vfprintf_r+0x160>
   12108:	008cf713          	andi	a4,s9,8
   1210c:	600710e3          	bnez	a4,12f0c <_vfprintf_r+0x1348>
   12110:	01412783          	lw	a5,20(sp)
   12114:	09010513          	addi	a0,sp,144
   12118:	01112823          	sw	a7,16(sp)
   1211c:	00778c13          	addi	s8,a5,7
   12120:	ff8c7c13          	andi	s8,s8,-8
   12124:	000c2583          	lw	a1,0(s8)
   12128:	004c2603          	lw	a2,4(s8)
   1212c:	008c0793          	addi	a5,s8,8
   12130:	00f12a23          	sw	a5,20(sp)
   12134:	3fc0e0ef          	jal	20530 <__extenddftf2>
   12138:	09012583          	lw	a1,144(sp)
   1213c:	09412603          	lw	a2,148(sp)
   12140:	09812683          	lw	a3,152(sp)
   12144:	09c12703          	lw	a4,156(sp)
   12148:	01012883          	lw	a7,16(sp)
   1214c:	0d010513          	addi	a0,sp,208
   12150:	01112823          	sw	a7,16(sp)
   12154:	0ce12e23          	sw	a4,220(sp)
   12158:	0cb12823          	sw	a1,208(sp)
   1215c:	0cc12a23          	sw	a2,212(sp)
   12160:	0cd12c23          	sw	a3,216(sp)
   12164:	244050ef          	jal	173a8 <_ldcheck>
   12168:	0aa12623          	sw	a0,172(sp)
   1216c:	00200713          	li	a4,2
   12170:	01012883          	lw	a7,16(sp)
   12174:	00e51463          	bne	a0,a4,1217c <_vfprintf_r+0x5b8>
   12178:	3f40106f          	j	1356c <_vfprintf_r+0x19a8>
   1217c:	00100713          	li	a4,1
   12180:	00e51463          	bne	a0,a4,12188 <_vfprintf_r+0x5c4>
   12184:	5540106f          	j	136d8 <_vfprintf_r+0x1b14>
   12188:	06100713          	li	a4,97
   1218c:	00e89463          	bne	a7,a4,12194 <_vfprintf_r+0x5d0>
   12190:	7a50006f          	j	13134 <_vfprintf_r+0x1570>
   12194:	04100713          	li	a4,65
   12198:	05800793          	li	a5,88
   1219c:	78e88ee3          	beq	a7,a4,13138 <_vfprintf_r+0x1574>
   121a0:	fff00713          	li	a4,-1
   121a4:	00eb1463          	bne	s6,a4,121ac <_vfprintf_r+0x5e8>
   121a8:	16c0206f          	j	14314 <_vfprintf_r+0x2750>
   121ac:	fdf8f713          	andi	a4,a7,-33
   121b0:	04700693          	li	a3,71
   121b4:	00012823          	sw	zero,16(sp)
   121b8:	00d71663          	bne	a4,a3,121c4 <_vfprintf_r+0x600>
   121bc:	000b1463          	bnez	s6,121c4 <_vfprintf_r+0x600>
   121c0:	00100b13          	li	s6,1
   121c4:	0dc12c03          	lw	s8,220(sp)
   121c8:	100ce793          	ori	a5,s9,256
   121cc:	04f12423          	sw	a5,72(sp)
   121d0:	02012a23          	sw	zero,52(sp)
   121d4:	0d012f83          	lw	t6,208(sp)
   121d8:	0d412f03          	lw	t5,212(sp)
   121dc:	0d812e83          	lw	t4,216(sp)
   121e0:	000c5a63          	bgez	s8,121f4 <_vfprintf_r+0x630>
   121e4:	80000737          	lui	a4,0x80000
   121e8:	02d00793          	li	a5,45
   121ec:	01874c33          	xor	s8,a4,s8
   121f0:	02f12a23          	sw	a5,52(sp)
   121f4:	fbf88713          	addi	a4,a7,-65
   121f8:	02500693          	li	a3,37
   121fc:	78e6e2e3          	bltu	a3,a4,13180 <_vfprintf_r+0x15bc>
   12200:	0000f697          	auipc	a3,0xf
   12204:	9d868693          	addi	a3,a3,-1576 # 20bd8 <_exit+0x3a0>
   12208:	00271713          	slli	a4,a4,0x2
   1220c:	00d70733          	add	a4,a4,a3
   12210:	00072703          	lw	a4,0(a4) # 80000000 <__BSS_END__+0x7ffdd200>
   12214:	00d70733          	add	a4,a4,a3
   12218:	00070067          	jr	a4
   1221c:	00812503          	lw	a0,8(sp)
   12220:	0c410613          	addi	a2,sp,196
   12224:	00048593          	mv	a1,s1
   12228:	05112623          	sw	a7,76(sp)
   1222c:	05f12423          	sw	t6,72(sp)
   12230:	02512a23          	sw	t0,52(sp)
   12234:	47c020ef          	jal	146b0 <__sprint_r>
   12238:	00051ae3          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   1223c:	0cc12603          	lw	a2,204(sp)
   12240:	04c12883          	lw	a7,76(sp)
   12244:	04812f83          	lw	t6,72(sp)
   12248:	03412283          	lw	t0,52(sp)
   1224c:	00098a13          	mv	s4,s3
   12250:	be5ff06f          	j	11e34 <_vfprintf_r+0x270>
   12254:	0c812903          	lw	s2,200(sp)
   12258:	01c12783          	lw	a5,28(sp)
   1225c:	00100713          	li	a4,1
   12260:	01aa2023          	sw	s10,0(s4)
   12264:	00160c13          	addi	s8,a2,1
   12268:	00190913          	addi	s2,s2,1
   1226c:	008a0b13          	addi	s6,s4,8
   12270:	32f75ae3          	bge	a4,a5,12da4 <_vfprintf_r+0x11e0>
   12274:	00100713          	li	a4,1
   12278:	00ea2223          	sw	a4,4(s4)
   1227c:	0d812623          	sw	s8,204(sp)
   12280:	0d212423          	sw	s2,200(sp)
   12284:	00700713          	li	a4,7
   12288:	01275463          	bge	a4,s2,12290 <_vfprintf_r+0x6cc>
   1228c:	0bc0106f          	j	13348 <_vfprintf_r+0x1784>
   12290:	02812783          	lw	a5,40(sp)
   12294:	02c12703          	lw	a4,44(sp)
   12298:	00190913          	addi	s2,s2,1
   1229c:	00fc0c33          	add	s8,s8,a5
   122a0:	00eb2023          	sw	a4,0(s6)
   122a4:	00fb2223          	sw	a5,4(s6)
   122a8:	0d812623          	sw	s8,204(sp)
   122ac:	0d212423          	sw	s2,200(sp)
   122b0:	00700713          	li	a4,7
   122b4:	008b0b13          	addi	s6,s6,8
   122b8:	01275463          	bge	a4,s2,122c0 <_vfprintf_r+0x6fc>
   122bc:	0b00106f          	j	1336c <_vfprintf_r+0x17a8>
   122c0:	0d012703          	lw	a4,208(sp)
   122c4:	01c12783          	lw	a5,28(sp)
   122c8:	08010593          	addi	a1,sp,128
   122cc:	08e12823          	sw	a4,144(sp)
   122d0:	0d412703          	lw	a4,212(sp)
   122d4:	09010513          	addi	a0,sp,144
   122d8:	08012023          	sw	zero,128(sp)
   122dc:	08e12a23          	sw	a4,148(sp)
   122e0:	0d812703          	lw	a4,216(sp)
   122e4:	08012223          	sw	zero,132(sp)
   122e8:	08012423          	sw	zero,136(sp)
   122ec:	08e12c23          	sw	a4,152(sp)
   122f0:	0dc12703          	lw	a4,220(sp)
   122f4:	08012623          	sw	zero,140(sp)
   122f8:	fff78a13          	addi	s4,a5,-1
   122fc:	08e12e23          	sw	a4,156(sp)
   12300:	7980b0ef          	jal	1da98 <__eqtf2>
   12304:	2e0500e3          	beqz	a0,12de4 <_vfprintf_r+0x1220>
   12308:	001d0793          	addi	a5,s10,1
   1230c:	00190913          	addi	s2,s2,1
   12310:	014c0c33          	add	s8,s8,s4
   12314:	00fb2023          	sw	a5,0(s6)
   12318:	014b2223          	sw	s4,4(s6)
   1231c:	0d812623          	sw	s8,204(sp)
   12320:	0d212423          	sw	s2,200(sp)
   12324:	00700793          	li	a5,7
   12328:	008b0b13          	addi	s6,s6,8
   1232c:	2927cae3          	blt	a5,s2,12dc0 <_vfprintf_r+0x11fc>
   12330:	03c12683          	lw	a3,60(sp)
   12334:	0b410713          	addi	a4,sp,180
   12338:	00190793          	addi	a5,s2,1
   1233c:	01868633          	add	a2,a3,s8
   12340:	00eb2023          	sw	a4,0(s6)
   12344:	00db2223          	sw	a3,4(s6)
   12348:	0cc12623          	sw	a2,204(sp)
   1234c:	0cf12423          	sw	a5,200(sp)
   12350:	00700713          	li	a4,7
   12354:	008b0a13          	addi	s4,s6,8
   12358:	b4f758e3          	bge	a4,a5,11ea8 <_vfprintf_r+0x2e4>
   1235c:	00812503          	lw	a0,8(sp)
   12360:	0c410613          	addi	a2,sp,196
   12364:	00048593          	mv	a1,s1
   12368:	348020ef          	jal	146b0 <__sprint_r>
   1236c:	6e051063          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   12370:	0cc12603          	lw	a2,204(sp)
   12374:	00098a13          	mv	s4,s3
   12378:	b31ff06f          	j	11ea8 <_vfprintf_r+0x2e4>
   1237c:	01412703          	lw	a4,20(sp)
   12380:	010cf793          	andi	a5,s9,16
   12384:	00072903          	lw	s2,0(a4)
   12388:	00470713          	addi	a4,a4,4
   1238c:	00e12a23          	sw	a4,20(sp)
   12390:	36079ae3          	bnez	a5,12f04 <_vfprintf_r+0x1340>
   12394:	040cf793          	andi	a5,s9,64
   12398:	360782e3          	beqz	a5,12efc <_vfprintf_r+0x1338>
   1239c:	01091913          	slli	s2,s2,0x10
   123a0:	01095913          	srli	s2,s2,0x10
   123a4:	00000d93          	li	s11,0
   123a8:	cf1ff06f          	j	12098 <_vfprintf_r+0x4d4>
   123ac:	01412703          	lw	a4,20(sp)
   123b0:	010cf793          	andi	a5,s9,16
   123b4:	00072903          	lw	s2,0(a4)
   123b8:	00470713          	addi	a4,a4,4
   123bc:	00e12a23          	sw	a4,20(sp)
   123c0:	320798e3          	bnez	a5,12ef0 <_vfprintf_r+0x132c>
   123c4:	040cf793          	andi	a5,s9,64
   123c8:	320780e3          	beqz	a5,12ee8 <_vfprintf_r+0x1324>
   123cc:	01091913          	slli	s2,s2,0x10
   123d0:	41095913          	srai	s2,s2,0x10
   123d4:	41f95d93          	srai	s11,s2,0x1f
   123d8:	000d8793          	mv	a5,s11
   123dc:	c407d2e3          	bgez	a5,12020 <_vfprintf_r+0x45c>
   123e0:	02d00713          	li	a4,45
   123e4:	012037b3          	snez	a5,s2
   123e8:	41b00db3          	neg	s11,s11
   123ec:	0ae103a3          	sb	a4,167(sp)
   123f0:	40fd8db3          	sub	s11,s11,a5
   123f4:	41200933          	neg	s2,s2
   123f8:	000c8e93          	mv	t4,s9
   123fc:	00100793          	li	a5,1
   12400:	ca0b56e3          	bgez	s6,120ac <_vfprintf_r+0x4e8>
   12404:	00100713          	li	a4,1
   12408:	10e78463          	beq	a5,a4,12510 <_vfprintf_r+0x94c>
   1240c:	00200713          	li	a4,2
   12410:	40e78263          	beq	a5,a4,12814 <_vfprintf_r+0xc50>
   12414:	19010d13          	addi	s10,sp,400
   12418:	01dd9793          	slli	a5,s11,0x1d
   1241c:	00797713          	andi	a4,s2,7
   12420:	00395913          	srli	s2,s2,0x3
   12424:	03070713          	addi	a4,a4,48
   12428:	0127e933          	or	s2,a5,s2
   1242c:	003ddd93          	srli	s11,s11,0x3
   12430:	feed0fa3          	sb	a4,-1(s10)
   12434:	01b967b3          	or	a5,s2,s11
   12438:	000d0613          	mv	a2,s10
   1243c:	fffd0d13          	addi	s10,s10,-1
   12440:	fc079ce3          	bnez	a5,12418 <_vfprintf_r+0x854>
   12444:	001ef693          	andi	a3,t4,1
   12448:	40068063          	beqz	a3,12848 <_vfprintf_r+0xc84>
   1244c:	03000693          	li	a3,48
   12450:	3ed70c63          	beq	a4,a3,12848 <_vfprintf_r+0xc84>
   12454:	ffe60613          	addi	a2,a2,-2
   12458:	19010793          	addi	a5,sp,400
   1245c:	fedd0fa3          	sb	a3,-1(s10)
   12460:	40c78933          	sub	s2,a5,a2
   12464:	000e8c93          	mv	s9,t4
   12468:	00060d13          	mv	s10,a2
   1246c:	bd9ff06f          	j	12044 <_vfprintf_r+0x480>
   12470:	01412703          	lw	a4,20(sp)
   12474:	0a0103a3          	sb	zero,167(sp)
   12478:	00100d93          	li	s11,1
   1247c:	00072783          	lw	a5,0(a4)
   12480:	00470713          	addi	a4,a4,4
   12484:	00e12a23          	sw	a4,20(sp)
   12488:	12f10623          	sb	a5,300(sp)
   1248c:	00100913          	li	s2,1
   12490:	12c10d13          	addi	s10,sp,300
   12494:	935ff06f          	j	11dc8 <_vfprintf_r+0x204>
   12498:	01412783          	lw	a5,20(sp)
   1249c:	0a0103a3          	sb	zero,167(sp)
   124a0:	0007ad03          	lw	s10,0(a5)
   124a4:	00478c13          	addi	s8,a5,4
   124a8:	3a0d02e3          	beqz	s10,1304c <_vfprintf_r+0x1488>
   124ac:	000b5463          	bgez	s6,124b4 <_vfprintf_r+0x8f0>
   124b0:	1bc0106f          	j	1366c <_vfprintf_r+0x1aa8>
   124b4:	000b0613          	mv	a2,s6
   124b8:	00000593          	li	a1,0
   124bc:	000d0513          	mv	a0,s10
   124c0:	01112a23          	sw	a7,20(sp)
   124c4:	274040ef          	jal	16738 <memchr>
   124c8:	00a12823          	sw	a0,16(sp)
   124cc:	01412883          	lw	a7,20(sp)
   124d0:	00051463          	bnez	a0,124d8 <_vfprintf_r+0x914>
   124d4:	3d90106f          	j	140ac <_vfprintf_r+0x24e8>
   124d8:	01012783          	lw	a5,16(sp)
   124dc:	0a714703          	lbu	a4,167(sp)
   124e0:	01812a23          	sw	s8,20(sp)
   124e4:	41a78933          	sub	s2,a5,s10
   124e8:	fff94693          	not	a3,s2
   124ec:	41f6d693          	srai	a3,a3,0x1f
   124f0:	00012823          	sw	zero,16(sp)
   124f4:	02012223          	sw	zero,36(sp)
   124f8:	02012023          	sw	zero,32(sp)
   124fc:	00012c23          	sw	zero,24(sp)
   12500:	00d97db3          	and	s11,s2,a3
   12504:	00000b13          	li	s6,0
   12508:	b60710e3          	bnez	a4,12068 <_vfprintf_r+0x4a4>
   1250c:	8d1ff06f          	j	11ddc <_vfprintf_r+0x218>
   12510:	680d98e3          	bnez	s11,133a0 <_vfprintf_r+0x17dc>
   12514:	00900793          	li	a5,9
   12518:	6927e4e3          	bltu	a5,s2,133a0 <_vfprintf_r+0x17dc>
   1251c:	03090913          	addi	s2,s2,48
   12520:	192107a3          	sb	s2,399(sp)
   12524:	000e8c93          	mv	s9,t4
   12528:	00100913          	li	s2,1
   1252c:	18f10d13          	addi	s10,sp,399
   12530:	b15ff06f          	j	12044 <_vfprintf_r+0x480>
   12534:	01412783          	lw	a5,20(sp)
   12538:	0007ab83          	lw	s7,0(a5)
   1253c:	00478793          	addi	a5,a5,4
   12540:	180bd2e3          	bgez	s7,12ec4 <_vfprintf_r+0x1300>
   12544:	41700bb3          	neg	s7,s7
   12548:	00f12a23          	sw	a5,20(sp)
   1254c:	000ac883          	lbu	a7,0(s5)
   12550:	004cec93          	ori	s9,s9,4
   12554:	801ff06f          	j	11d54 <_vfprintf_r+0x190>
   12558:	010cee93          	ori	t4,s9,16
   1255c:	020ef793          	andi	a5,t4,32
   12560:	10078ae3          	beqz	a5,12e74 <_vfprintf_r+0x12b0>
   12564:	01412783          	lw	a5,20(sp)
   12568:	00778c13          	addi	s8,a5,7
   1256c:	ff8c7c13          	andi	s8,s8,-8
   12570:	008c0793          	addi	a5,s8,8
   12574:	00f12a23          	sw	a5,20(sp)
   12578:	000c2903          	lw	s2,0(s8)
   1257c:	004c2d83          	lw	s11,4(s8)
   12580:	00100793          	li	a5,1
   12584:	b1dff06f          	j	120a0 <_vfprintf_r+0x4dc>
   12588:	02b00793          	li	a5,43
   1258c:	000ac883          	lbu	a7,0(s5)
   12590:	0af103a3          	sb	a5,167(sp)
   12594:	fc0ff06f          	j	11d54 <_vfprintf_r+0x190>
   12598:	01412703          	lw	a4,20(sp)
   1259c:	000087b7          	lui	a5,0x8
   125a0:	83078793          	addi	a5,a5,-2000 # 7830 <exit-0x8884>
   125a4:	0af11423          	sh	a5,168(sp)
   125a8:	00470793          	addi	a5,a4,4
   125ac:	00f12a23          	sw	a5,20(sp)
   125b0:	0000e797          	auipc	a5,0xe
   125b4:	37c78793          	addi	a5,a5,892 # 2092c <_exit+0xf4>
   125b8:	02f12823          	sw	a5,48(sp)
   125bc:	00072903          	lw	s2,0(a4)
   125c0:	00000d93          	li	s11,0
   125c4:	002cee93          	ori	t4,s9,2
   125c8:	00200793          	li	a5,2
   125cc:	07800893          	li	a7,120
   125d0:	ad1ff06f          	j	120a0 <_vfprintf_r+0x4dc>
   125d4:	020cf793          	andi	a5,s9,32
   125d8:	16078ee3          	beqz	a5,12f54 <_vfprintf_r+0x1390>
   125dc:	01412783          	lw	a5,20(sp)
   125e0:	00c12683          	lw	a3,12(sp)
   125e4:	0007a783          	lw	a5,0(a5)
   125e8:	41f6d713          	srai	a4,a3,0x1f
   125ec:	00d7a023          	sw	a3,0(a5)
   125f0:	00e7a223          	sw	a4,4(a5)
   125f4:	01412783          	lw	a5,20(sp)
   125f8:	000a8d13          	mv	s10,s5
   125fc:	00478793          	addi	a5,a5,4
   12600:	00f12a23          	sw	a5,20(sp)
   12604:	8f1ff06f          	j	11ef4 <_vfprintf_r+0x330>
   12608:	000ac883          	lbu	a7,0(s5)
   1260c:	06c00793          	li	a5,108
   12610:	22f886e3          	beq	a7,a5,1303c <_vfprintf_r+0x1478>
   12614:	010cec93          	ori	s9,s9,16
   12618:	f3cff06f          	j	11d54 <_vfprintf_r+0x190>
   1261c:	000ac883          	lbu	a7,0(s5)
   12620:	06800793          	li	a5,104
   12624:	20f884e3          	beq	a7,a5,1302c <_vfprintf_r+0x1468>
   12628:	040cec93          	ori	s9,s9,64
   1262c:	f28ff06f          	j	11d54 <_vfprintf_r+0x190>
   12630:	000ac883          	lbu	a7,0(s5)
   12634:	008cec93          	ori	s9,s9,8
   12638:	f1cff06f          	j	11d54 <_vfprintf_r+0x190>
   1263c:	000ac883          	lbu	a7,0(s5)
   12640:	001cec93          	ori	s9,s9,1
   12644:	f10ff06f          	j	11d54 <_vfprintf_r+0x190>
   12648:	0a714783          	lbu	a5,167(sp)
   1264c:	000ac883          	lbu	a7,0(s5)
   12650:	f0079263          	bnez	a5,11d54 <_vfprintf_r+0x190>
   12654:	02000793          	li	a5,32
   12658:	0af103a3          	sb	a5,167(sp)
   1265c:	ef8ff06f          	j	11d54 <_vfprintf_r+0x190>
   12660:	000ac883          	lbu	a7,0(s5)
   12664:	080cec93          	ori	s9,s9,128
   12668:	eecff06f          	j	11d54 <_vfprintf_r+0x190>
   1266c:	000ac883          	lbu	a7,0(s5)
   12670:	02a00793          	li	a5,42
   12674:	001a8693          	addi	a3,s5,1
   12678:	00f89463          	bne	a7,a5,12680 <_vfprintf_r+0xabc>
   1267c:	5e50106f          	j	14460 <_vfprintf_r+0x289c>
   12680:	fd088793          	addi	a5,a7,-48
   12684:	00900713          	li	a4,9
   12688:	00000b13          	li	s6,0
   1268c:	00900613          	li	a2,9
   12690:	02f76463          	bltu	a4,a5,126b8 <_vfprintf_r+0xaf4>
   12694:	0006c883          	lbu	a7,0(a3)
   12698:	002b1713          	slli	a4,s6,0x2
   1269c:	01670b33          	add	s6,a4,s6
   126a0:	001b1b13          	slli	s6,s6,0x1
   126a4:	00fb0b33          	add	s6,s6,a5
   126a8:	fd088793          	addi	a5,a7,-48
   126ac:	00168693          	addi	a3,a3,1
   126b0:	fef672e3          	bgeu	a2,a5,12694 <_vfprintf_r+0xad0>
   126b4:	0c0b4ce3          	bltz	s6,12f8c <_vfprintf_r+0x13c8>
   126b8:	00068a93          	mv	s5,a3
   126bc:	e9cff06f          	j	11d58 <_vfprintf_r+0x194>
   126c0:	06500713          	li	a4,101
   126c4:	b91758e3          	bge	a4,a7,12254 <_vfprintf_r+0x690>
   126c8:	0d012703          	lw	a4,208(sp)
   126cc:	08010593          	addi	a1,sp,128
   126d0:	09010513          	addi	a0,sp,144
   126d4:	08e12823          	sw	a4,144(sp)
   126d8:	0d412703          	lw	a4,212(sp)
   126dc:	02c12a23          	sw	a2,52(sp)
   126e0:	08012023          	sw	zero,128(sp)
   126e4:	08e12a23          	sw	a4,148(sp)
   126e8:	0d812703          	lw	a4,216(sp)
   126ec:	08012223          	sw	zero,132(sp)
   126f0:	08012423          	sw	zero,136(sp)
   126f4:	08e12c23          	sw	a4,152(sp)
   126f8:	0dc12703          	lw	a4,220(sp)
   126fc:	08012623          	sw	zero,140(sp)
   12700:	08e12e23          	sw	a4,156(sp)
   12704:	3940b0ef          	jal	1da98 <__eqtf2>
   12708:	03412603          	lw	a2,52(sp)
   1270c:	54051663          	bnez	a0,12c58 <_vfprintf_r+0x1094>
   12710:	0c812783          	lw	a5,200(sp)
   12714:	0000e717          	auipc	a4,0xe
   12718:	24870713          	addi	a4,a4,584 # 2095c <_exit+0x124>
   1271c:	00ea2023          	sw	a4,0(s4)
   12720:	00160613          	addi	a2,a2,1
   12724:	00100713          	li	a4,1
   12728:	00178793          	addi	a5,a5,1
   1272c:	00ea2223          	sw	a4,4(s4)
   12730:	0cc12623          	sw	a2,204(sp)
   12734:	0cf12423          	sw	a5,200(sp)
   12738:	00700713          	li	a4,7
   1273c:	008a0a13          	addi	s4,s4,8
   12740:	5ef74ce3          	blt	a4,a5,13538 <_vfprintf_r+0x1974>
   12744:	0ac12783          	lw	a5,172(sp)
   12748:	01c12703          	lw	a4,28(sp)
   1274c:	76e7d463          	bge	a5,a4,12eb4 <_vfprintf_r+0x12f0>
   12750:	02c12783          	lw	a5,44(sp)
   12754:	02812703          	lw	a4,40(sp)
   12758:	008a0a13          	addi	s4,s4,8
   1275c:	fefa2c23          	sw	a5,-8(s4)
   12760:	0c812783          	lw	a5,200(sp)
   12764:	00e60633          	add	a2,a2,a4
   12768:	feea2e23          	sw	a4,-4(s4)
   1276c:	00178793          	addi	a5,a5,1
   12770:	0cc12623          	sw	a2,204(sp)
   12774:	0cf12423          	sw	a5,200(sp)
   12778:	00700713          	li	a4,7
   1277c:	08f748e3          	blt	a4,a5,1300c <_vfprintf_r+0x1448>
   12780:	01c12783          	lw	a5,28(sp)
   12784:	fff78913          	addi	s2,a5,-1
   12788:	f3205063          	blez	s2,11ea8 <_vfprintf_r+0x2e4>
   1278c:	0000e817          	auipc	a6,0xe
   12790:	4e480813          	addi	a6,a6,1252 # 20c70 <zeroes.0>
   12794:	01000713          	li	a4,16
   12798:	0c812783          	lw	a5,200(sp)
   1279c:	01000b13          	li	s6,16
   127a0:	00700c13          	li	s8,7
   127a4:	00080d13          	mv	s10,a6
   127a8:	01274863          	blt	a4,s2,127b8 <_vfprintf_r+0xbf4>
   127ac:	5b10006f          	j	1355c <_vfprintf_r+0x1998>
   127b0:	ff090913          	addi	s2,s2,-16
   127b4:	5b2b52e3          	bge	s6,s2,13558 <_vfprintf_r+0x1994>
   127b8:	01060613          	addi	a2,a2,16
   127bc:	00178793          	addi	a5,a5,1
   127c0:	01aa2023          	sw	s10,0(s4)
   127c4:	016a2223          	sw	s6,4(s4)
   127c8:	0cc12623          	sw	a2,204(sp)
   127cc:	0cf12423          	sw	a5,200(sp)
   127d0:	008a0a13          	addi	s4,s4,8
   127d4:	fcfc5ee3          	bge	s8,a5,127b0 <_vfprintf_r+0xbec>
   127d8:	00812503          	lw	a0,8(sp)
   127dc:	0c410613          	addi	a2,sp,196
   127e0:	00048593          	mv	a1,s1
   127e4:	6cd010ef          	jal	146b0 <__sprint_r>
   127e8:	26051263          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   127ec:	0cc12603          	lw	a2,204(sp)
   127f0:	0c812783          	lw	a5,200(sp)
   127f4:	00098a13          	mv	s4,s3
   127f8:	fb9ff06f          	j	127b0 <_vfprintf_r+0xbec>
   127fc:	00100713          	li	a4,1
   12800:	00e79463          	bne	a5,a4,12808 <_vfprintf_r+0xc44>
   12804:	1510106f          	j	14154 <_vfprintf_r+0x2590>
   12808:	00200713          	li	a4,2
   1280c:	000c8e93          	mv	t4,s9
   12810:	c0e792e3          	bne	a5,a4,12414 <_vfprintf_r+0x850>
   12814:	03012683          	lw	a3,48(sp)
   12818:	19010d13          	addi	s10,sp,400
   1281c:	00f97793          	andi	a5,s2,15
   12820:	00f687b3          	add	a5,a3,a5
   12824:	0007c703          	lbu	a4,0(a5)
   12828:	00495913          	srli	s2,s2,0x4
   1282c:	01cd9793          	slli	a5,s11,0x1c
   12830:	0127e933          	or	s2,a5,s2
   12834:	004ddd93          	srli	s11,s11,0x4
   12838:	feed0fa3          	sb	a4,-1(s10)
   1283c:	01b967b3          	or	a5,s2,s11
   12840:	fffd0d13          	addi	s10,s10,-1
   12844:	fc079ce3          	bnez	a5,1281c <_vfprintf_r+0xc58>
   12848:	19010793          	addi	a5,sp,400
   1284c:	41a78933          	sub	s2,a5,s10
   12850:	000e8c93          	mv	s9,t4
   12854:	ff0ff06f          	j	12044 <_vfprintf_r+0x480>
   12858:	41bb8c33          	sub	s8,s7,s11
   1285c:	e1805a63          	blez	s8,11e70 <_vfprintf_r+0x2ac>
   12860:	01000513          	li	a0,16
   12864:	0c812583          	lw	a1,200(sp)
   12868:	0000e817          	auipc	a6,0xe
   1286c:	40880813          	addi	a6,a6,1032 # 20c70 <zeroes.0>
   12870:	09855c63          	bge	a0,s8,12908 <_vfprintf_r+0xd44>
   12874:	00090713          	mv	a4,s2
   12878:	000a0793          	mv	a5,s4
   1287c:	000c0913          	mv	s2,s8
   12880:	01000e93          	li	t4,16
   12884:	00700f93          	li	t6,7
   12888:	03112a23          	sw	a7,52(sp)
   1288c:	00080a13          	mv	s4,a6
   12890:	00070c13          	mv	s8,a4
   12894:	00c0006f          	j	128a0 <_vfprintf_r+0xcdc>
   12898:	ff090913          	addi	s2,s2,-16
   1289c:	052eda63          	bge	t4,s2,128f0 <_vfprintf_r+0xd2c>
   128a0:	01060613          	addi	a2,a2,16
   128a4:	00158593          	addi	a1,a1,1 # 2001 <exit-0xe0b3>
   128a8:	0147a023          	sw	s4,0(a5)
   128ac:	01d7a223          	sw	t4,4(a5)
   128b0:	0cc12623          	sw	a2,204(sp)
   128b4:	0cb12423          	sw	a1,200(sp)
   128b8:	00878793          	addi	a5,a5,8
   128bc:	fcbfdee3          	bge	t6,a1,12898 <_vfprintf_r+0xcd4>
   128c0:	00812503          	lw	a0,8(sp)
   128c4:	0c410613          	addi	a2,sp,196
   128c8:	00048593          	mv	a1,s1
   128cc:	5e5010ef          	jal	146b0 <__sprint_r>
   128d0:	16051e63          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   128d4:	01000e93          	li	t4,16
   128d8:	ff090913          	addi	s2,s2,-16
   128dc:	0cc12603          	lw	a2,204(sp)
   128e0:	0c812583          	lw	a1,200(sp)
   128e4:	00098793          	mv	a5,s3
   128e8:	00700f93          	li	t6,7
   128ec:	fb2ecae3          	blt	t4,s2,128a0 <_vfprintf_r+0xcdc>
   128f0:	03412883          	lw	a7,52(sp)
   128f4:	000c0713          	mv	a4,s8
   128f8:	000a0813          	mv	a6,s4
   128fc:	00090c13          	mv	s8,s2
   12900:	00078a13          	mv	s4,a5
   12904:	00070913          	mv	s2,a4
   12908:	01860633          	add	a2,a2,s8
   1290c:	00158593          	addi	a1,a1,1
   12910:	010a2023          	sw	a6,0(s4)
   12914:	018a2223          	sw	s8,4(s4)
   12918:	0cc12623          	sw	a2,204(sp)
   1291c:	0cb12423          	sw	a1,200(sp)
   12920:	00700713          	li	a4,7
   12924:	008a0a13          	addi	s4,s4,8
   12928:	d4b75463          	bge	a4,a1,11e70 <_vfprintf_r+0x2ac>
   1292c:	00812503          	lw	a0,8(sp)
   12930:	0c410613          	addi	a2,sp,196
   12934:	00048593          	mv	a1,s1
   12938:	03112a23          	sw	a7,52(sp)
   1293c:	575010ef          	jal	146b0 <__sprint_r>
   12940:	10051663          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   12944:	0cc12603          	lw	a2,204(sp)
   12948:	03412883          	lw	a7,52(sp)
   1294c:	00098a13          	mv	s4,s3
   12950:	d20ff06f          	j	11e70 <_vfprintf_r+0x2ac>
   12954:	0c812583          	lw	a1,200(sp)
   12958:	0000e817          	auipc	a6,0xe
   1295c:	31880813          	addi	a6,a6,792 # 20c70 <zeroes.0>
   12960:	09645663          	bge	s0,s6,129ec <_vfprintf_r+0xe28>
   12964:	000a0793          	mv	a5,s4
   12968:	00700c13          	li	s8,7
   1296c:	00090a13          	mv	s4,s2
   12970:	03112a23          	sw	a7,52(sp)
   12974:	000b0913          	mv	s2,s6
   12978:	00080b13          	mv	s6,a6
   1297c:	00c0006f          	j	12988 <_vfprintf_r+0xdc4>
   12980:	ff090913          	addi	s2,s2,-16
   12984:	05245a63          	bge	s0,s2,129d8 <_vfprintf_r+0xe14>
   12988:	01060613          	addi	a2,a2,16
   1298c:	00158593          	addi	a1,a1,1
   12990:	0000e717          	auipc	a4,0xe
   12994:	2e070713          	addi	a4,a4,736 # 20c70 <zeroes.0>
   12998:	00e7a023          	sw	a4,0(a5)
   1299c:	0087a223          	sw	s0,4(a5)
   129a0:	0cc12623          	sw	a2,204(sp)
   129a4:	0cb12423          	sw	a1,200(sp)
   129a8:	00878793          	addi	a5,a5,8
   129ac:	fcbc5ae3          	bge	s8,a1,12980 <_vfprintf_r+0xdbc>
   129b0:	00812503          	lw	a0,8(sp)
   129b4:	0c410613          	addi	a2,sp,196
   129b8:	00048593          	mv	a1,s1
   129bc:	4f5010ef          	jal	146b0 <__sprint_r>
   129c0:	08051663          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   129c4:	ff090913          	addi	s2,s2,-16
   129c8:	0cc12603          	lw	a2,204(sp)
   129cc:	0c812583          	lw	a1,200(sp)
   129d0:	00098793          	mv	a5,s3
   129d4:	fb244ae3          	blt	s0,s2,12988 <_vfprintf_r+0xdc4>
   129d8:	03412883          	lw	a7,52(sp)
   129dc:	000b0813          	mv	a6,s6
   129e0:	00090b13          	mv	s6,s2
   129e4:	000a0913          	mv	s2,s4
   129e8:	00078a13          	mv	s4,a5
   129ec:	01660633          	add	a2,a2,s6
   129f0:	00158593          	addi	a1,a1,1
   129f4:	010a2023          	sw	a6,0(s4)
   129f8:	016a2223          	sw	s6,4(s4)
   129fc:	0cc12623          	sw	a2,204(sp)
   12a00:	0cb12423          	sw	a1,200(sp)
   12a04:	00700713          	li	a4,7
   12a08:	008a0a13          	addi	s4,s4,8
   12a0c:	c6b75663          	bge	a4,a1,11e78 <_vfprintf_r+0x2b4>
   12a10:	00812503          	lw	a0,8(sp)
   12a14:	0c410613          	addi	a2,sp,196
   12a18:	00048593          	mv	a1,s1
   12a1c:	03112a23          	sw	a7,52(sp)
   12a20:	491010ef          	jal	146b0 <__sprint_r>
   12a24:	02051463          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   12a28:	0cc12603          	lw	a2,204(sp)
   12a2c:	03412883          	lw	a7,52(sp)
   12a30:	00098a13          	mv	s4,s3
   12a34:	c44ff06f          	j	11e78 <_vfprintf_r+0x2b4>
   12a38:	00812503          	lw	a0,8(sp)
   12a3c:	0c410613          	addi	a2,sp,196
   12a40:	00048593          	mv	a1,s1
   12a44:	46d010ef          	jal	146b0 <__sprint_r>
   12a48:	c8050663          	beqz	a0,11ed4 <_vfprintf_r+0x310>
   12a4c:	01012383          	lw	t2,16(sp)
   12a50:	ca038c63          	beqz	t2,11f08 <_vfprintf_r+0x344>
   12a54:	00812503          	lw	a0,8(sp)
   12a58:	00038593          	mv	a1,t2
   12a5c:	e94fe0ef          	jal	110f0 <_free_r>
   12a60:	ca8ff06f          	j	11f08 <_vfprintf_r+0x344>
   12a64:	000c8e93          	mv	t4,s9
   12a68:	99dff06f          	j	12404 <_vfprintf_r+0x840>
   12a6c:	01000513          	li	a0,16
   12a70:	0c812583          	lw	a1,200(sp)
   12a74:	0000ec17          	auipc	s8,0xe
   12a78:	20cc0c13          	addi	s8,s8,524 # 20c80 <blanks.1>
   12a7c:	0ae55063          	bge	a0,a4,12b1c <_vfprintf_r+0xf58>
   12a80:	000a0793          	mv	a5,s4
   12a84:	01000813          	li	a6,16
   12a88:	000c0a13          	mv	s4,s8
   12a8c:	00700393          	li	t2,7
   12a90:	00090c13          	mv	s8,s2
   12a94:	02512a23          	sw	t0,52(sp)
   12a98:	05f12423          	sw	t6,72(sp)
   12a9c:	05112623          	sw	a7,76(sp)
   12aa0:	00070913          	mv	s2,a4
   12aa4:	00c0006f          	j	12ab0 <_vfprintf_r+0xeec>
   12aa8:	ff090913          	addi	s2,s2,-16
   12aac:	05285a63          	bge	a6,s2,12b00 <_vfprintf_r+0xf3c>
   12ab0:	01060613          	addi	a2,a2,16
   12ab4:	00158593          	addi	a1,a1,1
   12ab8:	0147a023          	sw	s4,0(a5)
   12abc:	0107a223          	sw	a6,4(a5)
   12ac0:	0cc12623          	sw	a2,204(sp)
   12ac4:	0cb12423          	sw	a1,200(sp)
   12ac8:	00878793          	addi	a5,a5,8
   12acc:	fcb3dee3          	bge	t2,a1,12aa8 <_vfprintf_r+0xee4>
   12ad0:	00812503          	lw	a0,8(sp)
   12ad4:	0c410613          	addi	a2,sp,196
   12ad8:	00048593          	mv	a1,s1
   12adc:	3d5010ef          	jal	146b0 <__sprint_r>
   12ae0:	f60516e3          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   12ae4:	01000813          	li	a6,16
   12ae8:	ff090913          	addi	s2,s2,-16
   12aec:	0cc12603          	lw	a2,204(sp)
   12af0:	0c812583          	lw	a1,200(sp)
   12af4:	00098793          	mv	a5,s3
   12af8:	00700393          	li	t2,7
   12afc:	fb284ae3          	blt	a6,s2,12ab0 <_vfprintf_r+0xeec>
   12b00:	03412283          	lw	t0,52(sp)
   12b04:	04812f83          	lw	t6,72(sp)
   12b08:	04c12883          	lw	a7,76(sp)
   12b0c:	00090713          	mv	a4,s2
   12b10:	000c0913          	mv	s2,s8
   12b14:	000a0c13          	mv	s8,s4
   12b18:	00078a13          	mv	s4,a5
   12b1c:	00e60633          	add	a2,a2,a4
   12b20:	00158593          	addi	a1,a1,1
   12b24:	00ea2223          	sw	a4,4(s4)
   12b28:	018a2023          	sw	s8,0(s4)
   12b2c:	0cc12623          	sw	a2,204(sp)
   12b30:	0cb12423          	sw	a1,200(sp)
   12b34:	00700713          	li	a4,7
   12b38:	008a0a13          	addi	s4,s4,8
   12b3c:	acb75063          	bge	a4,a1,11dfc <_vfprintf_r+0x238>
   12b40:	00812503          	lw	a0,8(sp)
   12b44:	0c410613          	addi	a2,sp,196
   12b48:	00048593          	mv	a1,s1
   12b4c:	05112623          	sw	a7,76(sp)
   12b50:	05f12423          	sw	t6,72(sp)
   12b54:	02512a23          	sw	t0,52(sp)
   12b58:	359010ef          	jal	146b0 <__sprint_r>
   12b5c:	ee0518e3          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   12b60:	0cc12603          	lw	a2,204(sp)
   12b64:	04c12883          	lw	a7,76(sp)
   12b68:	04812f83          	lw	t6,72(sp)
   12b6c:	03412283          	lw	t0,52(sp)
   12b70:	00098a13          	mv	s4,s3
   12b74:	a88ff06f          	j	11dfc <_vfprintf_r+0x238>
   12b78:	00812503          	lw	a0,8(sp)
   12b7c:	0c410613          	addi	a2,sp,196
   12b80:	00048593          	mv	a1,s1
   12b84:	05112423          	sw	a7,72(sp)
   12b88:	03f12a23          	sw	t6,52(sp)
   12b8c:	325010ef          	jal	146b0 <__sprint_r>
   12b90:	ea051ee3          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   12b94:	0cc12603          	lw	a2,204(sp)
   12b98:	04812883          	lw	a7,72(sp)
   12b9c:	03412f83          	lw	t6,52(sp)
   12ba0:	00098a13          	mv	s4,s3
   12ba4:	ac4ff06f          	j	11e68 <_vfprintf_r+0x2a4>
   12ba8:	01000713          	li	a4,16
   12bac:	0c812783          	lw	a5,200(sp)
   12bb0:	0000ec17          	auipc	s8,0xe
   12bb4:	0d0c0c13          	addi	s8,s8,208 # 20c80 <blanks.1>
   12bb8:	07275263          	bge	a4,s2,12c1c <_vfprintf_r+0x1058>
   12bbc:	00812d03          	lw	s10,8(sp)
   12bc0:	01000b13          	li	s6,16
   12bc4:	00700c93          	li	s9,7
   12bc8:	00c0006f          	j	12bd4 <_vfprintf_r+0x1010>
   12bcc:	ff090913          	addi	s2,s2,-16
   12bd0:	052b5663          	bge	s6,s2,12c1c <_vfprintf_r+0x1058>
   12bd4:	01060613          	addi	a2,a2,16
   12bd8:	00178793          	addi	a5,a5,1
   12bdc:	018a2023          	sw	s8,0(s4)
   12be0:	016a2223          	sw	s6,4(s4)
   12be4:	0cc12623          	sw	a2,204(sp)
   12be8:	0cf12423          	sw	a5,200(sp)
   12bec:	008a0a13          	addi	s4,s4,8
   12bf0:	fcfcdee3          	bge	s9,a5,12bcc <_vfprintf_r+0x1008>
   12bf4:	0c410613          	addi	a2,sp,196
   12bf8:	00048593          	mv	a1,s1
   12bfc:	000d0513          	mv	a0,s10
   12c00:	2b1010ef          	jal	146b0 <__sprint_r>
   12c04:	e40514e3          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   12c08:	ff090913          	addi	s2,s2,-16
   12c0c:	0cc12603          	lw	a2,204(sp)
   12c10:	0c812783          	lw	a5,200(sp)
   12c14:	00098a13          	mv	s4,s3
   12c18:	fb2b4ee3          	blt	s6,s2,12bd4 <_vfprintf_r+0x1010>
   12c1c:	01260633          	add	a2,a2,s2
   12c20:	00178793          	addi	a5,a5,1
   12c24:	018a2023          	sw	s8,0(s4)
   12c28:	012a2223          	sw	s2,4(s4)
   12c2c:	0cc12623          	sw	a2,204(sp)
   12c30:	0cf12423          	sw	a5,200(sp)
   12c34:	00700713          	li	a4,7
   12c38:	a8f75063          	bge	a4,a5,11eb8 <_vfprintf_r+0x2f4>
   12c3c:	00812503          	lw	a0,8(sp)
   12c40:	0c410613          	addi	a2,sp,196
   12c44:	00048593          	mv	a1,s1
   12c48:	269010ef          	jal	146b0 <__sprint_r>
   12c4c:	e00510e3          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   12c50:	0cc12603          	lw	a2,204(sp)
   12c54:	a64ff06f          	j	11eb8 <_vfprintf_r+0x2f4>
   12c58:	0ac12583          	lw	a1,172(sp)
   12c5c:	04b058e3          	blez	a1,134ac <_vfprintf_r+0x18e8>
   12c60:	01812783          	lw	a5,24(sp)
   12c64:	01c12703          	lw	a4,28(sp)
   12c68:	00078b13          	mv	s6,a5
   12c6c:	30f74a63          	blt	a4,a5,12f80 <_vfprintf_r+0x13bc>
   12c70:	03605663          	blez	s6,12c9c <_vfprintf_r+0x10d8>
   12c74:	0c812703          	lw	a4,200(sp)
   12c78:	01660633          	add	a2,a2,s6
   12c7c:	01aa2023          	sw	s10,0(s4)
   12c80:	00170713          	addi	a4,a4,1
   12c84:	016a2223          	sw	s6,4(s4)
   12c88:	0cc12623          	sw	a2,204(sp)
   12c8c:	0ce12423          	sw	a4,200(sp)
   12c90:	00700593          	li	a1,7
   12c94:	008a0a13          	addi	s4,s4,8
   12c98:	5ae5c0e3          	blt	a1,a4,13a38 <_vfprintf_r+0x1e74>
   12c9c:	fffb4713          	not	a4,s6
   12ca0:	01812783          	lw	a5,24(sp)
   12ca4:	41f75713          	srai	a4,a4,0x1f
   12ca8:	00eb7b33          	and	s6,s6,a4
   12cac:	41678b33          	sub	s6,a5,s6
   12cb0:	3b604e63          	bgtz	s6,1306c <_vfprintf_r+0x14a8>
   12cb4:	01812783          	lw	a5,24(sp)
   12cb8:	400cf713          	andi	a4,s9,1024
   12cbc:	00fd0c33          	add	s8,s10,a5
   12cc0:	280716e3          	bnez	a4,1374c <_vfprintf_r+0x1b88>
   12cc4:	0ac12583          	lw	a1,172(sp)
   12cc8:	01c12783          	lw	a5,28(sp)
   12ccc:	40f5ca63          	blt	a1,a5,130e0 <_vfprintf_r+0x151c>
   12cd0:	001cf713          	andi	a4,s9,1
   12cd4:	40071663          	bnez	a4,130e0 <_vfprintf_r+0x151c>
   12cd8:	01c12703          	lw	a4,28(sp)
   12cdc:	00ed07b3          	add	a5,s10,a4
   12ce0:	40b705b3          	sub	a1,a4,a1
   12ce4:	41878b33          	sub	s6,a5,s8
   12ce8:	0165d463          	bge	a1,s6,12cf0 <_vfprintf_r+0x112c>
   12cec:	00058b13          	mv	s6,a1
   12cf0:	03605863          	blez	s6,12d20 <_vfprintf_r+0x115c>
   12cf4:	0c812703          	lw	a4,200(sp)
   12cf8:	01660633          	add	a2,a2,s6
   12cfc:	018a2023          	sw	s8,0(s4)
   12d00:	00170713          	addi	a4,a4,1
   12d04:	016a2223          	sw	s6,4(s4)
   12d08:	0cc12623          	sw	a2,204(sp)
   12d0c:	0ce12423          	sw	a4,200(sp)
   12d10:	00700793          	li	a5,7
   12d14:	008a0a13          	addi	s4,s4,8
   12d18:	00e7d463          	bge	a5,a4,12d20 <_vfprintf_r+0x115c>
   12d1c:	3cc0106f          	j	140e8 <_vfprintf_r+0x2524>
   12d20:	fffb4713          	not	a4,s6
   12d24:	41f75713          	srai	a4,a4,0x1f
   12d28:	00eb77b3          	and	a5,s6,a4
   12d2c:	40f58933          	sub	s2,a1,a5
   12d30:	97205c63          	blez	s2,11ea8 <_vfprintf_r+0x2e4>
   12d34:	0000e817          	auipc	a6,0xe
   12d38:	f3c80813          	addi	a6,a6,-196 # 20c70 <zeroes.0>
   12d3c:	01000713          	li	a4,16
   12d40:	0c812783          	lw	a5,200(sp)
   12d44:	01000b13          	li	s6,16
   12d48:	00700c13          	li	s8,7
   12d4c:	00080d13          	mv	s10,a6
   12d50:	01274863          	blt	a4,s2,12d60 <_vfprintf_r+0x119c>
   12d54:	0090006f          	j	1355c <_vfprintf_r+0x1998>
   12d58:	ff090913          	addi	s2,s2,-16
   12d5c:	7f2b5e63          	bge	s6,s2,13558 <_vfprintf_r+0x1994>
   12d60:	01060613          	addi	a2,a2,16
   12d64:	00178793          	addi	a5,a5,1
   12d68:	01aa2023          	sw	s10,0(s4)
   12d6c:	016a2223          	sw	s6,4(s4)
   12d70:	0cc12623          	sw	a2,204(sp)
   12d74:	0cf12423          	sw	a5,200(sp)
   12d78:	008a0a13          	addi	s4,s4,8
   12d7c:	fcfc5ee3          	bge	s8,a5,12d58 <_vfprintf_r+0x1194>
   12d80:	00812503          	lw	a0,8(sp)
   12d84:	0c410613          	addi	a2,sp,196
   12d88:	00048593          	mv	a1,s1
   12d8c:	125010ef          	jal	146b0 <__sprint_r>
   12d90:	ca051ee3          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   12d94:	0cc12603          	lw	a2,204(sp)
   12d98:	0c812783          	lw	a5,200(sp)
   12d9c:	00098a13          	mv	s4,s3
   12da0:	fb9ff06f          	j	12d58 <_vfprintf_r+0x1194>
   12da4:	001cf593          	andi	a1,s9,1
   12da8:	cc059663          	bnez	a1,12274 <_vfprintf_r+0x6b0>
   12dac:	00ea2223          	sw	a4,4(s4)
   12db0:	0d812623          	sw	s8,204(sp)
   12db4:	0d212423          	sw	s2,200(sp)
   12db8:	00700793          	li	a5,7
   12dbc:	d727da63          	bge	a5,s2,12330 <_vfprintf_r+0x76c>
   12dc0:	00812503          	lw	a0,8(sp)
   12dc4:	0c410613          	addi	a2,sp,196
   12dc8:	00048593          	mv	a1,s1
   12dcc:	0e5010ef          	jal	146b0 <__sprint_r>
   12dd0:	c6051ee3          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   12dd4:	0cc12c03          	lw	s8,204(sp)
   12dd8:	0c812903          	lw	s2,200(sp)
   12ddc:	00098b13          	mv	s6,s3
   12de0:	d50ff06f          	j	12330 <_vfprintf_r+0x76c>
   12de4:	01c12703          	lw	a4,28(sp)
   12de8:	00100793          	li	a5,1
   12dec:	d4e7d263          	bge	a5,a4,12330 <_vfprintf_r+0x76c>
   12df0:	01100793          	li	a5,17
   12df4:	0000e817          	auipc	a6,0xe
   12df8:	e7c80813          	addi	a6,a6,-388 # 20c70 <zeroes.0>
   12dfc:	0ce7d6e3          	bge	a5,a4,136c8 <_vfprintf_r+0x1b04>
   12e00:	00048713          	mv	a4,s1
   12e04:	01512c23          	sw	s5,24(sp)
   12e08:	000a0493          	mv	s1,s4
   12e0c:	01000793          	li	a5,16
   12e10:	00700d13          	li	s10,7
   12e14:	00080a93          	mv	s5,a6
   12e18:	00070a13          	mv	s4,a4
   12e1c:	00c0006f          	j	12e28 <_vfprintf_r+0x1264>
   12e20:	ff048493          	addi	s1,s1,-16
   12e24:	0897d8e3          	bge	a5,s1,136b4 <_vfprintf_r+0x1af0>
   12e28:	010c0c13          	addi	s8,s8,16
   12e2c:	00190913          	addi	s2,s2,1
   12e30:	015b2023          	sw	s5,0(s6)
   12e34:	00fb2223          	sw	a5,4(s6)
   12e38:	0d812623          	sw	s8,204(sp)
   12e3c:	0d212423          	sw	s2,200(sp)
   12e40:	008b0b13          	addi	s6,s6,8
   12e44:	fd2d5ee3          	bge	s10,s2,12e20 <_vfprintf_r+0x125c>
   12e48:	00812503          	lw	a0,8(sp)
   12e4c:	0c410613          	addi	a2,sp,196
   12e50:	000a0593          	mv	a1,s4
   12e54:	05d010ef          	jal	146b0 <__sprint_r>
   12e58:	00050463          	beqz	a0,12e60 <_vfprintf_r+0x129c>
   12e5c:	2ec0106f          	j	14148 <_vfprintf_r+0x2584>
   12e60:	0cc12c03          	lw	s8,204(sp)
   12e64:	0c812903          	lw	s2,200(sp)
   12e68:	00098b13          	mv	s6,s3
   12e6c:	01000793          	li	a5,16
   12e70:	fb1ff06f          	j	12e20 <_vfprintf_r+0x125c>
   12e74:	01412703          	lw	a4,20(sp)
   12e78:	010ef793          	andi	a5,t4,16
   12e7c:	00072903          	lw	s2,0(a4)
   12e80:	00470713          	addi	a4,a4,4
   12e84:	00e12a23          	sw	a4,20(sp)
   12e88:	04079a63          	bnez	a5,12edc <_vfprintf_r+0x1318>
   12e8c:	040ef793          	andi	a5,t4,64
   12e90:	04078063          	beqz	a5,12ed0 <_vfprintf_r+0x130c>
   12e94:	01091913          	slli	s2,s2,0x10
   12e98:	01095913          	srli	s2,s2,0x10
   12e9c:	00000d93          	li	s11,0
   12ea0:	00100793          	li	a5,1
   12ea4:	9fcff06f          	j	120a0 <_vfprintf_r+0x4dc>
   12ea8:	00000913          	li	s2,0
   12eac:	19010d13          	addi	s10,sp,400
   12eb0:	994ff06f          	j	12044 <_vfprintf_r+0x480>
   12eb4:	001cf793          	andi	a5,s9,1
   12eb8:	00079463          	bnez	a5,12ec0 <_vfprintf_r+0x12fc>
   12ebc:	fedfe06f          	j	11ea8 <_vfprintf_r+0x2e4>
   12ec0:	891ff06f          	j	12750 <_vfprintf_r+0xb8c>
   12ec4:	000ac883          	lbu	a7,0(s5)
   12ec8:	00f12a23          	sw	a5,20(sp)
   12ecc:	e89fe06f          	j	11d54 <_vfprintf_r+0x190>
   12ed0:	200ef793          	andi	a5,t4,512
   12ed4:	00078463          	beqz	a5,12edc <_vfprintf_r+0x1318>
   12ed8:	0ff97913          	zext.b	s2,s2
   12edc:	00000d93          	li	s11,0
   12ee0:	00100793          	li	a5,1
   12ee4:	9bcff06f          	j	120a0 <_vfprintf_r+0x4dc>
   12ee8:	200cf793          	andi	a5,s9,512
   12eec:	380792e3          	bnez	a5,13a70 <_vfprintf_r+0x1eac>
   12ef0:	41f95d93          	srai	s11,s2,0x1f
   12ef4:	000d8793          	mv	a5,s11
   12ef8:	924ff06f          	j	1201c <_vfprintf_r+0x458>
   12efc:	200cf793          	andi	a5,s9,512
   12f00:	360792e3          	bnez	a5,13a64 <_vfprintf_r+0x1ea0>
   12f04:	00000d93          	li	s11,0
   12f08:	990ff06f          	j	12098 <_vfprintf_r+0x4d4>
   12f0c:	01412783          	lw	a5,20(sp)
   12f10:	0007a703          	lw	a4,0(a5)
   12f14:	00478793          	addi	a5,a5,4
   12f18:	00f12a23          	sw	a5,20(sp)
   12f1c:	00072583          	lw	a1,0(a4)
   12f20:	00472603          	lw	a2,4(a4)
   12f24:	00872683          	lw	a3,8(a4)
   12f28:	00c72703          	lw	a4,12(a4)
   12f2c:	a20ff06f          	j	1214c <_vfprintf_r+0x588>
   12f30:	03812783          	lw	a5,56(sp)
   12f34:	000ac883          	lbu	a7,0(s5)
   12f38:	00079463          	bnez	a5,12f40 <_vfprintf_r+0x137c>
   12f3c:	e19fe06f          	j	11d54 <_vfprintf_r+0x190>
   12f40:	0007c783          	lbu	a5,0(a5)
   12f44:	00079463          	bnez	a5,12f4c <_vfprintf_r+0x1388>
   12f48:	e0dfe06f          	j	11d54 <_vfprintf_r+0x190>
   12f4c:	400cec93          	ori	s9,s9,1024
   12f50:	e05fe06f          	j	11d54 <_vfprintf_r+0x190>
   12f54:	010cf793          	andi	a5,s9,16
   12f58:	6a079063          	bnez	a5,135f8 <_vfprintf_r+0x1a34>
   12f5c:	040cf793          	andi	a5,s9,64
   12f60:	320798e3          	bnez	a5,13a90 <_vfprintf_r+0x1ecc>
   12f64:	200cfe13          	andi	t3,s9,512
   12f68:	680e0863          	beqz	t3,135f8 <_vfprintf_r+0x1a34>
   12f6c:	01412783          	lw	a5,20(sp)
   12f70:	00c12703          	lw	a4,12(sp)
   12f74:	0007a783          	lw	a5,0(a5)
   12f78:	00e78023          	sb	a4,0(a5)
   12f7c:	e78ff06f          	j	125f4 <_vfprintf_r+0xa30>
   12f80:	00070b13          	mv	s6,a4
   12f84:	cf6048e3          	bgtz	s6,12c74 <_vfprintf_r+0x10b0>
   12f88:	d15ff06f          	j	12c9c <_vfprintf_r+0x10d8>
   12f8c:	fff00b13          	li	s6,-1
   12f90:	00068a93          	mv	s5,a3
   12f94:	dc5fe06f          	j	11d58 <_vfprintf_r+0x194>
   12f98:	0000e797          	auipc	a5,0xe
   12f9c:	9a878793          	addi	a5,a5,-1624 # 20940 <_exit+0x108>
   12fa0:	02f12823          	sw	a5,48(sp)
   12fa4:	020cf793          	andi	a5,s9,32
   12fa8:	36078863          	beqz	a5,13318 <_vfprintf_r+0x1754>
   12fac:	01412783          	lw	a5,20(sp)
   12fb0:	00778c13          	addi	s8,a5,7
   12fb4:	ff8c7c13          	andi	s8,s8,-8
   12fb8:	000c2903          	lw	s2,0(s8)
   12fbc:	004c2d83          	lw	s11,4(s8)
   12fc0:	008c0793          	addi	a5,s8,8
   12fc4:	00f12a23          	sw	a5,20(sp)
   12fc8:	001cf793          	andi	a5,s9,1
   12fcc:	00078e63          	beqz	a5,12fe8 <_vfprintf_r+0x1424>
   12fd0:	01b967b3          	or	a5,s2,s11
   12fd4:	00078a63          	beqz	a5,12fe8 <_vfprintf_r+0x1424>
   12fd8:	03000793          	li	a5,48
   12fdc:	0af10423          	sb	a5,168(sp)
   12fe0:	0b1104a3          	sb	a7,169(sp)
   12fe4:	002cec93          	ori	s9,s9,2
   12fe8:	bffcfe93          	andi	t4,s9,-1025
   12fec:	00200793          	li	a5,2
   12ff0:	8b0ff06f          	j	120a0 <_vfprintf_r+0x4dc>
   12ff4:	000c8e93          	mv	t4,s9
   12ff8:	d64ff06f          	j	1255c <_vfprintf_r+0x998>
   12ffc:	0000e797          	auipc	a5,0xe
   13000:	93078793          	addi	a5,a5,-1744 # 2092c <_exit+0xf4>
   13004:	02f12823          	sw	a5,48(sp)
   13008:	f9dff06f          	j	12fa4 <_vfprintf_r+0x13e0>
   1300c:	00812503          	lw	a0,8(sp)
   13010:	0c410613          	addi	a2,sp,196
   13014:	00048593          	mv	a1,s1
   13018:	698010ef          	jal	146b0 <__sprint_r>
   1301c:	a20518e3          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   13020:	0cc12603          	lw	a2,204(sp)
   13024:	00098a13          	mv	s4,s3
   13028:	f58ff06f          	j	12780 <_vfprintf_r+0xbbc>
   1302c:	001ac883          	lbu	a7,1(s5)
   13030:	200cec93          	ori	s9,s9,512
   13034:	001a8a93          	addi	s5,s5,1
   13038:	d1dfe06f          	j	11d54 <_vfprintf_r+0x190>
   1303c:	001ac883          	lbu	a7,1(s5)
   13040:	020cec93          	ori	s9,s9,32
   13044:	001a8a93          	addi	s5,s5,1
   13048:	d0dfe06f          	j	11d54 <_vfprintf_r+0x190>
   1304c:	00600793          	li	a5,6
   13050:	000b0913          	mv	s2,s6
   13054:	2167e4e3          	bltu	a5,s6,13a5c <_vfprintf_r+0x1e98>
   13058:	00090d93          	mv	s11,s2
   1305c:	01812a23          	sw	s8,20(sp)
   13060:	0000ed17          	auipc	s10,0xe
   13064:	8f4d0d13          	addi	s10,s10,-1804 # 20954 <_exit+0x11c>
   13068:	d61fe06f          	j	11dc8 <_vfprintf_r+0x204>
   1306c:	01000593          	li	a1,16
   13070:	0c812703          	lw	a4,200(sp)
   13074:	0000e817          	auipc	a6,0xe
   13078:	bfc80813          	addi	a6,a6,-1028 # 20c70 <zeroes.0>
   1307c:	6965d663          	bge	a1,s6,13708 <_vfprintf_r+0x1b44>
   13080:	000a0793          	mv	a5,s4
   13084:	01000c13          	li	s8,16
   13088:	00700913          	li	s2,7
   1308c:	00080a13          	mv	s4,a6
   13090:	00c0006f          	j	1309c <_vfprintf_r+0x14d8>
   13094:	ff0b0b13          	addi	s6,s6,-16
   13098:	676c5463          	bge	s8,s6,13700 <_vfprintf_r+0x1b3c>
   1309c:	01060613          	addi	a2,a2,16
   130a0:	00170713          	addi	a4,a4,1
   130a4:	0147a023          	sw	s4,0(a5)
   130a8:	0187a223          	sw	s8,4(a5)
   130ac:	0cc12623          	sw	a2,204(sp)
   130b0:	0ce12423          	sw	a4,200(sp)
   130b4:	00878793          	addi	a5,a5,8
   130b8:	fce95ee3          	bge	s2,a4,13094 <_vfprintf_r+0x14d0>
   130bc:	00812503          	lw	a0,8(sp)
   130c0:	0c410613          	addi	a2,sp,196
   130c4:	00048593          	mv	a1,s1
   130c8:	5e8010ef          	jal	146b0 <__sprint_r>
   130cc:	980510e3          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   130d0:	0cc12603          	lw	a2,204(sp)
   130d4:	0c812703          	lw	a4,200(sp)
   130d8:	00098793          	mv	a5,s3
   130dc:	fb9ff06f          	j	13094 <_vfprintf_r+0x14d0>
   130e0:	02c12703          	lw	a4,44(sp)
   130e4:	02812783          	lw	a5,40(sp)
   130e8:	00700513          	li	a0,7
   130ec:	00ea2023          	sw	a4,0(s4)
   130f0:	0c812703          	lw	a4,200(sp)
   130f4:	00f60633          	add	a2,a2,a5
   130f8:	00fa2223          	sw	a5,4(s4)
   130fc:	00170713          	addi	a4,a4,1
   13100:	0cc12623          	sw	a2,204(sp)
   13104:	0ce12423          	sw	a4,200(sp)
   13108:	008a0a13          	addi	s4,s4,8
   1310c:	bce556e3          	bge	a0,a4,12cd8 <_vfprintf_r+0x1114>
   13110:	00812503          	lw	a0,8(sp)
   13114:	0c410613          	addi	a2,sp,196
   13118:	00048593          	mv	a1,s1
   1311c:	594010ef          	jal	146b0 <__sprint_r>
   13120:	920516e3          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   13124:	0ac12583          	lw	a1,172(sp)
   13128:	0cc12603          	lw	a2,204(sp)
   1312c:	00098a13          	mv	s4,s3
   13130:	ba9ff06f          	j	12cd8 <_vfprintf_r+0x1114>
   13134:	07800793          	li	a5,120
   13138:	03000713          	li	a4,48
   1313c:	0ae10423          	sb	a4,168(sp)
   13140:	0af104a3          	sb	a5,169(sp)
   13144:	06300713          	li	a4,99
   13148:	00012823          	sw	zero,16(sp)
   1314c:	002cec93          	ori	s9,s9,2
   13150:	12c10d13          	addi	s10,sp,300
   13154:	87675863          	bge	a4,s6,121c4 <_vfprintf_r+0x600>
   13158:	00812503          	lw	a0,8(sp)
   1315c:	001b0593          	addi	a1,s6,1
   13160:	01112823          	sw	a7,16(sp)
   13164:	a90fe0ef          	jal	113f4 <_malloc_r>
   13168:	01012883          	lw	a7,16(sp)
   1316c:	00050d13          	mv	s10,a0
   13170:	00051463          	bnez	a0,13178 <_vfprintf_r+0x15b4>
   13174:	3280106f          	j	1449c <_vfprintf_r+0x28d8>
   13178:	00a12823          	sw	a0,16(sp)
   1317c:	848ff06f          	j	121c4 <_vfprintf_r+0x600>
   13180:	00812503          	lw	a0,8(sp)
   13184:	09010913          	addi	s2,sp,144
   13188:	0ac10713          	addi	a4,sp,172
   1318c:	0bc10813          	addi	a6,sp,188
   13190:	0b010793          	addi	a5,sp,176
   13194:	000b0693          	mv	a3,s6
   13198:	00200613          	li	a2,2
   1319c:	00090593          	mv	a1,s2
   131a0:	03112223          	sw	a7,36(sp)
   131a4:	09f12823          	sw	t6,144(sp)
   131a8:	03f12023          	sw	t6,32(sp)
   131ac:	09e12a23          	sw	t5,148(sp)
   131b0:	01e12e23          	sw	t5,28(sp)
   131b4:	09d12c23          	sw	t4,152(sp)
   131b8:	01d12c23          	sw	t4,24(sp)
   131bc:	09812e23          	sw	s8,156(sp)
   131c0:	709030ef          	jal	170c8 <_ldtoa_r>
   131c4:	001cf713          	andi	a4,s9,1
   131c8:	01812e83          	lw	t4,24(sp)
   131cc:	01c12f03          	lw	t5,28(sp)
   131d0:	02012f83          	lw	t6,32(sp)
   131d4:	02412883          	lw	a7,36(sp)
   131d8:	00050d13          	mv	s10,a0
   131dc:	10071ce3          	bnez	a4,13af4 <_vfprintf_r+0x1f30>
   131e0:	0ac12783          	lw	a5,172(sp)
   131e4:	00f12c23          	sw	a5,24(sp)
   131e8:	0bc12783          	lw	a5,188(sp)
   131ec:	40a787b3          	sub	a5,a5,a0
   131f0:	00f12e23          	sw	a5,28(sp)
   131f4:	01812783          	lw	a5,24(sp)
   131f8:	ffd00713          	li	a4,-3
   131fc:	00e7c463          	blt	a5,a4,13204 <_vfprintf_r+0x1640>
   13200:	60fb56e3          	bge	s6,a5,1400c <_vfprintf_r+0x2448>
   13204:	01812783          	lw	a5,24(sp)
   13208:	ffe88893          	addi	a7,a7,-2
   1320c:	fff78713          	addi	a4,a5,-1
   13210:	0ae12623          	sw	a4,172(sp)
   13214:	0ff8f693          	zext.b	a3,a7
   13218:	00000613          	li	a2,0
   1321c:	0ad10a23          	sb	a3,180(sp)
   13220:	02b00693          	li	a3,43
   13224:	00075a63          	bgez	a4,13238 <_vfprintf_r+0x1674>
   13228:	01812783          	lw	a5,24(sp)
   1322c:	00100713          	li	a4,1
   13230:	02d00693          	li	a3,45
   13234:	40f70733          	sub	a4,a4,a5
   13238:	0ad10aa3          	sb	a3,181(sp)
   1323c:	00900693          	li	a3,9
   13240:	00e6c463          	blt	a3,a4,13248 <_vfprintf_r+0x1684>
   13244:	0900106f          	j	142d4 <_vfprintf_r+0x2710>
   13248:	0c310813          	addi	a6,sp,195
   1324c:	00080e93          	mv	t4,a6
   13250:	00a00613          	li	a2,10
   13254:	06300f13          	li	t5,99
   13258:	02c767b3          	rem	a5,a4,a2
   1325c:	000e8513          	mv	a0,t4
   13260:	00070693          	mv	a3,a4
   13264:	fffe8e93          	addi	t4,t4,-1
   13268:	03078793          	addi	a5,a5,48
   1326c:	fef50fa3          	sb	a5,-1(a0)
   13270:	02c74733          	div	a4,a4,a2
   13274:	fedf42e3          	blt	t5,a3,13258 <_vfprintf_r+0x1694>
   13278:	03070713          	addi	a4,a4,48
   1327c:	ffe50693          	addi	a3,a0,-2
   13280:	feee8fa3          	sb	a4,-1(t4)
   13284:	0106e463          	bltu	a3,a6,1328c <_vfprintf_r+0x16c8>
   13288:	1cc0106f          	j	14454 <_vfprintf_r+0x2890>
   1328c:	0b610613          	addi	a2,sp,182
   13290:	0006c783          	lbu	a5,0(a3)
   13294:	00168693          	addi	a3,a3,1
   13298:	00160613          	addi	a2,a2,1
   1329c:	fef60fa3          	sb	a5,-1(a2)
   132a0:	ff0698e3          	bne	a3,a6,13290 <_vfprintf_r+0x16cc>
   132a4:	19010793          	addi	a5,sp,400
   132a8:	40a78733          	sub	a4,a5,a0
   132ac:	f3770793          	addi	a5,a4,-201
   132b0:	02f12e23          	sw	a5,60(sp)
   132b4:	01c12783          	lw	a5,28(sp)
   132b8:	03c12683          	lw	a3,60(sp)
   132bc:	00100713          	li	a4,1
   132c0:	00d78933          	add	s2,a5,a3
   132c4:	00f74463          	blt	a4,a5,132cc <_vfprintf_r+0x1708>
   132c8:	03c0106f          	j	14304 <_vfprintf_r+0x2740>
   132cc:	02812783          	lw	a5,40(sp)
   132d0:	00f90933          	add	s2,s2,a5
   132d4:	fff94693          	not	a3,s2
   132d8:	bffcfe13          	andi	t3,s9,-1025
   132dc:	41f6d693          	srai	a3,a3,0x1f
   132e0:	100e6793          	ori	a5,t3,256
   132e4:	04f12423          	sw	a5,72(sp)
   132e8:	00d97db3          	and	s11,s2,a3
   132ec:	02012223          	sw	zero,36(sp)
   132f0:	02012023          	sw	zero,32(sp)
   132f4:	00012c23          	sw	zero,24(sp)
   132f8:	03412783          	lw	a5,52(sp)
   132fc:	4e078ce3          	beqz	a5,13ff4 <_vfprintf_r+0x2430>
   13300:	02d00713          	li	a4,45
   13304:	04812c83          	lw	s9,72(sp)
   13308:	0ae103a3          	sb	a4,167(sp)
   1330c:	00000b13          	li	s6,0
   13310:	001d8d93          	addi	s11,s11,1
   13314:	ac9fe06f          	j	11ddc <_vfprintf_r+0x218>
   13318:	01412703          	lw	a4,20(sp)
   1331c:	010cf793          	andi	a5,s9,16
   13320:	00072903          	lw	s2,0(a4)
   13324:	00470713          	addi	a4,a4,4
   13328:	00e12a23          	sw	a4,20(sp)
   1332c:	06079663          	bnez	a5,13398 <_vfprintf_r+0x17d4>
   13330:	040cf793          	andi	a5,s9,64
   13334:	04078e63          	beqz	a5,13390 <_vfprintf_r+0x17cc>
   13338:	01091913          	slli	s2,s2,0x10
   1333c:	01095913          	srli	s2,s2,0x10
   13340:	00000d93          	li	s11,0
   13344:	c85ff06f          	j	12fc8 <_vfprintf_r+0x1404>
   13348:	00812503          	lw	a0,8(sp)
   1334c:	0c410613          	addi	a2,sp,196
   13350:	00048593          	mv	a1,s1
   13354:	35c010ef          	jal	146b0 <__sprint_r>
   13358:	ee051a63          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   1335c:	0cc12c03          	lw	s8,204(sp)
   13360:	0c812903          	lw	s2,200(sp)
   13364:	00098b13          	mv	s6,s3
   13368:	f29fe06f          	j	12290 <_vfprintf_r+0x6cc>
   1336c:	00812503          	lw	a0,8(sp)
   13370:	0c410613          	addi	a2,sp,196
   13374:	00048593          	mv	a1,s1
   13378:	338010ef          	jal	146b0 <__sprint_r>
   1337c:	ec051863          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   13380:	0cc12c03          	lw	s8,204(sp)
   13384:	0c812903          	lw	s2,200(sp)
   13388:	00098b13          	mv	s6,s3
   1338c:	f35fe06f          	j	122c0 <_vfprintf_r+0x6fc>
   13390:	200cf793          	andi	a5,s9,512
   13394:	6e079863          	bnez	a5,13a84 <_vfprintf_r+0x1ec0>
   13398:	00000d93          	li	s11,0
   1339c:	c2dff06f          	j	12fc8 <_vfprintf_r+0x1404>
   133a0:	ccccd837          	lui	a6,0xccccd
   133a4:	ccccdcb7          	lui	s9,0xccccd
   133a8:	03812703          	lw	a4,56(sp)
   133ac:	400eff13          	andi	t5,t4,1024
   133b0:	00000613          	li	a2,0
   133b4:	19010593          	addi	a1,sp,400
   133b8:	00500e13          	li	t3,5
   133bc:	ccd80813          	addi	a6,a6,-819 # cccccccd <__BSS_END__+0xccca9ecd>
   133c0:	cccc8c93          	addi	s9,s9,-820 # cccccccc <__BSS_END__+0xccca9ecc>
   133c4:	0ff00c13          	li	s8,255
   133c8:	0540006f          	j	1341c <_vfprintf_r+0x1858>
   133cc:	012d37b3          	sltu	a5,s10,s2
   133d0:	00fd07b3          	add	a5,s10,a5
   133d4:	03c7f7b3          	remu	a5,a5,t3
   133d8:	40f907b3          	sub	a5,s2,a5
   133dc:	00f935b3          	sltu	a1,s2,a5
   133e0:	40bd85b3          	sub	a1,s11,a1
   133e4:	03978333          	mul	t1,a5,s9
   133e8:	030585b3          	mul	a1,a1,a6
   133ec:	0307b533          	mulhu	a0,a5,a6
   133f0:	006585b3          	add	a1,a1,t1
   133f4:	030787b3          	mul	a5,a5,a6
   133f8:	00a585b3          	add	a1,a1,a0
   133fc:	01f59513          	slli	a0,a1,0x1f
   13400:	0015d593          	srli	a1,a1,0x1
   13404:	0017d793          	srli	a5,a5,0x1
   13408:	00f567b3          	or	a5,a0,a5
   1340c:	480d82e3          	beqz	s11,14090 <_vfprintf_r+0x24cc>
   13410:	00058d93          	mv	s11,a1
   13414:	00078913          	mv	s2,a5
   13418:	00068593          	mv	a1,a3
   1341c:	01b90d33          	add	s10,s2,s11
   13420:	012d37b3          	sltu	a5,s10,s2
   13424:	00fd07b3          	add	a5,s10,a5
   13428:	03c7f7b3          	remu	a5,a5,t3
   1342c:	fff58693          	addi	a3,a1,-1
   13430:	00160613          	addi	a2,a2,1
   13434:	40f907b3          	sub	a5,s2,a5
   13438:	00f93533          	sltu	a0,s2,a5
   1343c:	40ad8533          	sub	a0,s11,a0
   13440:	0307b333          	mulhu	t1,a5,a6
   13444:	03050533          	mul	a0,a0,a6
   13448:	030787b3          	mul	a5,a5,a6
   1344c:	00650533          	add	a0,a0,t1
   13450:	01f51513          	slli	a0,a0,0x1f
   13454:	0017d793          	srli	a5,a5,0x1
   13458:	00f567b3          	or	a5,a0,a5
   1345c:	00279513          	slli	a0,a5,0x2
   13460:	00f507b3          	add	a5,a0,a5
   13464:	00179793          	slli	a5,a5,0x1
   13468:	40f907b3          	sub	a5,s2,a5
   1346c:	03078793          	addi	a5,a5,48
   13470:	fef58fa3          	sb	a5,-1(a1)
   13474:	f40f0ce3          	beqz	t5,133cc <_vfprintf_r+0x1808>
   13478:	00074783          	lbu	a5,0(a4)
   1347c:	f4f618e3          	bne	a2,a5,133cc <_vfprintf_r+0x1808>
   13480:	f58606e3          	beq	a2,s8,133cc <_vfprintf_r+0x1808>
   13484:	4c0d9c63          	bnez	s11,1395c <_vfprintf_r+0x1d98>
   13488:	00900793          	li	a5,9
   1348c:	4d27e863          	bltu	a5,s2,1395c <_vfprintf_r+0x1d98>
   13490:	00068d13          	mv	s10,a3
   13494:	19010793          	addi	a5,sp,400
   13498:	00c12e23          	sw	a2,28(sp)
   1349c:	02e12c23          	sw	a4,56(sp)
   134a0:	41a78933          	sub	s2,a5,s10
   134a4:	000e8c93          	mv	s9,t4
   134a8:	b9dfe06f          	j	12044 <_vfprintf_r+0x480>
   134ac:	0c812703          	lw	a4,200(sp)
   134b0:	0000d517          	auipc	a0,0xd
   134b4:	4ac50513          	addi	a0,a0,1196 # 2095c <_exit+0x124>
   134b8:	00aa2023          	sw	a0,0(s4)
   134bc:	00160613          	addi	a2,a2,1
   134c0:	00100513          	li	a0,1
   134c4:	00170713          	addi	a4,a4,1
   134c8:	00aa2223          	sw	a0,4(s4)
   134cc:	0cc12623          	sw	a2,204(sp)
   134d0:	0ce12423          	sw	a4,200(sp)
   134d4:	00700513          	li	a0,7
   134d8:	008a0a13          	addi	s4,s4,8
   134dc:	52e54a63          	blt	a0,a4,13a10 <_vfprintf_r+0x1e4c>
   134e0:	12059663          	bnez	a1,1360c <_vfprintf_r+0x1a48>
   134e4:	01c12783          	lw	a5,28(sp)
   134e8:	001cf713          	andi	a4,s9,1
   134ec:	00f76733          	or	a4,a4,a5
   134f0:	00071463          	bnez	a4,134f8 <_vfprintf_r+0x1934>
   134f4:	9b5fe06f          	j	11ea8 <_vfprintf_r+0x2e4>
   134f8:	02c12703          	lw	a4,44(sp)
   134fc:	02812783          	lw	a5,40(sp)
   13500:	00700593          	li	a1,7
   13504:	00ea2023          	sw	a4,0(s4)
   13508:	0c812703          	lw	a4,200(sp)
   1350c:	00c78633          	add	a2,a5,a2
   13510:	00fa2223          	sw	a5,4(s4)
   13514:	00170713          	addi	a4,a4,1
   13518:	0cc12623          	sw	a2,204(sp)
   1351c:	0ce12423          	sw	a4,200(sp)
   13520:	5ae5c463          	blt	a1,a4,13ac8 <_vfprintf_r+0x1f04>
   13524:	008a0a13          	addi	s4,s4,8
   13528:	1180006f          	j	13640 <_vfprintf_r+0x1a7c>
   1352c:	00812503          	lw	a0,8(sp)
   13530:	934fd0ef          	jal	10664 <__sinit>
   13534:	ef0fe06f          	j	11c24 <_vfprintf_r+0x60>
   13538:	00812503          	lw	a0,8(sp)
   1353c:	0c410613          	addi	a2,sp,196
   13540:	00048593          	mv	a1,s1
   13544:	16c010ef          	jal	146b0 <__sprint_r>
   13548:	d0051263          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   1354c:	0cc12603          	lw	a2,204(sp)
   13550:	00098a13          	mv	s4,s3
   13554:	9f0ff06f          	j	12744 <_vfprintf_r+0xb80>
   13558:	000d0813          	mv	a6,s10
   1355c:	01260633          	add	a2,a2,s2
   13560:	00178793          	addi	a5,a5,1
   13564:	010a2023          	sw	a6,0(s4)
   13568:	929fe06f          	j	11e90 <_vfprintf_r+0x2cc>
   1356c:	0d012783          	lw	a5,208(sp)
   13570:	08010593          	addi	a1,sp,128
   13574:	09010513          	addi	a0,sp,144
   13578:	08f12823          	sw	a5,144(sp)
   1357c:	0d412783          	lw	a5,212(sp)
   13580:	08012023          	sw	zero,128(sp)
   13584:	08012223          	sw	zero,132(sp)
   13588:	08f12a23          	sw	a5,148(sp)
   1358c:	0d812783          	lw	a5,216(sp)
   13590:	08012423          	sw	zero,136(sp)
   13594:	08012623          	sw	zero,140(sp)
   13598:	08f12c23          	sw	a5,152(sp)
   1359c:	0dc12783          	lw	a5,220(sp)
   135a0:	08f12e23          	sw	a5,156(sp)
   135a4:	6f00a0ef          	jal	1dc94 <__letf2>
   135a8:	01012883          	lw	a7,16(sp)
   135ac:	2e0548e3          	bltz	a0,1409c <_vfprintf_r+0x24d8>
   135b0:	0a714703          	lbu	a4,167(sp)
   135b4:	04700693          	li	a3,71
   135b8:	0000dd17          	auipc	s10,0xd
   135bc:	368d0d13          	addi	s10,s10,872 # 20920 <_exit+0xe8>
   135c0:	0116c663          	blt	a3,a7,135cc <_vfprintf_r+0x1a08>
   135c4:	0000dd17          	auipc	s10,0xd
   135c8:	358d0d13          	addi	s10,s10,856 # 2091c <_exit+0xe4>
   135cc:	00012823          	sw	zero,16(sp)
   135d0:	02012223          	sw	zero,36(sp)
   135d4:	02012023          	sw	zero,32(sp)
   135d8:	00012c23          	sw	zero,24(sp)
   135dc:	f7fcfc93          	andi	s9,s9,-129
   135e0:	00300d93          	li	s11,3
   135e4:	00300913          	li	s2,3
   135e8:	00000b13          	li	s6,0
   135ec:	00070463          	beqz	a4,135f4 <_vfprintf_r+0x1a30>
   135f0:	a79fe06f          	j	12068 <_vfprintf_r+0x4a4>
   135f4:	fe8fe06f          	j	11ddc <_vfprintf_r+0x218>
   135f8:	01412783          	lw	a5,20(sp)
   135fc:	00c12703          	lw	a4,12(sp)
   13600:	0007a783          	lw	a5,0(a5)
   13604:	00e7a023          	sw	a4,0(a5)
   13608:	fedfe06f          	j	125f4 <_vfprintf_r+0xa30>
   1360c:	02c12703          	lw	a4,44(sp)
   13610:	02812783          	lw	a5,40(sp)
   13614:	00700513          	li	a0,7
   13618:	00ea2023          	sw	a4,0(s4)
   1361c:	0c812703          	lw	a4,200(sp)
   13620:	00c78633          	add	a2,a5,a2
   13624:	00fa2223          	sw	a5,4(s4)
   13628:	00170713          	addi	a4,a4,1
   1362c:	0cc12623          	sw	a2,204(sp)
   13630:	0ce12423          	sw	a4,200(sp)
   13634:	008a0a13          	addi	s4,s4,8
   13638:	48e54863          	blt	a0,a4,13ac8 <_vfprintf_r+0x1f04>
   1363c:	3205c0e3          	bltz	a1,1415c <_vfprintf_r+0x2598>
   13640:	01c12783          	lw	a5,28(sp)
   13644:	00170713          	addi	a4,a4,1
   13648:	01aa2023          	sw	s10,0(s4)
   1364c:	00c78633          	add	a2,a5,a2
   13650:	00fa2223          	sw	a5,4(s4)
   13654:	0cc12623          	sw	a2,204(sp)
   13658:	0ce12423          	sw	a4,200(sp)
   1365c:	00700793          	li	a5,7
   13660:	00e7c463          	blt	a5,a4,13668 <_vfprintf_r+0x1aa4>
   13664:	841fe06f          	j	11ea4 <_vfprintf_r+0x2e0>
   13668:	cf5fe06f          	j	1235c <_vfprintf_r+0x798>
   1366c:	000d0513          	mv	a0,s10
   13670:	03112a23          	sw	a7,52(sp)
   13674:	f6cfd0ef          	jal	10de0 <strlen>
   13678:	0a714703          	lbu	a4,167(sp)
   1367c:	fff54693          	not	a3,a0
   13680:	41f6d693          	srai	a3,a3,0x1f
   13684:	01812a23          	sw	s8,20(sp)
   13688:	00012823          	sw	zero,16(sp)
   1368c:	02012223          	sw	zero,36(sp)
   13690:	02012023          	sw	zero,32(sp)
   13694:	00012c23          	sw	zero,24(sp)
   13698:	03412883          	lw	a7,52(sp)
   1369c:	00050913          	mv	s2,a0
   136a0:	00d57db3          	and	s11,a0,a3
   136a4:	00000b13          	li	s6,0
   136a8:	00070463          	beqz	a4,136b0 <_vfprintf_r+0x1aec>
   136ac:	9bdfe06f          	j	12068 <_vfprintf_r+0x4a4>
   136b0:	f2cfe06f          	j	11ddc <_vfprintf_r+0x218>
   136b4:	000a8813          	mv	a6,s5
   136b8:	01812a83          	lw	s5,24(sp)
   136bc:	000a0793          	mv	a5,s4
   136c0:	00048a13          	mv	s4,s1
   136c4:	00078493          	mv	s1,a5
   136c8:	014c0c33          	add	s8,s8,s4
   136cc:	00190913          	addi	s2,s2,1
   136d0:	010b2023          	sw	a6,0(s6)
   136d4:	c45fe06f          	j	12318 <_vfprintf_r+0x754>
   136d8:	0dc12783          	lw	a5,220(sp)
   136dc:	3c07ce63          	bltz	a5,13ab8 <_vfprintf_r+0x1ef4>
   136e0:	0a714703          	lbu	a4,167(sp)
   136e4:	04700693          	li	a3,71
   136e8:	0000dd17          	auipc	s10,0xd
   136ec:	240d0d13          	addi	s10,s10,576 # 20928 <_exit+0xf0>
   136f0:	ed16cee3          	blt	a3,a7,135cc <_vfprintf_r+0x1a08>
   136f4:	0000dd17          	auipc	s10,0xd
   136f8:	230d0d13          	addi	s10,s10,560 # 20924 <_exit+0xec>
   136fc:	ed1ff06f          	j	135cc <_vfprintf_r+0x1a08>
   13700:	000a0813          	mv	a6,s4
   13704:	00078a13          	mv	s4,a5
   13708:	01660633          	add	a2,a2,s6
   1370c:	00170713          	addi	a4,a4,1
   13710:	010a2023          	sw	a6,0(s4)
   13714:	016a2223          	sw	s6,4(s4)
   13718:	0cc12623          	sw	a2,204(sp)
   1371c:	0ce12423          	sw	a4,200(sp)
   13720:	00700593          	li	a1,7
   13724:	008a0a13          	addi	s4,s4,8
   13728:	d8e5d663          	bge	a1,a4,12cb4 <_vfprintf_r+0x10f0>
   1372c:	00812503          	lw	a0,8(sp)
   13730:	0c410613          	addi	a2,sp,196
   13734:	00048593          	mv	a1,s1
   13738:	779000ef          	jal	146b0 <__sprint_r>
   1373c:	b0051863          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   13740:	0cc12603          	lw	a2,204(sp)
   13744:	00098a13          	mv	s4,s3
   13748:	d6cff06f          	j	12cb4 <_vfprintf_r+0x10f0>
   1374c:	01c12783          	lw	a5,28(sp)
   13750:	03512a23          	sw	s5,52(sp)
   13754:	02012a83          	lw	s5,32(sp)
   13758:	00fd07b3          	add	a5,s10,a5
   1375c:	05912423          	sw	s9,72(sp)
   13760:	05712623          	sw	s7,76(sp)
   13764:	03b12023          	sw	s11,32(sp)
   13768:	02412d83          	lw	s11,36(sp)
   1376c:	03a12223          	sw	s10,36(sp)
   13770:	03812c83          	lw	s9,56(sp)
   13774:	000c0d13          	mv	s10,s8
   13778:	00812903          	lw	s2,8(sp)
   1377c:	04412c03          	lw	s8,68(sp)
   13780:	00700813          	li	a6,7
   13784:	01000713          	li	a4,16
   13788:	0000db17          	auipc	s6,0xd
   1378c:	4e8b0b13          	addi	s6,s6,1256 # 20c70 <zeroes.0>
   13790:	000a0593          	mv	a1,s4
   13794:	00078b93          	mv	s7,a5
   13798:	09505663          	blez	s5,13824 <_vfprintf_r+0x1c60>
   1379c:	17b05063          	blez	s11,138fc <_vfprintf_r+0x1d38>
   137a0:	fffd8d93          	addi	s11,s11,-1
   137a4:	04012783          	lw	a5,64(sp)
   137a8:	01860633          	add	a2,a2,s8
   137ac:	0185a223          	sw	s8,4(a1)
   137b0:	00f5a023          	sw	a5,0(a1)
   137b4:	0c812783          	lw	a5,200(sp)
   137b8:	0cc12623          	sw	a2,204(sp)
   137bc:	00858593          	addi	a1,a1,8
   137c0:	00178793          	addi	a5,a5,1
   137c4:	0cf12423          	sw	a5,200(sp)
   137c8:	14f84063          	blt	a6,a5,13908 <_vfprintf_r+0x1d44>
   137cc:	000cc683          	lbu	a3,0(s9)
   137d0:	41ab8a33          	sub	s4,s7,s10
   137d4:	0146d463          	bge	a3,s4,137dc <_vfprintf_r+0x1c18>
   137d8:	00068a13          	mv	s4,a3
   137dc:	03405663          	blez	s4,13808 <_vfprintf_r+0x1c44>
   137e0:	0c812683          	lw	a3,200(sp)
   137e4:	01460633          	add	a2,a2,s4
   137e8:	01a5a023          	sw	s10,0(a1)
   137ec:	00168693          	addi	a3,a3,1
   137f0:	0145a223          	sw	s4,4(a1)
   137f4:	0cc12623          	sw	a2,204(sp)
   137f8:	0cd12423          	sw	a3,200(sp)
   137fc:	12d84a63          	blt	a6,a3,13930 <_vfprintf_r+0x1d6c>
   13800:	000cc683          	lbu	a3,0(s9)
   13804:	00858593          	addi	a1,a1,8
   13808:	fffa4513          	not	a0,s4
   1380c:	41f55513          	srai	a0,a0,0x1f
   13810:	00aa77b3          	and	a5,s4,a0
   13814:	40f68a33          	sub	s4,a3,a5
   13818:	05404263          	bgtz	s4,1385c <_vfprintf_r+0x1c98>
   1381c:	00dd0d33          	add	s10,s10,a3
   13820:	f7504ee3          	bgtz	s5,1379c <_vfprintf_r+0x1bd8>
   13824:	f7b04ee3          	bgtz	s11,137a0 <_vfprintf_r+0x1bdc>
   13828:	01c12783          	lw	a5,28(sp)
   1382c:	000d0c13          	mv	s8,s10
   13830:	02412d03          	lw	s10,36(sp)
   13834:	03912c23          	sw	s9,56(sp)
   13838:	03412a83          	lw	s5,52(sp)
   1383c:	00fd0733          	add	a4,s10,a5
   13840:	04812c83          	lw	s9,72(sp)
   13844:	04c12b83          	lw	s7,76(sp)
   13848:	02012d83          	lw	s11,32(sp)
   1384c:	00058a13          	mv	s4,a1
   13850:	c7877a63          	bgeu	a4,s8,12cc4 <_vfprintf_r+0x1100>
   13854:	00070c13          	mv	s8,a4
   13858:	c6cff06f          	j	12cc4 <_vfprintf_r+0x1100>
   1385c:	0c812683          	lw	a3,200(sp)
   13860:	0000df17          	auipc	t5,0xd
   13864:	410f0f13          	addi	t5,t5,1040 # 20c70 <zeroes.0>
   13868:	07475463          	bge	a4,s4,138d0 <_vfprintf_r+0x1d0c>
   1386c:	01612c23          	sw	s6,24(sp)
   13870:	00c0006f          	j	1387c <_vfprintf_r+0x1cb8>
   13874:	ff0a0a13          	addi	s4,s4,-16
   13878:	05475a63          	bge	a4,s4,138cc <_vfprintf_r+0x1d08>
   1387c:	01060613          	addi	a2,a2,16
   13880:	00168693          	addi	a3,a3,1
   13884:	0165a023          	sw	s6,0(a1)
   13888:	00e5a223          	sw	a4,4(a1)
   1388c:	0cc12623          	sw	a2,204(sp)
   13890:	0cd12423          	sw	a3,200(sp)
   13894:	00858593          	addi	a1,a1,8
   13898:	fcd85ee3          	bge	a6,a3,13874 <_vfprintf_r+0x1cb0>
   1389c:	0c410613          	addi	a2,sp,196
   138a0:	00048593          	mv	a1,s1
   138a4:	00090513          	mv	a0,s2
   138a8:	609000ef          	jal	146b0 <__sprint_r>
   138ac:	9a051063          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   138b0:	01000713          	li	a4,16
   138b4:	ff0a0a13          	addi	s4,s4,-16
   138b8:	0cc12603          	lw	a2,204(sp)
   138bc:	0c812683          	lw	a3,200(sp)
   138c0:	00098593          	mv	a1,s3
   138c4:	00700813          	li	a6,7
   138c8:	fb474ae3          	blt	a4,s4,1387c <_vfprintf_r+0x1cb8>
   138cc:	01812f03          	lw	t5,24(sp)
   138d0:	01460633          	add	a2,a2,s4
   138d4:	00168693          	addi	a3,a3,1
   138d8:	01e5a023          	sw	t5,0(a1)
   138dc:	0145a223          	sw	s4,4(a1)
   138e0:	0cc12623          	sw	a2,204(sp)
   138e4:	0cd12423          	sw	a3,200(sp)
   138e8:	76d84a63          	blt	a6,a3,1405c <_vfprintf_r+0x2498>
   138ec:	000cc683          	lbu	a3,0(s9)
   138f0:	00858593          	addi	a1,a1,8
   138f4:	00dd0d33          	add	s10,s10,a3
   138f8:	f29ff06f          	j	13820 <_vfprintf_r+0x1c5c>
   138fc:	fffc8c93          	addi	s9,s9,-1
   13900:	fffa8a93          	addi	s5,s5,-1
   13904:	ea1ff06f          	j	137a4 <_vfprintf_r+0x1be0>
   13908:	0c410613          	addi	a2,sp,196
   1390c:	00048593          	mv	a1,s1
   13910:	00090513          	mv	a0,s2
   13914:	59d000ef          	jal	146b0 <__sprint_r>
   13918:	92051a63          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   1391c:	0cc12603          	lw	a2,204(sp)
   13920:	00098593          	mv	a1,s3
   13924:	01000713          	li	a4,16
   13928:	00700813          	li	a6,7
   1392c:	ea1ff06f          	j	137cc <_vfprintf_r+0x1c08>
   13930:	0c410613          	addi	a2,sp,196
   13934:	00048593          	mv	a1,s1
   13938:	00090513          	mv	a0,s2
   1393c:	575000ef          	jal	146b0 <__sprint_r>
   13940:	90051663          	bnez	a0,12a4c <_vfprintf_r+0xe88>
   13944:	000cc683          	lbu	a3,0(s9)
   13948:	0cc12603          	lw	a2,204(sp)
   1394c:	00098593          	mv	a1,s3
   13950:	01000713          	li	a4,16
   13954:	00700813          	li	a6,7
   13958:	eb1ff06f          	j	13808 <_vfprintf_r+0x1c44>
   1395c:	04412783          	lw	a5,68(sp)
   13960:	04012583          	lw	a1,64(sp)
   13964:	03012a23          	sw	a6,52(sp)
   13968:	40f686b3          	sub	a3,a3,a5
   1396c:	00078613          	mv	a2,a5
   13970:	00068513          	mv	a0,a3
   13974:	03e12223          	sw	t5,36(sp)
   13978:	03d12023          	sw	t4,32(sp)
   1397c:	01112e23          	sw	a7,28(sp)
   13980:	00e12c23          	sw	a4,24(sp)
   13984:	00d12823          	sw	a3,16(sp)
   13988:	675020ef          	jal	167fc <strncpy>
   1398c:	012d37b3          	sltu	a5,s10,s2
   13990:	00500613          	li	a2,5
   13994:	00fd07b3          	add	a5,s10,a5
   13998:	02c7f7b3          	remu	a5,a5,a2
   1399c:	01812703          	lw	a4,24(sp)
   139a0:	ccccd337          	lui	t1,0xccccd
   139a4:	ccccd537          	lui	a0,0xccccd
   139a8:	00174583          	lbu	a1,1(a4)
   139ac:	ccd30313          	addi	t1,t1,-819 # cccccccd <__BSS_END__+0xccca9ecd>
   139b0:	ccc50513          	addi	a0,a0,-820 # cccccccc <__BSS_END__+0xccca9ecc>
   139b4:	00b035b3          	snez	a1,a1
   139b8:	00b70733          	add	a4,a4,a1
   139bc:	01012683          	lw	a3,16(sp)
   139c0:	01c12883          	lw	a7,28(sp)
   139c4:	02012e83          	lw	t4,32(sp)
   139c8:	02412f03          	lw	t5,36(sp)
   139cc:	03412803          	lw	a6,52(sp)
   139d0:	00000613          	li	a2,0
   139d4:	00500e13          	li	t3,5
   139d8:	40f907b3          	sub	a5,s2,a5
   139dc:	00f935b3          	sltu	a1,s2,a5
   139e0:	40bd85b3          	sub	a1,s11,a1
   139e4:	02a78533          	mul	a0,a5,a0
   139e8:	026585b3          	mul	a1,a1,t1
   139ec:	0267bfb3          	mulhu	t6,a5,t1
   139f0:	00a585b3          	add	a1,a1,a0
   139f4:	02678533          	mul	a0,a5,t1
   139f8:	01f585b3          	add	a1,a1,t6
   139fc:	01f59793          	slli	a5,a1,0x1f
   13a00:	0015d593          	srli	a1,a1,0x1
   13a04:	00155513          	srli	a0,a0,0x1
   13a08:	00a7e7b3          	or	a5,a5,a0
   13a0c:	a05ff06f          	j	13410 <_vfprintf_r+0x184c>
   13a10:	00812503          	lw	a0,8(sp)
   13a14:	0c410613          	addi	a2,sp,196
   13a18:	00048593          	mv	a1,s1
   13a1c:	495000ef          	jal	146b0 <__sprint_r>
   13a20:	00050463          	beqz	a0,13a28 <_vfprintf_r+0x1e64>
   13a24:	828ff06f          	j	12a4c <_vfprintf_r+0xe88>
   13a28:	0ac12583          	lw	a1,172(sp)
   13a2c:	0cc12603          	lw	a2,204(sp)
   13a30:	00098a13          	mv	s4,s3
   13a34:	aadff06f          	j	134e0 <_vfprintf_r+0x191c>
   13a38:	00812503          	lw	a0,8(sp)
   13a3c:	0c410613          	addi	a2,sp,196
   13a40:	00048593          	mv	a1,s1
   13a44:	46d000ef          	jal	146b0 <__sprint_r>
   13a48:	00050463          	beqz	a0,13a50 <_vfprintf_r+0x1e8c>
   13a4c:	800ff06f          	j	12a4c <_vfprintf_r+0xe88>
   13a50:	0cc12603          	lw	a2,204(sp)
   13a54:	00098a13          	mv	s4,s3
   13a58:	a44ff06f          	j	12c9c <_vfprintf_r+0x10d8>
   13a5c:	00600913          	li	s2,6
   13a60:	df8ff06f          	j	13058 <_vfprintf_r+0x1494>
   13a64:	0ff97913          	zext.b	s2,s2
   13a68:	00000d93          	li	s11,0
   13a6c:	e2cfe06f          	j	12098 <_vfprintf_r+0x4d4>
   13a70:	01891913          	slli	s2,s2,0x18
   13a74:	41895913          	srai	s2,s2,0x18
   13a78:	41f95d93          	srai	s11,s2,0x1f
   13a7c:	000d8793          	mv	a5,s11
   13a80:	d9cfe06f          	j	1201c <_vfprintf_r+0x458>
   13a84:	0ff97913          	zext.b	s2,s2
   13a88:	00000d93          	li	s11,0
   13a8c:	d3cff06f          	j	12fc8 <_vfprintf_r+0x1404>
   13a90:	01412783          	lw	a5,20(sp)
   13a94:	00c12703          	lw	a4,12(sp)
   13a98:	0007a783          	lw	a5,0(a5)
   13a9c:	00e79023          	sh	a4,0(a5)
   13aa0:	b55fe06f          	j	125f4 <_vfprintf_r+0xa30>
   13aa4:	00812503          	lw	a0,8(sp)
   13aa8:	0c410613          	addi	a2,sp,196
   13aac:	00048593          	mv	a1,s1
   13ab0:	401000ef          	jal	146b0 <__sprint_r>
   13ab4:	c54fe06f          	j	11f08 <_vfprintf_r+0x344>
   13ab8:	02d00793          	li	a5,45
   13abc:	0af103a3          	sb	a5,167(sp)
   13ac0:	02d00713          	li	a4,45
   13ac4:	c21ff06f          	j	136e4 <_vfprintf_r+0x1b20>
   13ac8:	00812503          	lw	a0,8(sp)
   13acc:	0c410613          	addi	a2,sp,196
   13ad0:	00048593          	mv	a1,s1
   13ad4:	3dd000ef          	jal	146b0 <__sprint_r>
   13ad8:	00050463          	beqz	a0,13ae0 <_vfprintf_r+0x1f1c>
   13adc:	f71fe06f          	j	12a4c <_vfprintf_r+0xe88>
   13ae0:	0ac12583          	lw	a1,172(sp)
   13ae4:	0cc12603          	lw	a2,204(sp)
   13ae8:	0c812703          	lw	a4,200(sp)
   13aec:	00098a13          	mv	s4,s3
   13af0:	b4dff06f          	j	1363c <_vfprintf_r+0x1a78>
   13af4:	01650733          	add	a4,a0,s6
   13af8:	04700613          	li	a2,71
   13afc:	08010593          	addi	a1,sp,128
   13b00:	00090513          	mv	a0,s2
   13b04:	02e12023          	sw	a4,32(sp)
   13b08:	00c12e23          	sw	a2,28(sp)
   13b0c:	01112c23          	sw	a7,24(sp)
   13b10:	09f12823          	sw	t6,144(sp)
   13b14:	09e12a23          	sw	t5,148(sp)
   13b18:	09d12c23          	sw	t4,152(sp)
   13b1c:	09812e23          	sw	s8,156(sp)
   13b20:	08012023          	sw	zero,128(sp)
   13b24:	08012223          	sw	zero,132(sp)
   13b28:	08012423          	sw	zero,136(sp)
   13b2c:	08012623          	sw	zero,140(sp)
   13b30:	769090ef          	jal	1da98 <__eqtf2>
   13b34:	01812883          	lw	a7,24(sp)
   13b38:	01c12603          	lw	a2,28(sp)
   13b3c:	02012703          	lw	a4,32(sp)
   13b40:	58050c63          	beqz	a0,140d8 <_vfprintf_r+0x2514>
   13b44:	0bc12783          	lw	a5,188(sp)
   13b48:	00e7fe63          	bgeu	a5,a4,13b64 <_vfprintf_r+0x1fa0>
   13b4c:	03000593          	li	a1,48
   13b50:	00178693          	addi	a3,a5,1
   13b54:	0ad12e23          	sw	a3,188(sp)
   13b58:	00b78023          	sb	a1,0(a5)
   13b5c:	0bc12783          	lw	a5,188(sp)
   13b60:	fee7e8e3          	bltu	a5,a4,13b50 <_vfprintf_r+0x1f8c>
   13b64:	0ac12703          	lw	a4,172(sp)
   13b68:	00e12c23          	sw	a4,24(sp)
   13b6c:	41a787b3          	sub	a5,a5,s10
   13b70:	04700713          	li	a4,71
   13b74:	00f12e23          	sw	a5,28(sp)
   13b78:	e6e60e63          	beq	a2,a4,131f4 <_vfprintf_r+0x1630>
   13b7c:	04600713          	li	a4,70
   13b80:	6ee60263          	beq	a2,a4,14264 <_vfprintf_r+0x26a0>
   13b84:	01812783          	lw	a5,24(sp)
   13b88:	fff78713          	addi	a4,a5,-1
   13b8c:	e84ff06f          	j	13210 <_vfprintf_r+0x164c>
   13b90:	001b0693          	addi	a3,s6,1
   13b94:	00200613          	li	a2,2
   13b98:	00812503          	lw	a0,8(sp)
   13b9c:	09010913          	addi	s2,sp,144
   13ba0:	0ac10713          	addi	a4,sp,172
   13ba4:	00090593          	mv	a1,s2
   13ba8:	0bc10813          	addi	a6,sp,188
   13bac:	0b010793          	addi	a5,sp,176
   13bb0:	05112623          	sw	a7,76(sp)
   13bb4:	02d12223          	sw	a3,36(sp)
   13bb8:	09f12823          	sw	t6,144(sp)
   13bbc:	03f12023          	sw	t6,32(sp)
   13bc0:	09e12a23          	sw	t5,148(sp)
   13bc4:	01e12e23          	sw	t5,28(sp)
   13bc8:	09d12c23          	sw	t4,152(sp)
   13bcc:	01d12c23          	sw	t4,24(sp)
   13bd0:	09812e23          	sw	s8,156(sp)
   13bd4:	4f4030ef          	jal	170c8 <_ldtoa_r>
   13bd8:	04c12883          	lw	a7,76(sp)
   13bdc:	02412683          	lw	a3,36(sp)
   13be0:	04600593          	li	a1,70
   13be4:	fdf8f613          	andi	a2,a7,-33
   13be8:	01812e83          	lw	t4,24(sp)
   13bec:	01c12f03          	lw	t5,28(sp)
   13bf0:	02012f83          	lw	t6,32(sp)
   13bf4:	00050d13          	mv	s10,a0
   13bf8:	00d50733          	add	a4,a0,a3
   13bfc:	08b61ae3          	bne	a2,a1,14490 <_vfprintf_r+0x28cc>
   13c00:	000d4503          	lbu	a0,0(s10)
   13c04:	03000593          	li	a1,48
   13c08:	5cb50c63          	beq	a0,a1,141e0 <_vfprintf_r+0x261c>
   13c0c:	0ac12683          	lw	a3,172(sp)
   13c10:	08010593          	addi	a1,sp,128
   13c14:	00d70733          	add	a4,a4,a3
   13c18:	ee9ff06f          	j	13b00 <_vfprintf_r+0x1f3c>
   13c1c:	09010913          	addi	s2,sp,144
   13c20:	08010593          	addi	a1,sp,128
   13c24:	0ac10613          	addi	a2,sp,172
   13c28:	00090513          	mv	a0,s2
   13c2c:	03112e23          	sw	a7,60(sp)
   13c30:	09f12023          	sw	t6,128(sp)
   13c34:	09e12223          	sw	t5,132(sp)
   13c38:	09d12423          	sw	t4,136(sp)
   13c3c:	00b12c23          	sw	a1,24(sp)
   13c40:	09812623          	sw	s8,140(sp)
   13c44:	254030ef          	jal	16e98 <frexpl>
   13c48:	09012803          	lw	a6,144(sp)
   13c4c:	0000d717          	auipc	a4,0xd
   13c50:	04470713          	addi	a4,a4,68 # 20c90 <blanks.1+0x10>
   13c54:	00072503          	lw	a0,0(a4)
   13c58:	09012023          	sw	a6,128(sp)
   13c5c:	09412803          	lw	a6,148(sp)
   13c60:	00472603          	lw	a2,4(a4)
   13c64:	00872683          	lw	a3,8(a4)
   13c68:	09012223          	sw	a6,132(sp)
   13c6c:	09812803          	lw	a6,152(sp)
   13c70:	00c72703          	lw	a4,12(a4)
   13c74:	01812583          	lw	a1,24(sp)
   13c78:	09012423          	sw	a6,136(sp)
   13c7c:	09c12803          	lw	a6,156(sp)
   13c80:	06a12823          	sw	a0,112(sp)
   13c84:	06c12a23          	sw	a2,116(sp)
   13c88:	00090513          	mv	a0,s2
   13c8c:	07010613          	addi	a2,sp,112
   13c90:	09012623          	sw	a6,140(sp)
   13c94:	06d12c23          	sw	a3,120(sp)
   13c98:	06e12e23          	sw	a4,124(sp)
   13c9c:	1280a0ef          	jal	1ddc4 <__multf3>
   13ca0:	01812583          	lw	a1,24(sp)
   13ca4:	09012f03          	lw	t5,144(sp)
   13ca8:	09412e83          	lw	t4,148(sp)
   13cac:	09812803          	lw	a6,152(sp)
   13cb0:	00090513          	mv	a0,s2
   13cb4:	02b12223          	sw	a1,36(sp)
   13cb8:	03e12023          	sw	t5,32(sp)
   13cbc:	01d12e23          	sw	t4,28(sp)
   13cc0:	01012c23          	sw	a6,24(sp)
   13cc4:	08012023          	sw	zero,128(sp)
   13cc8:	08012223          	sw	zero,132(sp)
   13ccc:	08012423          	sw	zero,136(sp)
   13cd0:	08012623          	sw	zero,140(sp)
   13cd4:	5c5090ef          	jal	1da98 <__eqtf2>
   13cd8:	09c12d83          	lw	s11,156(sp)
   13cdc:	01812803          	lw	a6,24(sp)
   13ce0:	01c12e83          	lw	t4,28(sp)
   13ce4:	02012f03          	lw	t5,32(sp)
   13ce8:	02412583          	lw	a1,36(sp)
   13cec:	03c12883          	lw	a7,60(sp)
   13cf0:	00051663          	bnez	a0,13cfc <_vfprintf_r+0x2138>
   13cf4:	00100713          	li	a4,1
   13cf8:	0ae12623          	sw	a4,172(sp)
   13cfc:	0000d797          	auipc	a5,0xd
   13d00:	c4478793          	addi	a5,a5,-956 # 20940 <_exit+0x108>
   13d04:	06100713          	li	a4,97
   13d08:	00f12c23          	sw	a5,24(sp)
   13d0c:	00e89863          	bne	a7,a4,13d1c <_vfprintf_r+0x2158>
   13d10:	0000d797          	auipc	a5,0xd
   13d14:	c1c78793          	addi	a5,a5,-996 # 2092c <_exit+0xf4>
   13d18:	00f12c23          	sw	a5,24(sp)
   13d1c:	0000d717          	auipc	a4,0xd
   13d20:	f8470713          	addi	a4,a4,-124 # 20ca0 <blanks.1+0x20>
   13d24:	00072783          	lw	a5,0(a4)
   13d28:	03512e23          	sw	s5,60(sp)
   13d2c:	05712823          	sw	s7,80(sp)
   13d30:	00f12e23          	sw	a5,28(sp)
   13d34:	00472783          	lw	a5,4(a4)
   13d38:	05412a23          	sw	s4,84(sp)
   13d3c:	000d0a93          	mv	s5,s10
   13d40:	02f12023          	sw	a5,32(sp)
   13d44:	00872783          	lw	a5,8(a4)
   13d48:	04912c23          	sw	s1,88(sp)
   13d4c:	05a12e23          	sw	s10,92(sp)
   13d50:	02f12223          	sw	a5,36(sp)
   13d54:	00c72783          	lw	a5,12(a4)
   13d58:	000d8c13          	mv	s8,s11
   13d5c:	fffb0b13          	addi	s6,s6,-1
   13d60:	05112423          	sw	a7,72(sp)
   13d64:	05912623          	sw	s9,76(sp)
   13d68:	00078d13          	mv	s10,a5
   13d6c:	000e8a13          	mv	s4,t4
   13d70:	00080b93          	mv	s7,a6
   13d74:	000f0d93          	mv	s11,t5
   13d78:	00058493          	mv	s1,a1
   13d7c:	03c0006f          	j	13db8 <_vfprintf_r+0x21f4>
   13d80:	00048593          	mv	a1,s1
   13d84:	00090513          	mv	a0,s2
   13d88:	09b12823          	sw	s11,144(sp)
   13d8c:	09412a23          	sw	s4,148(sp)
   13d90:	09712c23          	sw	s7,152(sp)
   13d94:	09812e23          	sw	s8,156(sp)
   13d98:	08012023          	sw	zero,128(sp)
   13d9c:	08012223          	sw	zero,132(sp)
   13da0:	08012423          	sw	zero,136(sp)
   13da4:	08012623          	sw	zero,140(sp)
   13da8:	fffb0c93          	addi	s9,s6,-1
   13dac:	4ed090ef          	jal	1da98 <__eqtf2>
   13db0:	5e050863          	beqz	a0,143a0 <_vfprintf_r+0x27dc>
   13db4:	000c8b13          	mv	s6,s9
   13db8:	01c12783          	lw	a5,28(sp)
   13dbc:	07010613          	addi	a2,sp,112
   13dc0:	00048593          	mv	a1,s1
   13dc4:	06f12823          	sw	a5,112(sp)
   13dc8:	02012783          	lw	a5,32(sp)
   13dcc:	00090513          	mv	a0,s2
   13dd0:	09b12023          	sw	s11,128(sp)
   13dd4:	06f12a23          	sw	a5,116(sp)
   13dd8:	02412783          	lw	a5,36(sp)
   13ddc:	09412223          	sw	s4,132(sp)
   13de0:	09712423          	sw	s7,136(sp)
   13de4:	06f12c23          	sw	a5,120(sp)
   13de8:	09812623          	sw	s8,140(sp)
   13dec:	07a12e23          	sw	s10,124(sp)
   13df0:	7d5090ef          	jal	1ddc4 <__multf3>
   13df4:	00090513          	mv	a0,s2
   13df8:	4ec0c0ef          	jal	202e4 <__fixtfsi>
   13dfc:	00050593          	mv	a1,a0
   13e00:	00050c93          	mv	s9,a0
   13e04:	00090513          	mv	a0,s2
   13e08:	09012d83          	lw	s11,144(sp)
   13e0c:	09412c03          	lw	s8,148(sp)
   13e10:	09812b83          	lw	s7,152(sp)
   13e14:	09c12a03          	lw	s4,156(sp)
   13e18:	5c40c0ef          	jal	203dc <__floatsitf>
   13e1c:	09012683          	lw	a3,144(sp)
   13e20:	06010613          	addi	a2,sp,96
   13e24:	07010593          	addi	a1,sp,112
   13e28:	06d12023          	sw	a3,96(sp)
   13e2c:	09412683          	lw	a3,148(sp)
   13e30:	00048513          	mv	a0,s1
   13e34:	07b12823          	sw	s11,112(sp)
   13e38:	06d12223          	sw	a3,100(sp)
   13e3c:	09812683          	lw	a3,152(sp)
   13e40:	07812a23          	sw	s8,116(sp)
   13e44:	07712c23          	sw	s7,120(sp)
   13e48:	06d12423          	sw	a3,104(sp)
   13e4c:	09c12683          	lw	a3,156(sp)
   13e50:	07412e23          	sw	s4,124(sp)
   13e54:	06d12623          	sw	a3,108(sp)
   13e58:	7710a0ef          	jal	1edc8 <__subtf3>
   13e5c:	01812783          	lw	a5,24(sp)
   13e60:	000a8f93          	mv	t6,s5
   13e64:	001a8a93          	addi	s5,s5,1
   13e68:	019786b3          	add	a3,a5,s9
   13e6c:	0006c603          	lbu	a2,0(a3)
   13e70:	08012d83          	lw	s11,128(sp)
   13e74:	08412a03          	lw	s4,132(sp)
   13e78:	08812b83          	lw	s7,136(sp)
   13e7c:	08c12c03          	lw	s8,140(sp)
   13e80:	fff00793          	li	a5,-1
   13e84:	feca8fa3          	sb	a2,-1(s5)
   13e88:	eefb1ce3          	bne	s6,a5,13d80 <_vfprintf_r+0x21bc>
   13e8c:	0000d517          	auipc	a0,0xd
   13e90:	e2450513          	addi	a0,a0,-476 # 20cb0 <blanks.1+0x30>
   13e94:	04812883          	lw	a7,72(sp)
   13e98:	00452283          	lw	t0,4(a0)
   13e9c:	00852383          	lw	t2,8(a0)
   13ea0:	00c52783          	lw	a5,12(a0)
   13ea4:	00052b03          	lw	s6,0(a0)
   13ea8:	000d8f13          	mv	t5,s11
   13eac:	000a0e93          	mv	t4,s4
   13eb0:	000b8813          	mv	a6,s7
   13eb4:	00048593          	mv	a1,s1
   13eb8:	000c0d93          	mv	s11,s8
   13ebc:	05c12d03          	lw	s10,92(sp)
   13ec0:	05412a03          	lw	s4,84(sp)
   13ec4:	05012b83          	lw	s7,80(sp)
   13ec8:	05812483          	lw	s1,88(sp)
   13ecc:	03f12023          	sw	t6,32(sp)
   13ed0:	01112e23          	sw	a7,28(sp)
   13ed4:	05912e23          	sw	s9,92(sp)
   13ed8:	000a8c13          	mv	s8,s5
   13edc:	04c12c83          	lw	s9,76(sp)
   13ee0:	03c12a83          	lw	s5,60(sp)
   13ee4:	09e12823          	sw	t5,144(sp)
   13ee8:	05e12c23          	sw	t5,88(sp)
   13eec:	09d12a23          	sw	t4,148(sp)
   13ef0:	05d12a23          	sw	t4,84(sp)
   13ef4:	09012c23          	sw	a6,152(sp)
   13ef8:	05012823          	sw	a6,80(sp)
   13efc:	09b12e23          	sw	s11,156(sp)
   13f00:	09612023          	sw	s6,128(sp)
   13f04:	08512223          	sw	t0,132(sp)
   13f08:	04512623          	sw	t0,76(sp)
   13f0c:	08712423          	sw	t2,136(sp)
   13f10:	04712423          	sw	t2,72(sp)
   13f14:	08f12623          	sw	a5,140(sp)
   13f18:	02f12e23          	sw	a5,60(sp)
   13f1c:	02b12223          	sw	a1,36(sp)
   13f20:	00090513          	mv	a0,s2
   13f24:	441090ef          	jal	1db64 <__getf2>
   13f28:	01c12883          	lw	a7,28(sp)
   13f2c:	02012f83          	lw	t6,32(sp)
   13f30:	02a04863          	bgtz	a0,13f60 <_vfprintf_r+0x239c>
   13f34:	02412583          	lw	a1,36(sp)
   13f38:	00090513          	mv	a0,s2
   13f3c:	03112023          	sw	a7,32(sp)
   13f40:	01f12e23          	sw	t6,28(sp)
   13f44:	355090ef          	jal	1da98 <__eqtf2>
   13f48:	02012883          	lw	a7,32(sp)
   13f4c:	04051e63          	bnez	a0,13fa8 <_vfprintf_r+0x23e4>
   13f50:	05c12703          	lw	a4,92(sp)
   13f54:	01c12f83          	lw	t6,28(sp)
   13f58:	00177693          	andi	a3,a4,1
   13f5c:	04068663          	beqz	a3,13fa8 <_vfprintf_r+0x23e4>
   13f60:	01812783          	lw	a5,24(sp)
   13f64:	0bf12e23          	sw	t6,188(sp)
   13f68:	fffc4603          	lbu	a2,-1(s8)
   13f6c:	00f7c583          	lbu	a1,15(a5)
   13f70:	000c0693          	mv	a3,s8
   13f74:	02b61063          	bne	a2,a1,13f94 <_vfprintf_r+0x23d0>
   13f78:	03000513          	li	a0,48
   13f7c:	fea68fa3          	sb	a0,-1(a3)
   13f80:	0bc12683          	lw	a3,188(sp)
   13f84:	fff68793          	addi	a5,a3,-1
   13f88:	0af12e23          	sw	a5,188(sp)
   13f8c:	fff6c603          	lbu	a2,-1(a3)
   13f90:	fec586e3          	beq	a1,a2,13f7c <_vfprintf_r+0x23b8>
   13f94:	00160593          	addi	a1,a2,1
   13f98:	03900513          	li	a0,57
   13f9c:	0ff5f593          	zext.b	a1,a1
   13fa0:	04a60463          	beq	a2,a0,13fe8 <_vfprintf_r+0x2424>
   13fa4:	feb68fa3          	sb	a1,-1(a3)
   13fa8:	0ac12783          	lw	a5,172(sp)
   13fac:	41ac0733          	sub	a4,s8,s10
   13fb0:	00e12e23          	sw	a4,28(sp)
   13fb4:	fff78713          	addi	a4,a5,-1
   13fb8:	00f12c23          	sw	a5,24(sp)
   13fbc:	06100613          	li	a2,97
   13fc0:	0ae12623          	sw	a4,172(sp)
   13fc4:	07000693          	li	a3,112
   13fc8:	00c88663          	beq	a7,a2,13fd4 <_vfprintf_r+0x2410>
   13fcc:	05000693          	li	a3,80
   13fd0:	04100893          	li	a7,65
   13fd4:	00100613          	li	a2,1
   13fd8:	a44ff06f          	j	1321c <_vfprintf_r+0x1658>
   13fdc:	000b0693          	mv	a3,s6
   13fe0:	00300613          	li	a2,3
   13fe4:	bb5ff06f          	j	13b98 <_vfprintf_r+0x1fd4>
   13fe8:	01812783          	lw	a5,24(sp)
   13fec:	00a7c583          	lbu	a1,10(a5)
   13ff0:	fb5ff06f          	j	13fa4 <_vfprintf_r+0x23e0>
   13ff4:	0a714703          	lbu	a4,167(sp)
   13ff8:	04812c83          	lw	s9,72(sp)
   13ffc:	00000b13          	li	s6,0
   14000:	00070463          	beqz	a4,14008 <_vfprintf_r+0x2444>
   14004:	864fe06f          	j	12068 <_vfprintf_r+0x4a4>
   14008:	dd5fd06f          	j	11ddc <_vfprintf_r+0x218>
   1400c:	01c12783          	lw	a5,28(sp)
   14010:	01812703          	lw	a4,24(sp)
   14014:	10f74263          	blt	a4,a5,14118 <_vfprintf_r+0x2554>
   14018:	01812783          	lw	a5,24(sp)
   1401c:	001cf713          	andi	a4,s9,1
   14020:	00078913          	mv	s2,a5
   14024:	00070663          	beqz	a4,14030 <_vfprintf_r+0x246c>
   14028:	02812703          	lw	a4,40(sp)
   1402c:	00e78933          	add	s2,a5,a4
   14030:	400cfe13          	andi	t3,s9,1024
   14034:	000e0663          	beqz	t3,14040 <_vfprintf_r+0x247c>
   14038:	01812783          	lw	a5,24(sp)
   1403c:	2ef04263          	bgtz	a5,14320 <_vfprintf_r+0x275c>
   14040:	fff94693          	not	a3,s2
   14044:	41f6d693          	srai	a3,a3,0x1f
   14048:	00d97db3          	and	s11,s2,a3
   1404c:	06700893          	li	a7,103
   14050:	02012223          	sw	zero,36(sp)
   14054:	02012023          	sw	zero,32(sp)
   14058:	aa0ff06f          	j	132f8 <_vfprintf_r+0x1734>
   1405c:	0c410613          	addi	a2,sp,196
   14060:	00048593          	mv	a1,s1
   14064:	00090513          	mv	a0,s2
   14068:	648000ef          	jal	146b0 <__sprint_r>
   1406c:	00050463          	beqz	a0,14074 <_vfprintf_r+0x24b0>
   14070:	9ddfe06f          	j	12a4c <_vfprintf_r+0xe88>
   14074:	000cc683          	lbu	a3,0(s9)
   14078:	0cc12603          	lw	a2,204(sp)
   1407c:	00098593          	mv	a1,s3
   14080:	01000713          	li	a4,16
   14084:	00700813          	li	a6,7
   14088:	00dd0d33          	add	s10,s10,a3
   1408c:	f94ff06f          	j	13820 <_vfprintf_r+0x1c5c>
   14090:	00900513          	li	a0,9
   14094:	b7256e63          	bltu	a0,s2,13410 <_vfprintf_r+0x184c>
   14098:	bf8ff06f          	j	13490 <_vfprintf_r+0x18cc>
   1409c:	02d00793          	li	a5,45
   140a0:	0af103a3          	sb	a5,167(sp)
   140a4:	02d00713          	li	a4,45
   140a8:	d0cff06f          	j	135b4 <_vfprintf_r+0x19f0>
   140ac:	0a714703          	lbu	a4,167(sp)
   140b0:	01812a23          	sw	s8,20(sp)
   140b4:	02012223          	sw	zero,36(sp)
   140b8:	02012023          	sw	zero,32(sp)
   140bc:	00012c23          	sw	zero,24(sp)
   140c0:	000b0d93          	mv	s11,s6
   140c4:	000b0913          	mv	s2,s6
   140c8:	00000b13          	li	s6,0
   140cc:	00070463          	beqz	a4,140d4 <_vfprintf_r+0x2510>
   140d0:	f99fd06f          	j	12068 <_vfprintf_r+0x4a4>
   140d4:	d09fd06f          	j	11ddc <_vfprintf_r+0x218>
   140d8:	0ac12783          	lw	a5,172(sp)
   140dc:	00f12c23          	sw	a5,24(sp)
   140e0:	00070793          	mv	a5,a4
   140e4:	a89ff06f          	j	13b6c <_vfprintf_r+0x1fa8>
   140e8:	00812503          	lw	a0,8(sp)
   140ec:	0c410613          	addi	a2,sp,196
   140f0:	00048593          	mv	a1,s1
   140f4:	5bc000ef          	jal	146b0 <__sprint_r>
   140f8:	00050463          	beqz	a0,14100 <_vfprintf_r+0x253c>
   140fc:	951fe06f          	j	12a4c <_vfprintf_r+0xe88>
   14100:	0ac12583          	lw	a1,172(sp)
   14104:	01c12783          	lw	a5,28(sp)
   14108:	0cc12603          	lw	a2,204(sp)
   1410c:	00098a13          	mv	s4,s3
   14110:	40b785b3          	sub	a1,a5,a1
   14114:	c0dfe06f          	j	12d20 <_vfprintf_r+0x115c>
   14118:	01c12783          	lw	a5,28(sp)
   1411c:	02812703          	lw	a4,40(sp)
   14120:	06700893          	li	a7,103
   14124:	00e78933          	add	s2,a5,a4
   14128:	01812783          	lw	a5,24(sp)
   1412c:	2af05a63          	blez	a5,143e0 <_vfprintf_r+0x281c>
   14130:	400cfe13          	andi	t3,s9,1024
   14134:	1e0e1863          	bnez	t3,14324 <_vfprintf_r+0x2760>
   14138:	fff94693          	not	a3,s2
   1413c:	41f6d693          	srai	a3,a3,0x1f
   14140:	00d97db3          	and	s11,s2,a3
   14144:	f0dff06f          	j	14050 <_vfprintf_r+0x248c>
   14148:	01012383          	lw	t2,16(sp)
   1414c:	000a0493          	mv	s1,s4
   14150:	901fe06f          	j	12a50 <_vfprintf_r+0xe8c>
   14154:	000c8e93          	mv	t4,s9
   14158:	bc4fe06f          	j	1251c <_vfprintf_r+0x958>
   1415c:	ff000513          	li	a0,-16
   14160:	40b00933          	neg	s2,a1
   14164:	0000d817          	auipc	a6,0xd
   14168:	b0c80813          	addi	a6,a6,-1268 # 20c70 <zeroes.0>
   1416c:	12a5d063          	bge	a1,a0,1428c <_vfprintf_r+0x26c8>
   14170:	01512c23          	sw	s5,24(sp)
   14174:	01000b13          	li	s6,16
   14178:	00700c13          	li	s8,7
   1417c:	00080a93          	mv	s5,a6
   14180:	00c0006f          	j	1418c <_vfprintf_r+0x25c8>
   14184:	ff090913          	addi	s2,s2,-16
   14188:	0f2b5e63          	bge	s6,s2,14284 <_vfprintf_r+0x26c0>
   1418c:	01060613          	addi	a2,a2,16
   14190:	00170713          	addi	a4,a4,1
   14194:	015a2023          	sw	s5,0(s4)
   14198:	016a2223          	sw	s6,4(s4)
   1419c:	0cc12623          	sw	a2,204(sp)
   141a0:	0ce12423          	sw	a4,200(sp)
   141a4:	008a0a13          	addi	s4,s4,8
   141a8:	fcec5ee3          	bge	s8,a4,14184 <_vfprintf_r+0x25c0>
   141ac:	00812503          	lw	a0,8(sp)
   141b0:	0c410613          	addi	a2,sp,196
   141b4:	00048593          	mv	a1,s1
   141b8:	4f8000ef          	jal	146b0 <__sprint_r>
   141bc:	00050463          	beqz	a0,141c4 <_vfprintf_r+0x2600>
   141c0:	88dfe06f          	j	12a4c <_vfprintf_r+0xe88>
   141c4:	0cc12603          	lw	a2,204(sp)
   141c8:	0c812703          	lw	a4,200(sp)
   141cc:	00098a13          	mv	s4,s3
   141d0:	fb5ff06f          	j	14184 <_vfprintf_r+0x25c0>
   141d4:	fff00793          	li	a5,-1
   141d8:	00f12623          	sw	a5,12(sp)
   141dc:	d61fd06f          	j	11f3c <_vfprintf_r+0x378>
   141e0:	08010593          	addi	a1,sp,128
   141e4:	00090513          	mv	a0,s2
   141e8:	04e12c23          	sw	a4,88(sp)
   141ec:	04c12a23          	sw	a2,84(sp)
   141f0:	05112823          	sw	a7,80(sp)
   141f4:	04d12623          	sw	a3,76(sp)
   141f8:	09f12823          	sw	t6,144(sp)
   141fc:	03f12223          	sw	t6,36(sp)
   14200:	09e12a23          	sw	t5,148(sp)
   14204:	03e12023          	sw	t5,32(sp)
   14208:	09d12c23          	sw	t4,152(sp)
   1420c:	01d12e23          	sw	t4,28(sp)
   14210:	00b12c23          	sw	a1,24(sp)
   14214:	09812e23          	sw	s8,156(sp)
   14218:	08012023          	sw	zero,128(sp)
   1421c:	08012223          	sw	zero,132(sp)
   14220:	08012423          	sw	zero,136(sp)
   14224:	08012623          	sw	zero,140(sp)
   14228:	071090ef          	jal	1da98 <__eqtf2>
   1422c:	01812583          	lw	a1,24(sp)
   14230:	01c12e83          	lw	t4,28(sp)
   14234:	02012f03          	lw	t5,32(sp)
   14238:	02412f83          	lw	t6,36(sp)
   1423c:	04c12683          	lw	a3,76(sp)
   14240:	05012883          	lw	a7,80(sp)
   14244:	05412603          	lw	a2,84(sp)
   14248:	05812703          	lw	a4,88(sp)
   1424c:	1c051063          	bnez	a0,1440c <_vfprintf_r+0x2848>
   14250:	0ac12783          	lw	a5,172(sp)
   14254:	00f70733          	add	a4,a4,a5
   14258:	00f12c23          	sw	a5,24(sp)
   1425c:	41a707b3          	sub	a5,a4,s10
   14260:	00f12e23          	sw	a5,28(sp)
   14264:	01812783          	lw	a5,24(sp)
   14268:	001cf713          	andi	a4,s9,1
   1426c:	01676733          	or	a4,a4,s6
   14270:	1af05863          	blez	a5,14420 <_vfprintf_r+0x285c>
   14274:	18071263          	bnez	a4,143f8 <_vfprintf_r+0x2834>
   14278:	01812903          	lw	s2,24(sp)
   1427c:	06600893          	li	a7,102
   14280:	eb1ff06f          	j	14130 <_vfprintf_r+0x256c>
   14284:	000a8813          	mv	a6,s5
   14288:	01812a83          	lw	s5,24(sp)
   1428c:	01260633          	add	a2,a2,s2
   14290:	00170713          	addi	a4,a4,1
   14294:	010a2023          	sw	a6,0(s4)
   14298:	012a2223          	sw	s2,4(s4)
   1429c:	0cc12623          	sw	a2,204(sp)
   142a0:	0ce12423          	sw	a4,200(sp)
   142a4:	00700593          	li	a1,7
   142a8:	a6e5de63          	bge	a1,a4,13524 <_vfprintf_r+0x1960>
   142ac:	00812503          	lw	a0,8(sp)
   142b0:	0c410613          	addi	a2,sp,196
   142b4:	00048593          	mv	a1,s1
   142b8:	3f8000ef          	jal	146b0 <__sprint_r>
   142bc:	00050463          	beqz	a0,142c4 <_vfprintf_r+0x2700>
   142c0:	f8cfe06f          	j	12a4c <_vfprintf_r+0xe88>
   142c4:	0cc12603          	lw	a2,204(sp)
   142c8:	0c812703          	lw	a4,200(sp)
   142cc:	00098a13          	mv	s4,s3
   142d0:	b70ff06f          	j	13640 <_vfprintf_r+0x1a7c>
   142d4:	0b610693          	addi	a3,sp,182
   142d8:	00061863          	bnez	a2,142e8 <_vfprintf_r+0x2724>
   142dc:	03000693          	li	a3,48
   142e0:	0ad10b23          	sb	a3,182(sp)
   142e4:	0b710693          	addi	a3,sp,183
   142e8:	19010793          	addi	a5,sp,400
   142ec:	40f68633          	sub	a2,a3,a5
   142f0:	03070713          	addi	a4,a4,48
   142f4:	0dd60793          	addi	a5,a2,221
   142f8:	00e68023          	sb	a4,0(a3)
   142fc:	02f12e23          	sw	a5,60(sp)
   14300:	fb5fe06f          	j	132b4 <_vfprintf_r+0x16f0>
   14304:	001cf713          	andi	a4,s9,1
   14308:	00071463          	bnez	a4,14310 <_vfprintf_r+0x274c>
   1430c:	fc9fe06f          	j	132d4 <_vfprintf_r+0x1710>
   14310:	fbdfe06f          	j	132cc <_vfprintf_r+0x1708>
   14314:	00012823          	sw	zero,16(sp)
   14318:	00600b13          	li	s6,6
   1431c:	ea9fd06f          	j	121c4 <_vfprintf_r+0x600>
   14320:	06700893          	li	a7,103
   14324:	03812603          	lw	a2,56(sp)
   14328:	0ff00693          	li	a3,255
   1432c:	00064703          	lbu	a4,0(a2)
   14330:	14d70a63          	beq	a4,a3,14484 <_vfprintf_r+0x28c0>
   14334:	01812783          	lw	a5,24(sp)
   14338:	00000513          	li	a0,0
   1433c:	00000593          	li	a1,0
   14340:	00f75e63          	bge	a4,a5,1435c <_vfprintf_r+0x2798>
   14344:	40e787b3          	sub	a5,a5,a4
   14348:	00164703          	lbu	a4,1(a2)
   1434c:	04070463          	beqz	a4,14394 <_vfprintf_r+0x27d0>
   14350:	00158593          	addi	a1,a1,1
   14354:	00160613          	addi	a2,a2,1
   14358:	fed714e3          	bne	a4,a3,14340 <_vfprintf_r+0x277c>
   1435c:	02c12c23          	sw	a2,56(sp)
   14360:	02b12023          	sw	a1,32(sp)
   14364:	02a12223          	sw	a0,36(sp)
   14368:	00f12c23          	sw	a5,24(sp)
   1436c:	02412783          	lw	a5,36(sp)
   14370:	02012703          	lw	a4,32(sp)
   14374:	00e78733          	add	a4,a5,a4
   14378:	04412783          	lw	a5,68(sp)
   1437c:	02f70733          	mul	a4,a4,a5
   14380:	01270933          	add	s2,a4,s2
   14384:	fff94693          	not	a3,s2
   14388:	41f6d693          	srai	a3,a3,0x1f
   1438c:	00d97db3          	and	s11,s2,a3
   14390:	f69fe06f          	j	132f8 <_vfprintf_r+0x1734>
   14394:	00064703          	lbu	a4,0(a2)
   14398:	00150513          	addi	a0,a0,1
   1439c:	fbdff06f          	j	14358 <_vfprintf_r+0x2794>
   143a0:	000a8c13          	mv	s8,s5
   143a4:	001b0693          	addi	a3,s6,1
   143a8:	04812883          	lw	a7,72(sp)
   143ac:	04c12c83          	lw	s9,76(sp)
   143b0:	05012b83          	lw	s7,80(sp)
   143b4:	05412a03          	lw	s4,84(sp)
   143b8:	03c12a83          	lw	s5,60(sp)
   143bc:	05812483          	lw	s1,88(sp)
   143c0:	05c12d03          	lw	s10,92(sp)
   143c4:	00dc06b3          	add	a3,s8,a3
   143c8:	03000613          	li	a2,48
   143cc:	bc0b4ee3          	bltz	s6,13fa8 <_vfprintf_r+0x23e4>
   143d0:	001c0c13          	addi	s8,s8,1
   143d4:	fecc0fa3          	sb	a2,-1(s8)
   143d8:	ff869ce3          	bne	a3,s8,143d0 <_vfprintf_r+0x280c>
   143dc:	bcdff06f          	j	13fa8 <_vfprintf_r+0x23e4>
   143e0:	40f90f33          	sub	t5,s2,a5
   143e4:	001f0913          	addi	s2,t5,1
   143e8:	fff94693          	not	a3,s2
   143ec:	41f6d693          	srai	a3,a3,0x1f
   143f0:	00d97db3          	and	s11,s2,a3
   143f4:	c5dff06f          	j	14050 <_vfprintf_r+0x248c>
   143f8:	02812703          	lw	a4,40(sp)
   143fc:	06600893          	li	a7,102
   14400:	00eb0f33          	add	t5,s6,a4
   14404:	00ff0933          	add	s2,t5,a5
   14408:	d29ff06f          	j	14130 <_vfprintf_r+0x256c>
   1440c:	00100513          	li	a0,1
   14410:	40d506b3          	sub	a3,a0,a3
   14414:	0ad12623          	sw	a3,172(sp)
   14418:	00d70733          	add	a4,a4,a3
   1441c:	ee4ff06f          	j	13b00 <_vfprintf_r+0x1f3c>
   14420:	00071a63          	bnez	a4,14434 <_vfprintf_r+0x2870>
   14424:	00100d93          	li	s11,1
   14428:	06600893          	li	a7,102
   1442c:	00100913          	li	s2,1
   14430:	c21ff06f          	j	14050 <_vfprintf_r+0x248c>
   14434:	02812783          	lw	a5,40(sp)
   14438:	06600893          	li	a7,102
   1443c:	00178f13          	addi	t5,a5,1
   14440:	016f0933          	add	s2,t5,s6
   14444:	fff94693          	not	a3,s2
   14448:	41f6d693          	srai	a3,a3,0x1f
   1444c:	00d97db3          	and	s11,s2,a3
   14450:	c01ff06f          	j	14050 <_vfprintf_r+0x248c>
   14454:	00200793          	li	a5,2
   14458:	02f12e23          	sw	a5,60(sp)
   1445c:	e59fe06f          	j	132b4 <_vfprintf_r+0x16f0>
   14460:	01412783          	lw	a5,20(sp)
   14464:	0007ab03          	lw	s6,0(a5)
   14468:	00478793          	addi	a5,a5,4
   1446c:	000b5463          	bgez	s6,14474 <_vfprintf_r+0x28b0>
   14470:	fff00b13          	li	s6,-1
   14474:	001ac883          	lbu	a7,1(s5)
   14478:	00f12a23          	sw	a5,20(sp)
   1447c:	00068a93          	mv	s5,a3
   14480:	8d5fd06f          	j	11d54 <_vfprintf_r+0x190>
   14484:	02012223          	sw	zero,36(sp)
   14488:	02012023          	sw	zero,32(sp)
   1448c:	ee1ff06f          	j	1436c <_vfprintf_r+0x27a8>
   14490:	04500613          	li	a2,69
   14494:	08010593          	addi	a1,sp,128
   14498:	e68ff06f          	j	13b00 <_vfprintf_r+0x1f3c>
   1449c:	00c4d783          	lhu	a5,12(s1)
   144a0:	0407e793          	ori	a5,a5,64
   144a4:	00f49623          	sh	a5,12(s1)
   144a8:	a61fd06f          	j	11f08 <_vfprintf_r+0x344>

000144ac <vfprintf>:
   144ac:	00060693          	mv	a3,a2
   144b0:	00058613          	mv	a2,a1
   144b4:	00050593          	mv	a1,a0
   144b8:	f5c1a503          	lw	a0,-164(gp) # 229dc <_impure_ptr>
   144bc:	f08fd06f          	j	11bc4 <_vfprintf_r>

000144c0 <__sbprintf>:
   144c0:	b8010113          	addi	sp,sp,-1152
   144c4:	00c59783          	lh	a5,12(a1)
   144c8:	00e5d703          	lhu	a4,14(a1)
   144cc:	46812c23          	sw	s0,1144(sp)
   144d0:	00058413          	mv	s0,a1
   144d4:	000105b7          	lui	a1,0x10
   144d8:	ffd58593          	addi	a1,a1,-3 # fffd <exit-0xb7>
   144dc:	06442e03          	lw	t3,100(s0)
   144e0:	01c42303          	lw	t1,28(s0)
   144e4:	02442883          	lw	a7,36(s0)
   144e8:	01071713          	slli	a4,a4,0x10
   144ec:	00b7f7b3          	and	a5,a5,a1
   144f0:	00e7e7b3          	or	a5,a5,a4
   144f4:	40000813          	li	a6,1024
   144f8:	00f12a23          	sw	a5,20(sp)
   144fc:	00810593          	addi	a1,sp,8
   14500:	07010793          	addi	a5,sp,112
   14504:	46912a23          	sw	s1,1140(sp)
   14508:	47212823          	sw	s2,1136(sp)
   1450c:	46112e23          	sw	ra,1148(sp)
   14510:	00050913          	mv	s2,a0
   14514:	07c12623          	sw	t3,108(sp)
   14518:	02612223          	sw	t1,36(sp)
   1451c:	03112623          	sw	a7,44(sp)
   14520:	00f12423          	sw	a5,8(sp)
   14524:	00f12c23          	sw	a5,24(sp)
   14528:	01012823          	sw	a6,16(sp)
   1452c:	01012e23          	sw	a6,28(sp)
   14530:	02012023          	sw	zero,32(sp)
   14534:	e90fd0ef          	jal	11bc4 <_vfprintf_r>
   14538:	00050493          	mv	s1,a0
   1453c:	02055c63          	bgez	a0,14574 <__sbprintf+0xb4>
   14540:	01415783          	lhu	a5,20(sp)
   14544:	0407f793          	andi	a5,a5,64
   14548:	00078863          	beqz	a5,14558 <__sbprintf+0x98>
   1454c:	00c45783          	lhu	a5,12(s0)
   14550:	0407e793          	ori	a5,a5,64
   14554:	00f41623          	sh	a5,12(s0)
   14558:	47c12083          	lw	ra,1148(sp)
   1455c:	47812403          	lw	s0,1144(sp)
   14560:	47012903          	lw	s2,1136(sp)
   14564:	00048513          	mv	a0,s1
   14568:	47412483          	lw	s1,1140(sp)
   1456c:	48010113          	addi	sp,sp,1152
   14570:	00008067          	ret
   14574:	00810593          	addi	a1,sp,8
   14578:	00090513          	mv	a0,s2
   1457c:	0ed010ef          	jal	15e68 <_fflush_r>
   14580:	fc0500e3          	beqz	a0,14540 <__sbprintf+0x80>
   14584:	fff00493          	li	s1,-1
   14588:	fb9ff06f          	j	14540 <__sbprintf+0x80>

0001458c <__sprint_r.part.0>:
   1458c:	0645a783          	lw	a5,100(a1)
   14590:	fd010113          	addi	sp,sp,-48
   14594:	01612823          	sw	s6,16(sp)
   14598:	02112623          	sw	ra,44(sp)
   1459c:	01279713          	slli	a4,a5,0x12
   145a0:	00060b13          	mv	s6,a2
   145a4:	0e075863          	bgez	a4,14694 <__sprint_r.part.0+0x108>
   145a8:	00862783          	lw	a5,8(a2)
   145ac:	03212023          	sw	s2,32(sp)
   145b0:	01312e23          	sw	s3,28(sp)
   145b4:	01512a23          	sw	s5,20(sp)
   145b8:	01712623          	sw	s7,12(sp)
   145bc:	00058913          	mv	s2,a1
   145c0:	00062b83          	lw	s7,0(a2)
   145c4:	00050993          	mv	s3,a0
   145c8:	fff00a93          	li	s5,-1
   145cc:	0a078863          	beqz	a5,1467c <__sprint_r.part.0+0xf0>
   145d0:	02812423          	sw	s0,40(sp)
   145d4:	02912223          	sw	s1,36(sp)
   145d8:	01412c23          	sw	s4,24(sp)
   145dc:	01812423          	sw	s8,8(sp)
   145e0:	004bac03          	lw	s8,4(s7)
   145e4:	000ba403          	lw	s0,0(s7)
   145e8:	002c5a13          	srli	s4,s8,0x2
   145ec:	060a0663          	beqz	s4,14658 <__sprint_r.part.0+0xcc>
   145f0:	00000493          	li	s1,0
   145f4:	00c0006f          	j	14600 <__sprint_r.part.0+0x74>
   145f8:	00440413          	addi	s0,s0,4
   145fc:	049a0c63          	beq	s4,s1,14654 <__sprint_r.part.0+0xc8>
   14600:	00042583          	lw	a1,0(s0)
   14604:	00090613          	mv	a2,s2
   14608:	00098513          	mv	a0,s3
   1460c:	084020ef          	jal	16690 <_fputwc_r>
   14610:	00148493          	addi	s1,s1,1
   14614:	ff5512e3          	bne	a0,s5,145f8 <__sprint_r.part.0+0x6c>
   14618:	02812403          	lw	s0,40(sp)
   1461c:	02412483          	lw	s1,36(sp)
   14620:	02012903          	lw	s2,32(sp)
   14624:	01c12983          	lw	s3,28(sp)
   14628:	01812a03          	lw	s4,24(sp)
   1462c:	01412a83          	lw	s5,20(sp)
   14630:	00c12b83          	lw	s7,12(sp)
   14634:	00812c03          	lw	s8,8(sp)
   14638:	fff00513          	li	a0,-1
   1463c:	02c12083          	lw	ra,44(sp)
   14640:	000b2423          	sw	zero,8(s6)
   14644:	000b2223          	sw	zero,4(s6)
   14648:	01012b03          	lw	s6,16(sp)
   1464c:	03010113          	addi	sp,sp,48
   14650:	00008067          	ret
   14654:	008b2783          	lw	a5,8(s6)
   14658:	ffcc7c13          	andi	s8,s8,-4
   1465c:	418787b3          	sub	a5,a5,s8
   14660:	00fb2423          	sw	a5,8(s6)
   14664:	008b8b93          	addi	s7,s7,8
   14668:	f6079ce3          	bnez	a5,145e0 <__sprint_r.part.0+0x54>
   1466c:	02812403          	lw	s0,40(sp)
   14670:	02412483          	lw	s1,36(sp)
   14674:	01812a03          	lw	s4,24(sp)
   14678:	00812c03          	lw	s8,8(sp)
   1467c:	02012903          	lw	s2,32(sp)
   14680:	01c12983          	lw	s3,28(sp)
   14684:	01412a83          	lw	s5,20(sp)
   14688:	00c12b83          	lw	s7,12(sp)
   1468c:	00000513          	li	a0,0
   14690:	fadff06f          	j	1463c <__sprint_r.part.0+0xb0>
   14694:	0b9010ef          	jal	15f4c <__sfvwrite_r>
   14698:	02c12083          	lw	ra,44(sp)
   1469c:	000b2423          	sw	zero,8(s6)
   146a0:	000b2223          	sw	zero,4(s6)
   146a4:	01012b03          	lw	s6,16(sp)
   146a8:	03010113          	addi	sp,sp,48
   146ac:	00008067          	ret

000146b0 <__sprint_r>:
   146b0:	00862703          	lw	a4,8(a2)
   146b4:	00070463          	beqz	a4,146bc <__sprint_r+0xc>
   146b8:	ed5ff06f          	j	1458c <__sprint_r.part.0>
   146bc:	00062223          	sw	zero,4(a2)
   146c0:	00000513          	li	a0,0
   146c4:	00008067          	ret

000146c8 <_vfiprintf_r>:
   146c8:	ed010113          	addi	sp,sp,-304
   146cc:	11312e23          	sw	s3,284(sp)
   146d0:	11612823          	sw	s6,272(sp)
   146d4:	11712623          	sw	s7,268(sp)
   146d8:	11812423          	sw	s8,264(sp)
   146dc:	12112623          	sw	ra,300(sp)
   146e0:	0fb12e23          	sw	s11,252(sp)
   146e4:	00050c13          	mv	s8,a0
   146e8:	00058b13          	mv	s6,a1
   146ec:	00060993          	mv	s3,a2
   146f0:	00068b93          	mv	s7,a3
   146f4:	00050863          	beqz	a0,14704 <_vfiprintf_r+0x3c>
   146f8:	03452783          	lw	a5,52(a0)
   146fc:	00079463          	bnez	a5,14704 <_vfiprintf_r+0x3c>
   14700:	1e80106f          	j	158e8 <_vfiprintf_r+0x1220>
   14704:	00cb1783          	lh	a5,12(s6)
   14708:	01279713          	slli	a4,a5,0x12
   1470c:	02074663          	bltz	a4,14738 <_vfiprintf_r+0x70>
   14710:	064b2703          	lw	a4,100(s6)
   14714:	00002637          	lui	a2,0x2
   14718:	ffffe6b7          	lui	a3,0xffffe
   1471c:	00c7e7b3          	or	a5,a5,a2
   14720:	fff68693          	addi	a3,a3,-1 # ffffdfff <__BSS_END__+0xfffdb1ff>
   14724:	01079793          	slli	a5,a5,0x10
   14728:	4107d793          	srai	a5,a5,0x10
   1472c:	00d77733          	and	a4,a4,a3
   14730:	00fb1623          	sh	a5,12(s6)
   14734:	06eb2223          	sw	a4,100(s6)
   14738:	0087f713          	andi	a4,a5,8
   1473c:	12070e63          	beqz	a4,14878 <_vfiprintf_r+0x1b0>
   14740:	010b2703          	lw	a4,16(s6)
   14744:	12070a63          	beqz	a4,14878 <_vfiprintf_r+0x1b0>
   14748:	01a7f793          	andi	a5,a5,26
   1474c:	00a00713          	li	a4,10
   14750:	14e78663          	beq	a5,a4,1489c <_vfiprintf_r+0x1d4>
   14754:	11512a23          	sw	s5,276(sp)
   14758:	00000d93          	li	s11,0
   1475c:	04c10a93          	addi	s5,sp,76
   14760:	12812423          	sw	s0,296(sp)
   14764:	11412c23          	sw	s4,280(sp)
   14768:	11912223          	sw	s9,260(sp)
   1476c:	11a12023          	sw	s10,256(sp)
   14770:	01812023          	sw	s8,0(sp)
   14774:	12912223          	sw	s1,292(sp)
   14778:	13212023          	sw	s2,288(sp)
   1477c:	05512023          	sw	s5,64(sp)
   14780:	04012423          	sw	zero,72(sp)
   14784:	04012223          	sw	zero,68(sp)
   14788:	000a8413          	mv	s0,s5
   1478c:	00012223          	sw	zero,4(sp)
   14790:	00012a23          	sw	zero,20(sp)
   14794:	00012c23          	sw	zero,24(sp)
   14798:	00012e23          	sw	zero,28(sp)
   1479c:	0000cd17          	auipc	s10,0xc
   147a0:	554d0d13          	addi	s10,s10,1364 # 20cf0 <blanks.1+0x70>
   147a4:	01000a13          	li	s4,16
   147a8:	0000cc97          	auipc	s9,0xc
   147ac:	6b4c8c93          	addi	s9,s9,1716 # 20e5c <zeroes.0>
   147b0:	000d8c13          	mv	s8,s11
   147b4:	000b8813          	mv	a6,s7
   147b8:	0009c783          	lbu	a5,0(s3)
   147bc:	32078863          	beqz	a5,14aec <_vfiprintf_r+0x424>
   147c0:	00098493          	mv	s1,s3
   147c4:	02500713          	li	a4,37
   147c8:	3ce78863          	beq	a5,a4,14b98 <_vfiprintf_r+0x4d0>
   147cc:	0014c783          	lbu	a5,1(s1)
   147d0:	00148493          	addi	s1,s1,1
   147d4:	fe079ae3          	bnez	a5,147c8 <_vfiprintf_r+0x100>
   147d8:	41348933          	sub	s2,s1,s3
   147dc:	31348863          	beq	s1,s3,14aec <_vfiprintf_r+0x424>
   147e0:	04812703          	lw	a4,72(sp)
   147e4:	04412783          	lw	a5,68(sp)
   147e8:	01342023          	sw	s3,0(s0)
   147ec:	00e90733          	add	a4,s2,a4
   147f0:	00178793          	addi	a5,a5,1
   147f4:	01242223          	sw	s2,4(s0)
   147f8:	04e12423          	sw	a4,72(sp)
   147fc:	04f12223          	sw	a5,68(sp)
   14800:	00700693          	li	a3,7
   14804:	00840413          	addi	s0,s0,8
   14808:	02f6d463          	bge	a3,a5,14830 <_vfiprintf_r+0x168>
   1480c:	4e070ce3          	beqz	a4,15504 <_vfiprintf_r+0xe3c>
   14810:	00012503          	lw	a0,0(sp)
   14814:	04010613          	addi	a2,sp,64
   14818:	000b0593          	mv	a1,s6
   1481c:	01012423          	sw	a6,8(sp)
   14820:	d6dff0ef          	jal	1458c <__sprint_r.part.0>
   14824:	00812803          	lw	a6,8(sp)
   14828:	0c051463          	bnez	a0,148f0 <_vfiprintf_r+0x228>
   1482c:	000a8413          	mv	s0,s5
   14830:	0004c783          	lbu	a5,0(s1)
   14834:	012c0c33          	add	s8,s8,s2
   14838:	2a078a63          	beqz	a5,14aec <_vfiprintf_r+0x424>
   1483c:	0014c703          	lbu	a4,1(s1)
   14840:	00148993          	addi	s3,s1,1
   14844:	02010da3          	sb	zero,59(sp)
   14848:	fff00493          	li	s1,-1
   1484c:	00000b93          	li	s7,0
   14850:	00000d93          	li	s11,0
   14854:	05a00913          	li	s2,90
   14858:	00198993          	addi	s3,s3,1
   1485c:	fe070793          	addi	a5,a4,-32
   14860:	10f96c63          	bltu	s2,a5,14978 <_vfiprintf_r+0x2b0>
   14864:	00279793          	slli	a5,a5,0x2
   14868:	01a787b3          	add	a5,a5,s10
   1486c:	0007a783          	lw	a5,0(a5)
   14870:	01a787b3          	add	a5,a5,s10
   14874:	00078067          	jr	a5
   14878:	000b0593          	mv	a1,s6
   1487c:	000c0513          	mv	a0,s8
   14880:	3dd010ef          	jal	1645c <__swsetup_r>
   14884:	00050463          	beqz	a0,1488c <_vfiprintf_r+0x1c4>
   14888:	16c0106f          	j	159f4 <_vfiprintf_r+0x132c>
   1488c:	00cb1783          	lh	a5,12(s6)
   14890:	00a00713          	li	a4,10
   14894:	01a7f793          	andi	a5,a5,26
   14898:	eae79ee3          	bne	a5,a4,14754 <_vfiprintf_r+0x8c>
   1489c:	00eb1783          	lh	a5,14(s6)
   148a0:	ea07cae3          	bltz	a5,14754 <_vfiprintf_r+0x8c>
   148a4:	12c12083          	lw	ra,300(sp)
   148a8:	0fc12d83          	lw	s11,252(sp)
   148ac:	000b8693          	mv	a3,s7
   148b0:	00098613          	mv	a2,s3
   148b4:	10c12b83          	lw	s7,268(sp)
   148b8:	11c12983          	lw	s3,284(sp)
   148bc:	000b0593          	mv	a1,s6
   148c0:	000c0513          	mv	a0,s8
   148c4:	11012b03          	lw	s6,272(sp)
   148c8:	10812c03          	lw	s8,264(sp)
   148cc:	13010113          	addi	sp,sp,304
   148d0:	16c0106f          	j	15a3c <__sbprintf>
   148d4:	00012503          	lw	a0,0(sp)
   148d8:	04010613          	addi	a2,sp,64
   148dc:	000b0593          	mv	a1,s6
   148e0:	01012423          	sw	a6,8(sp)
   148e4:	ca9ff0ef          	jal	1458c <__sprint_r.part.0>
   148e8:	00812803          	lw	a6,8(sp)
   148ec:	1e050863          	beqz	a0,14adc <_vfiprintf_r+0x414>
   148f0:	000c0d93          	mv	s11,s8
   148f4:	00cb5783          	lhu	a5,12(s6)
   148f8:	12812403          	lw	s0,296(sp)
   148fc:	12412483          	lw	s1,292(sp)
   14900:	0407f793          	andi	a5,a5,64
   14904:	12012903          	lw	s2,288(sp)
   14908:	11812a03          	lw	s4,280(sp)
   1490c:	11412a83          	lw	s5,276(sp)
   14910:	10412c83          	lw	s9,260(sp)
   14914:	10012d03          	lw	s10,256(sp)
   14918:	00078463          	beqz	a5,14920 <_vfiprintf_r+0x258>
   1491c:	0d80106f          	j	159f4 <_vfiprintf_r+0x132c>
   14920:	12c12083          	lw	ra,300(sp)
   14924:	11c12983          	lw	s3,284(sp)
   14928:	11012b03          	lw	s6,272(sp)
   1492c:	10c12b83          	lw	s7,268(sp)
   14930:	10812c03          	lw	s8,264(sp)
   14934:	000d8513          	mv	a0,s11
   14938:	0fc12d83          	lw	s11,252(sp)
   1493c:	13010113          	addi	sp,sp,304
   14940:	00008067          	ret
   14944:	00000b93          	li	s7,0
   14948:	fd070793          	addi	a5,a4,-48
   1494c:	00900613          	li	a2,9
   14950:	0009c703          	lbu	a4,0(s3)
   14954:	002b9693          	slli	a3,s7,0x2
   14958:	01768bb3          	add	s7,a3,s7
   1495c:	001b9b93          	slli	s7,s7,0x1
   14960:	01778bb3          	add	s7,a5,s7
   14964:	fd070793          	addi	a5,a4,-48
   14968:	00198993          	addi	s3,s3,1
   1496c:	fef672e3          	bgeu	a2,a5,14950 <_vfiprintf_r+0x288>
   14970:	fe070793          	addi	a5,a4,-32
   14974:	eef978e3          	bgeu	s2,a5,14864 <_vfiprintf_r+0x19c>
   14978:	16070a63          	beqz	a4,14aec <_vfiprintf_r+0x424>
   1497c:	08e10623          	sb	a4,140(sp)
   14980:	02010da3          	sb	zero,59(sp)
   14984:	00100693          	li	a3,1
   14988:	00100893          	li	a7,1
   1498c:	08c10913          	addi	s2,sp,140
   14990:	00000493          	li	s1,0
   14994:	00000f13          	li	t5,0
   14998:	04412603          	lw	a2,68(sp)
   1499c:	084dff93          	andi	t6,s11,132
   149a0:	04812783          	lw	a5,72(sp)
   149a4:	00160593          	addi	a1,a2,1 # 2001 <exit-0xe0b3>
   149a8:	00058e13          	mv	t3,a1
   149ac:	000f9663          	bnez	t6,149b8 <_vfiprintf_r+0x2f0>
   149b0:	40db8733          	sub	a4,s7,a3
   149b4:	1ee04ae3          	bgtz	a4,153a8 <_vfiprintf_r+0xce0>
   149b8:	03b14703          	lbu	a4,59(sp)
   149bc:	02070a63          	beqz	a4,149f0 <_vfiprintf_r+0x328>
   149c0:	03b10713          	addi	a4,sp,59
   149c4:	00178793          	addi	a5,a5,1
   149c8:	00e42023          	sw	a4,0(s0)
   149cc:	00100713          	li	a4,1
   149d0:	00e42223          	sw	a4,4(s0)
   149d4:	04f12423          	sw	a5,72(sp)
   149d8:	05c12223          	sw	t3,68(sp)
   149dc:	00700713          	li	a4,7
   149e0:	0fc748e3          	blt	a4,t3,152d0 <_vfiprintf_r+0xc08>
   149e4:	000e0613          	mv	a2,t3
   149e8:	00840413          	addi	s0,s0,8
   149ec:	001e0e13          	addi	t3,t3,1
   149f0:	060f0863          	beqz	t5,14a60 <_vfiprintf_r+0x398>
   149f4:	03c10713          	addi	a4,sp,60
   149f8:	00278793          	addi	a5,a5,2
   149fc:	00e42023          	sw	a4,0(s0)
   14a00:	00200713          	li	a4,2
   14a04:	00e42223          	sw	a4,4(s0)
   14a08:	04f12423          	sw	a5,72(sp)
   14a0c:	05c12223          	sw	t3,68(sp)
   14a10:	00700713          	li	a4,7
   14a14:	13c756e3          	bge	a4,t3,15340 <_vfiprintf_r+0xc78>
   14a18:	34078ce3          	beqz	a5,15570 <_vfiprintf_r+0xea8>
   14a1c:	00012503          	lw	a0,0(sp)
   14a20:	04010613          	addi	a2,sp,64
   14a24:	000b0593          	mv	a1,s6
   14a28:	03012023          	sw	a6,32(sp)
   14a2c:	00d12823          	sw	a3,16(sp)
   14a30:	01112623          	sw	a7,12(sp)
   14a34:	01f12423          	sw	t6,8(sp)
   14a38:	b55ff0ef          	jal	1458c <__sprint_r.part.0>
   14a3c:	ea051ae3          	bnez	a0,148f0 <_vfiprintf_r+0x228>
   14a40:	04412603          	lw	a2,68(sp)
   14a44:	04812783          	lw	a5,72(sp)
   14a48:	02012803          	lw	a6,32(sp)
   14a4c:	01012683          	lw	a3,16(sp)
   14a50:	00c12883          	lw	a7,12(sp)
   14a54:	00812f83          	lw	t6,8(sp)
   14a58:	000a8413          	mv	s0,s5
   14a5c:	00160e13          	addi	t3,a2,1
   14a60:	08000713          	li	a4,128
   14a64:	64ef8a63          	beq	t6,a4,150b8 <_vfiprintf_r+0x9f0>
   14a68:	411484b3          	sub	s1,s1,a7
   14a6c:	76904a63          	bgtz	s1,151e0 <_vfiprintf_r+0xb18>
   14a70:	00f887b3          	add	a5,a7,a5
   14a74:	01242023          	sw	s2,0(s0)
   14a78:	01142223          	sw	a7,4(s0)
   14a7c:	04f12423          	sw	a5,72(sp)
   14a80:	05c12223          	sw	t3,68(sp)
   14a84:	00700713          	li	a4,7
   14a88:	61c75c63          	bge	a4,t3,150a0 <_vfiprintf_r+0x9d8>
   14a8c:	10078e63          	beqz	a5,14ba8 <_vfiprintf_r+0x4e0>
   14a90:	00012503          	lw	a0,0(sp)
   14a94:	04010613          	addi	a2,sp,64
   14a98:	000b0593          	mv	a1,s6
   14a9c:	01012623          	sw	a6,12(sp)
   14aa0:	00d12423          	sw	a3,8(sp)
   14aa4:	ae9ff0ef          	jal	1458c <__sprint_r.part.0>
   14aa8:	e40514e3          	bnez	a0,148f0 <_vfiprintf_r+0x228>
   14aac:	04812783          	lw	a5,72(sp)
   14ab0:	00c12803          	lw	a6,12(sp)
   14ab4:	00812683          	lw	a3,8(sp)
   14ab8:	000a8413          	mv	s0,s5
   14abc:	004dfd93          	andi	s11,s11,4
   14ac0:	000d8663          	beqz	s11,14acc <_vfiprintf_r+0x404>
   14ac4:	40db84b3          	sub	s1,s7,a3
   14ac8:	0e904c63          	bgtz	s1,14bc0 <_vfiprintf_r+0x4f8>
   14acc:	00dbd463          	bge	s7,a3,14ad4 <_vfiprintf_r+0x40c>
   14ad0:	00068b93          	mv	s7,a3
   14ad4:	017c0c33          	add	s8,s8,s7
   14ad8:	de079ee3          	bnez	a5,148d4 <_vfiprintf_r+0x20c>
   14adc:	0009c783          	lbu	a5,0(s3)
   14ae0:	04012223          	sw	zero,68(sp)
   14ae4:	000a8413          	mv	s0,s5
   14ae8:	cc079ce3          	bnez	a5,147c0 <_vfiprintf_r+0xf8>
   14aec:	04812783          	lw	a5,72(sp)
   14af0:	000c0d93          	mv	s11,s8
   14af4:	00012c03          	lw	s8,0(sp)
   14af8:	de078ee3          	beqz	a5,148f4 <_vfiprintf_r+0x22c>
   14afc:	04010613          	addi	a2,sp,64
   14b00:	000b0593          	mv	a1,s6
   14b04:	000c0513          	mv	a0,s8
   14b08:	a85ff0ef          	jal	1458c <__sprint_r.part.0>
   14b0c:	de9ff06f          	j	148f4 <_vfiprintf_r+0x22c>
   14b10:	00082b83          	lw	s7,0(a6)
   14b14:	00480813          	addi	a6,a6,4
   14b18:	2a0bc663          	bltz	s7,14dc4 <_vfiprintf_r+0x6fc>
   14b1c:	0009c703          	lbu	a4,0(s3)
   14b20:	d39ff06f          	j	14858 <_vfiprintf_r+0x190>
   14b24:	0009c703          	lbu	a4,0(s3)
   14b28:	020ded93          	ori	s11,s11,32
   14b2c:	d2dff06f          	j	14858 <_vfiprintf_r+0x190>
   14b30:	010ded93          	ori	s11,s11,16
   14b34:	020df793          	andi	a5,s11,32
   14b38:	16078e63          	beqz	a5,14cb4 <_vfiprintf_r+0x5ec>
   14b3c:	00780813          	addi	a6,a6,7
   14b40:	ff887813          	andi	a6,a6,-8
   14b44:	00482703          	lw	a4,4(a6)
   14b48:	00082783          	lw	a5,0(a6)
   14b4c:	00880813          	addi	a6,a6,8
   14b50:	00070893          	mv	a7,a4
   14b54:	18074663          	bltz	a4,14ce0 <_vfiprintf_r+0x618>
   14b58:	1a04c463          	bltz	s1,14d00 <_vfiprintf_r+0x638>
   14b5c:	0117e733          	or	a4,a5,a7
   14b60:	f7fdfd93          	andi	s11,s11,-129
   14b64:	18071e63          	bnez	a4,14d00 <_vfiprintf_r+0x638>
   14b68:	660492e3          	bnez	s1,159cc <_vfiprintf_r+0x1304>
   14b6c:	03b14783          	lbu	a5,59(sp)
   14b70:	00000693          	li	a3,0
   14b74:	00000893          	li	a7,0
   14b78:	0f010913          	addi	s2,sp,240
   14b7c:	00078463          	beqz	a5,14b84 <_vfiprintf_r+0x4bc>
   14b80:	00168693          	addi	a3,a3,1
   14b84:	002dff13          	andi	t5,s11,2
   14b88:	e00f08e3          	beqz	t5,14998 <_vfiprintf_r+0x2d0>
   14b8c:	00268693          	addi	a3,a3,2
   14b90:	00200f13          	li	t5,2
   14b94:	e05ff06f          	j	14998 <_vfiprintf_r+0x2d0>
   14b98:	41348933          	sub	s2,s1,s3
   14b9c:	c53492e3          	bne	s1,s3,147e0 <_vfiprintf_r+0x118>
   14ba0:	0004c783          	lbu	a5,0(s1)
   14ba4:	c95ff06f          	j	14838 <_vfiprintf_r+0x170>
   14ba8:	04012223          	sw	zero,68(sp)
   14bac:	004dfd93          	andi	s11,s11,4
   14bb0:	100d80e3          	beqz	s11,154b0 <_vfiprintf_r+0xde8>
   14bb4:	40db84b3          	sub	s1,s7,a3
   14bb8:	0e905ce3          	blez	s1,154b0 <_vfiprintf_r+0xde8>
   14bbc:	000a8413          	mv	s0,s5
   14bc0:	01000713          	li	a4,16
   14bc4:	04412603          	lw	a2,68(sp)
   14bc8:	60975ee3          	bge	a4,s1,159e4 <_vfiprintf_r+0x131c>
   14bcc:	0000ce17          	auipc	t3,0xc
   14bd0:	2a0e0e13          	addi	t3,t3,672 # 20e6c <blanks.1>
   14bd4:	00d12423          	sw	a3,8(sp)
   14bd8:	01000913          	li	s2,16
   14bdc:	00040693          	mv	a3,s0
   14be0:	00700d93          	li	s11,7
   14be4:	00048413          	mv	s0,s1
   14be8:	01012623          	sw	a6,12(sp)
   14bec:	000e0493          	mv	s1,t3
   14bf0:	0180006f          	j	14c08 <_vfiprintf_r+0x540>
   14bf4:	00260593          	addi	a1,a2,2
   14bf8:	00868693          	addi	a3,a3,8
   14bfc:	00070613          	mv	a2,a4
   14c00:	ff040413          	addi	s0,s0,-16
   14c04:	04895863          	bge	s2,s0,14c54 <_vfiprintf_r+0x58c>
   14c08:	01078793          	addi	a5,a5,16
   14c0c:	00160713          	addi	a4,a2,1
   14c10:	0096a023          	sw	s1,0(a3)
   14c14:	0126a223          	sw	s2,4(a3)
   14c18:	04f12423          	sw	a5,72(sp)
   14c1c:	04e12223          	sw	a4,68(sp)
   14c20:	fceddae3          	bge	s11,a4,14bf4 <_vfiprintf_r+0x52c>
   14c24:	48078263          	beqz	a5,150a8 <_vfiprintf_r+0x9e0>
   14c28:	00012503          	lw	a0,0(sp)
   14c2c:	04010613          	addi	a2,sp,64
   14c30:	000b0593          	mv	a1,s6
   14c34:	959ff0ef          	jal	1458c <__sprint_r.part.0>
   14c38:	ca051ce3          	bnez	a0,148f0 <_vfiprintf_r+0x228>
   14c3c:	04412603          	lw	a2,68(sp)
   14c40:	ff040413          	addi	s0,s0,-16
   14c44:	04812783          	lw	a5,72(sp)
   14c48:	000a8693          	mv	a3,s5
   14c4c:	00160593          	addi	a1,a2,1
   14c50:	fa894ce3          	blt	s2,s0,14c08 <_vfiprintf_r+0x540>
   14c54:	00048e13          	mv	t3,s1
   14c58:	00c12803          	lw	a6,12(sp)
   14c5c:	00040493          	mv	s1,s0
   14c60:	00068413          	mv	s0,a3
   14c64:	00812683          	lw	a3,8(sp)
   14c68:	009787b3          	add	a5,a5,s1
   14c6c:	01c42023          	sw	t3,0(s0)
   14c70:	00942223          	sw	s1,4(s0)
   14c74:	04f12423          	sw	a5,72(sp)
   14c78:	04b12223          	sw	a1,68(sp)
   14c7c:	00700713          	li	a4,7
   14c80:	e4b756e3          	bge	a4,a1,14acc <_vfiprintf_r+0x404>
   14c84:	020786e3          	beqz	a5,154b0 <_vfiprintf_r+0xde8>
   14c88:	00012503          	lw	a0,0(sp)
   14c8c:	04010613          	addi	a2,sp,64
   14c90:	000b0593          	mv	a1,s6
   14c94:	01012623          	sw	a6,12(sp)
   14c98:	00d12423          	sw	a3,8(sp)
   14c9c:	8f1ff0ef          	jal	1458c <__sprint_r.part.0>
   14ca0:	c40518e3          	bnez	a0,148f0 <_vfiprintf_r+0x228>
   14ca4:	04812783          	lw	a5,72(sp)
   14ca8:	00c12803          	lw	a6,12(sp)
   14cac:	00812683          	lw	a3,8(sp)
   14cb0:	e1dff06f          	j	14acc <_vfiprintf_r+0x404>
   14cb4:	010df713          	andi	a4,s11,16
   14cb8:	00082783          	lw	a5,0(a6)
   14cbc:	00480813          	addi	a6,a6,4
   14cc0:	0e071c63          	bnez	a4,14db8 <_vfiprintf_r+0x6f0>
   14cc4:	040df713          	andi	a4,s11,64
   14cc8:	0e070463          	beqz	a4,14db0 <_vfiprintf_r+0x6e8>
   14ccc:	01079793          	slli	a5,a5,0x10
   14cd0:	4107d793          	srai	a5,a5,0x10
   14cd4:	41f7d893          	srai	a7,a5,0x1f
   14cd8:	00088713          	mv	a4,a7
   14cdc:	e6075ee3          	bgez	a4,14b58 <_vfiprintf_r+0x490>
   14ce0:	02d00693          	li	a3,45
   14ce4:	00f03733          	snez	a4,a5
   14ce8:	411008b3          	neg	a7,a7
   14cec:	02d10da3          	sb	a3,59(sp)
   14cf0:	40e888b3          	sub	a7,a7,a4
   14cf4:	40f007b3          	neg	a5,a5
   14cf8:	0004c463          	bltz	s1,14d00 <_vfiprintf_r+0x638>
   14cfc:	f7fdfd93          	andi	s11,s11,-129
   14d00:	0c089ae3          	bnez	a7,155d4 <_vfiprintf_r+0xf0c>
   14d04:	00900713          	li	a4,9
   14d08:	0cf766e3          	bltu	a4,a5,155d4 <_vfiprintf_r+0xf0c>
   14d0c:	03078793          	addi	a5,a5,48
   14d10:	0ff7f793          	zext.b	a5,a5
   14d14:	0ef107a3          	sb	a5,239(sp)
   14d18:	00048693          	mv	a3,s1
   14d1c:	00904463          	bgtz	s1,14d24 <_vfiprintf_r+0x65c>
   14d20:	00100693          	li	a3,1
   14d24:	00100893          	li	a7,1
   14d28:	0ef10913          	addi	s2,sp,239
   14d2c:	03b14783          	lbu	a5,59(sp)
   14d30:	e40798e3          	bnez	a5,14b80 <_vfiprintf_r+0x4b8>
   14d34:	e51ff06f          	j	14b84 <_vfiprintf_r+0x4bc>
   14d38:	00082903          	lw	s2,0(a6)
   14d3c:	02010da3          	sb	zero,59(sp)
   14d40:	00480813          	addi	a6,a6,4
   14d44:	3c090ee3          	beqz	s2,15920 <_vfiprintf_r+0x1258>
   14d48:	01012423          	sw	a6,8(sp)
   14d4c:	2604cee3          	bltz	s1,157c8 <_vfiprintf_r+0x1100>
   14d50:	00048613          	mv	a2,s1
   14d54:	00000593          	li	a1,0
   14d58:	00090513          	mv	a0,s2
   14d5c:	1dd010ef          	jal	16738 <memchr>
   14d60:	00812803          	lw	a6,8(sp)
   14d64:	00048893          	mv	a7,s1
   14d68:	00050463          	beqz	a0,14d70 <_vfiprintf_r+0x6a8>
   14d6c:	412508b3          	sub	a7,a0,s2
   14d70:	03b14783          	lbu	a5,59(sp)
   14d74:	fff8c693          	not	a3,a7
   14d78:	41f6d693          	srai	a3,a3,0x1f
   14d7c:	00d8f6b3          	and	a3,a7,a3
   14d80:	00000493          	li	s1,0
   14d84:	00000f13          	li	t5,0
   14d88:	de079ce3          	bnez	a5,14b80 <_vfiprintf_r+0x4b8>
   14d8c:	c0dff06f          	j	14998 <_vfiprintf_r+0x2d0>
   14d90:	00082783          	lw	a5,0(a6)
   14d94:	02010da3          	sb	zero,59(sp)
   14d98:	00480813          	addi	a6,a6,4
   14d9c:	08f10623          	sb	a5,140(sp)
   14da0:	00100693          	li	a3,1
   14da4:	00100893          	li	a7,1
   14da8:	08c10913          	addi	s2,sp,140
   14dac:	be5ff06f          	j	14990 <_vfiprintf_r+0x2c8>
   14db0:	200df713          	andi	a4,s11,512
   14db4:	3e0716e3          	bnez	a4,159a0 <_vfiprintf_r+0x12d8>
   14db8:	41f7d893          	srai	a7,a5,0x1f
   14dbc:	00088713          	mv	a4,a7
   14dc0:	d95ff06f          	j	14b54 <_vfiprintf_r+0x48c>
   14dc4:	41700bb3          	neg	s7,s7
   14dc8:	0009c703          	lbu	a4,0(s3)
   14dcc:	004ded93          	ori	s11,s11,4
   14dd0:	a89ff06f          	j	14858 <_vfiprintf_r+0x190>
   14dd4:	02b00793          	li	a5,43
   14dd8:	0009c703          	lbu	a4,0(s3)
   14ddc:	02f10da3          	sb	a5,59(sp)
   14de0:	a79ff06f          	j	14858 <_vfiprintf_r+0x190>
   14de4:	0009c703          	lbu	a4,0(s3)
   14de8:	080ded93          	ori	s11,s11,128
   14dec:	a6dff06f          	j	14858 <_vfiprintf_r+0x190>
   14df0:	0009c703          	lbu	a4,0(s3)
   14df4:	02a00793          	li	a5,42
   14df8:	00198613          	addi	a2,s3,1
   14dfc:	40f708e3          	beq	a4,a5,15a0c <_vfiprintf_r+0x1344>
   14e00:	fd070793          	addi	a5,a4,-48
   14e04:	00900693          	li	a3,9
   14e08:	00000493          	li	s1,0
   14e0c:	00900593          	li	a1,9
   14e10:	02f6e463          	bltu	a3,a5,14e38 <_vfiprintf_r+0x770>
   14e14:	00064703          	lbu	a4,0(a2)
   14e18:	00249693          	slli	a3,s1,0x2
   14e1c:	009684b3          	add	s1,a3,s1
   14e20:	00149493          	slli	s1,s1,0x1
   14e24:	00f484b3          	add	s1,s1,a5
   14e28:	fd070793          	addi	a5,a4,-48
   14e2c:	00160613          	addi	a2,a2,1
   14e30:	fef5f2e3          	bgeu	a1,a5,14e14 <_vfiprintf_r+0x74c>
   14e34:	0e04c6e3          	bltz	s1,15720 <_vfiprintf_r+0x1058>
   14e38:	00060993          	mv	s3,a2
   14e3c:	a21ff06f          	j	1485c <_vfiprintf_r+0x194>
   14e40:	00012503          	lw	a0,0(sp)
   14e44:	01012423          	sw	a6,8(sp)
   14e48:	369010ef          	jal	169b0 <_localeconv_r>
   14e4c:	00452783          	lw	a5,4(a0)
   14e50:	00078513          	mv	a0,a5
   14e54:	00f12e23          	sw	a5,28(sp)
   14e58:	f89fb0ef          	jal	10de0 <strlen>
   14e5c:	00050793          	mv	a5,a0
   14e60:	00012503          	lw	a0,0(sp)
   14e64:	00f12c23          	sw	a5,24(sp)
   14e68:	349010ef          	jal	169b0 <_localeconv_r>
   14e6c:	00852703          	lw	a4,8(a0)
   14e70:	01812783          	lw	a5,24(sp)
   14e74:	00812803          	lw	a6,8(sp)
   14e78:	00e12a23          	sw	a4,20(sp)
   14e7c:	ca0780e3          	beqz	a5,14b1c <_vfiprintf_r+0x454>
   14e80:	01412783          	lw	a5,20(sp)
   14e84:	0009c703          	lbu	a4,0(s3)
   14e88:	9c0788e3          	beqz	a5,14858 <_vfiprintf_r+0x190>
   14e8c:	0007c783          	lbu	a5,0(a5)
   14e90:	9c0784e3          	beqz	a5,14858 <_vfiprintf_r+0x190>
   14e94:	400ded93          	ori	s11,s11,1024
   14e98:	9c1ff06f          	j	14858 <_vfiprintf_r+0x190>
   14e9c:	0009c703          	lbu	a4,0(s3)
   14ea0:	001ded93          	ori	s11,s11,1
   14ea4:	9b5ff06f          	j	14858 <_vfiprintf_r+0x190>
   14ea8:	03b14783          	lbu	a5,59(sp)
   14eac:	0009c703          	lbu	a4,0(s3)
   14eb0:	9a0794e3          	bnez	a5,14858 <_vfiprintf_r+0x190>
   14eb4:	02000793          	li	a5,32
   14eb8:	02f10da3          	sb	a5,59(sp)
   14ebc:	99dff06f          	j	14858 <_vfiprintf_r+0x190>
   14ec0:	010de713          	ori	a4,s11,16
   14ec4:	02077793          	andi	a5,a4,32
   14ec8:	66078863          	beqz	a5,15538 <_vfiprintf_r+0xe70>
   14ecc:	00780813          	addi	a6,a6,7
   14ed0:	ff887813          	andi	a6,a6,-8
   14ed4:	00082783          	lw	a5,0(a6)
   14ed8:	00482603          	lw	a2,4(a6)
   14edc:	00880813          	addi	a6,a6,8
   14ee0:	02010da3          	sb	zero,59(sp)
   14ee4:	bff77d93          	andi	s11,a4,-1025
   14ee8:	0c04c063          	bltz	s1,14fa8 <_vfiprintf_r+0x8e0>
   14eec:	00c7e6b3          	or	a3,a5,a2
   14ef0:	b7f77713          	andi	a4,a4,-1153
   14ef4:	1e0692e3          	bnez	a3,158d8 <_vfiprintf_r+0x1210>
   14ef8:	000d8693          	mv	a3,s11
   14efc:	00000793          	li	a5,0
   14f00:	00070d93          	mv	s11,a4
   14f04:	08049663          	bnez	s1,14f90 <_vfiprintf_r+0x8c8>
   14f08:	c60792e3          	bnez	a5,14b6c <_vfiprintf_r+0x4a4>
   14f0c:	0016f893          	andi	a7,a3,1
   14f10:	7c088063          	beqz	a7,156d0 <_vfiprintf_r+0x1008>
   14f14:	03000793          	li	a5,48
   14f18:	0ef107a3          	sb	a5,239(sp)
   14f1c:	00088693          	mv	a3,a7
   14f20:	0ef10913          	addi	s2,sp,239
   14f24:	e09ff06f          	j	14d2c <_vfiprintf_r+0x664>
   14f28:	0009c703          	lbu	a4,0(s3)
   14f2c:	06c00793          	li	a5,108
   14f30:	1cf708e3          	beq	a4,a5,15900 <_vfiprintf_r+0x1238>
   14f34:	010ded93          	ori	s11,s11,16
   14f38:	921ff06f          	j	14858 <_vfiprintf_r+0x190>
   14f3c:	0009c703          	lbu	a4,0(s3)
   14f40:	06800793          	li	a5,104
   14f44:	1af706e3          	beq	a4,a5,158f0 <_vfiprintf_r+0x1228>
   14f48:	040ded93          	ori	s11,s11,64
   14f4c:	90dff06f          	j	14858 <_vfiprintf_r+0x190>
   14f50:	010de693          	ori	a3,s11,16
   14f54:	0206f793          	andi	a5,a3,32
   14f58:	5a078c63          	beqz	a5,15510 <_vfiprintf_r+0xe48>
   14f5c:	00780813          	addi	a6,a6,7
   14f60:	ff887813          	andi	a6,a6,-8
   14f64:	00082783          	lw	a5,0(a6)
   14f68:	00482883          	lw	a7,4(a6)
   14f6c:	00880813          	addi	a6,a6,8
   14f70:	02010da3          	sb	zero,59(sp)
   14f74:	00068d93          	mv	s11,a3
   14f78:	d804c4e3          	bltz	s1,14d00 <_vfiprintf_r+0x638>
   14f7c:	0117e733          	or	a4,a5,a7
   14f80:	f7f6fd93          	andi	s11,a3,-129
   14f84:	d6071ee3          	bnez	a4,14d00 <_vfiprintf_r+0x638>
   14f88:	00100793          	li	a5,1
   14f8c:	f6048ee3          	beqz	s1,14f08 <_vfiprintf_r+0x840>
   14f90:	00100713          	li	a4,1
   14f94:	22e78ce3          	beq	a5,a4,159cc <_vfiprintf_r+0x1304>
   14f98:	00200713          	li	a4,2
   14f9c:	1ae78ae3          	beq	a5,a4,15950 <_vfiprintf_r+0x1288>
   14fa0:	00000793          	li	a5,0
   14fa4:	00000613          	li	a2,0
   14fa8:	0f010913          	addi	s2,sp,240
   14fac:	01d61693          	slli	a3,a2,0x1d
   14fb0:	0077f713          	andi	a4,a5,7
   14fb4:	0037d793          	srli	a5,a5,0x3
   14fb8:	03070713          	addi	a4,a4,48
   14fbc:	00f6e7b3          	or	a5,a3,a5
   14fc0:	00365613          	srli	a2,a2,0x3
   14fc4:	fee90fa3          	sb	a4,-1(s2)
   14fc8:	00c7e6b3          	or	a3,a5,a2
   14fcc:	00090593          	mv	a1,s2
   14fd0:	fff90913          	addi	s2,s2,-1
   14fd4:	fc069ce3          	bnez	a3,14fac <_vfiprintf_r+0x8e4>
   14fd8:	001df793          	andi	a5,s11,1
   14fdc:	3a078a63          	beqz	a5,15390 <_vfiprintf_r+0xcc8>
   14fe0:	03000793          	li	a5,48
   14fe4:	3af70663          	beq	a4,a5,15390 <_vfiprintf_r+0xcc8>
   14fe8:	ffe58593          	addi	a1,a1,-2
   14fec:	fef90fa3          	sb	a5,-1(s2)
   14ff0:	0f010793          	addi	a5,sp,240
   14ff4:	40b788b3          	sub	a7,a5,a1
   14ff8:	00048693          	mv	a3,s1
   14ffc:	7114cc63          	blt	s1,a7,15714 <_vfiprintf_r+0x104c>
   15000:	00058913          	mv	s2,a1
   15004:	d29ff06f          	j	14d2c <_vfiprintf_r+0x664>
   15008:	00008737          	lui	a4,0x8
   1500c:	83070713          	addi	a4,a4,-2000 # 7830 <exit-0x8884>
   15010:	02e11e23          	sh	a4,60(sp)
   15014:	0000c717          	auipc	a4,0xc
   15018:	91870713          	addi	a4,a4,-1768 # 2092c <_exit+0xf4>
   1501c:	00082783          	lw	a5,0(a6)
   15020:	00000613          	li	a2,0
   15024:	002ded93          	ori	s11,s11,2
   15028:	00480813          	addi	a6,a6,4
   1502c:	00e12223          	sw	a4,4(sp)
   15030:	02010da3          	sb	zero,59(sp)
   15034:	3204c463          	bltz	s1,1535c <_vfiprintf_r+0xc94>
   15038:	00c7e733          	or	a4,a5,a2
   1503c:	f7fdf593          	andi	a1,s11,-129
   15040:	30071863          	bnez	a4,15350 <_vfiprintf_r+0xc88>
   15044:	000d8693          	mv	a3,s11
   15048:	00200793          	li	a5,2
   1504c:	00058d93          	mv	s11,a1
   15050:	eb5ff06f          	j	14f04 <_vfiprintf_r+0x83c>
   15054:	020df793          	andi	a5,s11,32
   15058:	68079a63          	bnez	a5,156ec <_vfiprintf_r+0x1024>
   1505c:	010df793          	andi	a5,s11,16
   15060:	0a0798e3          	bnez	a5,15910 <_vfiprintf_r+0x1248>
   15064:	040df793          	andi	a5,s11,64
   15068:	120794e3          	bnez	a5,15990 <_vfiprintf_r+0x12c8>
   1506c:	200dfd93          	andi	s11,s11,512
   15070:	0a0d80e3          	beqz	s11,15910 <_vfiprintf_r+0x1248>
   15074:	00082783          	lw	a5,0(a6)
   15078:	00480813          	addi	a6,a6,4
   1507c:	01878023          	sb	s8,0(a5)
   15080:	f38ff06f          	j	147b8 <_vfiprintf_r+0xf0>
   15084:	00100713          	li	a4,1
   15088:	00088793          	mv	a5,a7
   1508c:	05212623          	sw	s2,76(sp)
   15090:	05112823          	sw	a7,80(sp)
   15094:	05112423          	sw	a7,72(sp)
   15098:	04e12223          	sw	a4,68(sp)
   1509c:	000a8413          	mv	s0,s5
   150a0:	00840413          	addi	s0,s0,8
   150a4:	a19ff06f          	j	14abc <_vfiprintf_r+0x3f4>
   150a8:	00100593          	li	a1,1
   150ac:	00000613          	li	a2,0
   150b0:	000a8693          	mv	a3,s5
   150b4:	b4dff06f          	j	14c00 <_vfiprintf_r+0x538>
   150b8:	40db8f33          	sub	t5,s7,a3
   150bc:	9be056e3          	blez	t5,14a68 <_vfiprintf_r+0x3a0>
   150c0:	01000713          	li	a4,16
   150c4:	13e75ce3          	bge	a4,t5,159fc <_vfiprintf_r+0x1334>
   150c8:	0000ce97          	auipc	t4,0xc
   150cc:	d94e8e93          	addi	t4,t4,-620 # 20e5c <zeroes.0>
   150d0:	00912623          	sw	s1,12(sp)
   150d4:	00d12823          	sw	a3,16(sp)
   150d8:	01000e13          	li	t3,16
   150dc:	00040693          	mv	a3,s0
   150e0:	00700f93          	li	t6,7
   150e4:	01112423          	sw	a7,8(sp)
   150e8:	000f0413          	mv	s0,t5
   150ec:	03012023          	sw	a6,32(sp)
   150f0:	000e8493          	mv	s1,t4
   150f4:	0180006f          	j	1510c <_vfiprintf_r+0xa44>
   150f8:	00260593          	addi	a1,a2,2
   150fc:	00868693          	addi	a3,a3,8
   15100:	00070613          	mv	a2,a4
   15104:	ff040413          	addi	s0,s0,-16
   15108:	048e5c63          	bge	t3,s0,15160 <_vfiprintf_r+0xa98>
   1510c:	01078793          	addi	a5,a5,16
   15110:	00160713          	addi	a4,a2,1
   15114:	0096a023          	sw	s1,0(a3)
   15118:	01c6a223          	sw	t3,4(a3)
   1511c:	04f12423          	sw	a5,72(sp)
   15120:	04e12223          	sw	a4,68(sp)
   15124:	fcefdae3          	bge	t6,a4,150f8 <_vfiprintf_r+0xa30>
   15128:	18078c63          	beqz	a5,152c0 <_vfiprintf_r+0xbf8>
   1512c:	00012503          	lw	a0,0(sp)
   15130:	04010613          	addi	a2,sp,64
   15134:	000b0593          	mv	a1,s6
   15138:	c54ff0ef          	jal	1458c <__sprint_r.part.0>
   1513c:	fa051a63          	bnez	a0,148f0 <_vfiprintf_r+0x228>
   15140:	04412603          	lw	a2,68(sp)
   15144:	01000e13          	li	t3,16
   15148:	ff040413          	addi	s0,s0,-16
   1514c:	04812783          	lw	a5,72(sp)
   15150:	000a8693          	mv	a3,s5
   15154:	00160593          	addi	a1,a2,1
   15158:	00700f93          	li	t6,7
   1515c:	fa8e48e3          	blt	t3,s0,1510c <_vfiprintf_r+0xa44>
   15160:	00040f13          	mv	t5,s0
   15164:	00048e93          	mv	t4,s1
   15168:	00068413          	mv	s0,a3
   1516c:	00812883          	lw	a7,8(sp)
   15170:	01012683          	lw	a3,16(sp)
   15174:	02012803          	lw	a6,32(sp)
   15178:	00c12483          	lw	s1,12(sp)
   1517c:	01e787b3          	add	a5,a5,t5
   15180:	01d42023          	sw	t4,0(s0)
   15184:	01e42223          	sw	t5,4(s0)
   15188:	04f12423          	sw	a5,72(sp)
   1518c:	04b12223          	sw	a1,68(sp)
   15190:	00700713          	li	a4,7
   15194:	54b75463          	bge	a4,a1,156dc <_vfiprintf_r+0x1014>
   15198:	7c078263          	beqz	a5,1595c <_vfiprintf_r+0x1294>
   1519c:	00012503          	lw	a0,0(sp)
   151a0:	04010613          	addi	a2,sp,64
   151a4:	000b0593          	mv	a1,s6
   151a8:	01012823          	sw	a6,16(sp)
   151ac:	00d12623          	sw	a3,12(sp)
   151b0:	01112423          	sw	a7,8(sp)
   151b4:	bd8ff0ef          	jal	1458c <__sprint_r.part.0>
   151b8:	f2051c63          	bnez	a0,148f0 <_vfiprintf_r+0x228>
   151bc:	00812883          	lw	a7,8(sp)
   151c0:	04412603          	lw	a2,68(sp)
   151c4:	04812783          	lw	a5,72(sp)
   151c8:	411484b3          	sub	s1,s1,a7
   151cc:	01012803          	lw	a6,16(sp)
   151d0:	00c12683          	lw	a3,12(sp)
   151d4:	000a8413          	mv	s0,s5
   151d8:	00160e13          	addi	t3,a2,1
   151dc:	88905ae3          	blez	s1,14a70 <_vfiprintf_r+0x3a8>
   151e0:	0000ce97          	auipc	t4,0xc
   151e4:	c7ce8e93          	addi	t4,t4,-900 # 20e5c <zeroes.0>
   151e8:	0a9a5063          	bge	s4,s1,15288 <_vfiprintf_r+0xbc0>
   151ec:	00d12623          	sw	a3,12(sp)
   151f0:	00700f13          	li	t5,7
   151f4:	00040693          	mv	a3,s0
   151f8:	01112423          	sw	a7,8(sp)
   151fc:	00048413          	mv	s0,s1
   15200:	01012823          	sw	a6,16(sp)
   15204:	000c8493          	mv	s1,s9
   15208:	0180006f          	j	15220 <_vfiprintf_r+0xb58>
   1520c:	00260e13          	addi	t3,a2,2
   15210:	00868693          	addi	a3,a3,8
   15214:	00070613          	mv	a2,a4
   15218:	ff040413          	addi	s0,s0,-16
   1521c:	048a5a63          	bge	s4,s0,15270 <_vfiprintf_r+0xba8>
   15220:	01078793          	addi	a5,a5,16
   15224:	00160713          	addi	a4,a2,1
   15228:	0196a023          	sw	s9,0(a3)
   1522c:	0146a223          	sw	s4,4(a3)
   15230:	04f12423          	sw	a5,72(sp)
   15234:	04e12223          	sw	a4,68(sp)
   15238:	fcef5ae3          	bge	t5,a4,1520c <_vfiprintf_r+0xb44>
   1523c:	06078a63          	beqz	a5,152b0 <_vfiprintf_r+0xbe8>
   15240:	00012503          	lw	a0,0(sp)
   15244:	04010613          	addi	a2,sp,64
   15248:	000b0593          	mv	a1,s6
   1524c:	b40ff0ef          	jal	1458c <__sprint_r.part.0>
   15250:	ea051063          	bnez	a0,148f0 <_vfiprintf_r+0x228>
   15254:	04412603          	lw	a2,68(sp)
   15258:	ff040413          	addi	s0,s0,-16
   1525c:	04812783          	lw	a5,72(sp)
   15260:	000a8693          	mv	a3,s5
   15264:	00160e13          	addi	t3,a2,1
   15268:	00700f13          	li	t5,7
   1526c:	fa8a4ae3          	blt	s4,s0,15220 <_vfiprintf_r+0xb58>
   15270:	00048e93          	mv	t4,s1
   15274:	00812883          	lw	a7,8(sp)
   15278:	00040493          	mv	s1,s0
   1527c:	01012803          	lw	a6,16(sp)
   15280:	00068413          	mv	s0,a3
   15284:	00c12683          	lw	a3,12(sp)
   15288:	009787b3          	add	a5,a5,s1
   1528c:	01d42023          	sw	t4,0(s0)
   15290:	00942223          	sw	s1,4(s0)
   15294:	04f12423          	sw	a5,72(sp)
   15298:	05c12223          	sw	t3,68(sp)
   1529c:	00700713          	li	a4,7
   152a0:	23c74063          	blt	a4,t3,154c0 <_vfiprintf_r+0xdf8>
   152a4:	00840413          	addi	s0,s0,8
   152a8:	001e0e13          	addi	t3,t3,1
   152ac:	fc4ff06f          	j	14a70 <_vfiprintf_r+0x3a8>
   152b0:	00100e13          	li	t3,1
   152b4:	00000613          	li	a2,0
   152b8:	000a8693          	mv	a3,s5
   152bc:	f5dff06f          	j	15218 <_vfiprintf_r+0xb50>
   152c0:	00100593          	li	a1,1
   152c4:	00000613          	li	a2,0
   152c8:	000a8693          	mv	a3,s5
   152cc:	e39ff06f          	j	15104 <_vfiprintf_r+0xa3c>
   152d0:	04078a63          	beqz	a5,15324 <_vfiprintf_r+0xc5c>
   152d4:	00012503          	lw	a0,0(sp)
   152d8:	04010613          	addi	a2,sp,64
   152dc:	000b0593          	mv	a1,s6
   152e0:	03012223          	sw	a6,36(sp)
   152e4:	02d12023          	sw	a3,32(sp)
   152e8:	01112823          	sw	a7,16(sp)
   152ec:	01f12623          	sw	t6,12(sp)
   152f0:	01e12423          	sw	t5,8(sp)
   152f4:	a98ff0ef          	jal	1458c <__sprint_r.part.0>
   152f8:	de051c63          	bnez	a0,148f0 <_vfiprintf_r+0x228>
   152fc:	04412603          	lw	a2,68(sp)
   15300:	04812783          	lw	a5,72(sp)
   15304:	02412803          	lw	a6,36(sp)
   15308:	02012683          	lw	a3,32(sp)
   1530c:	01012883          	lw	a7,16(sp)
   15310:	00c12f83          	lw	t6,12(sp)
   15314:	00812f03          	lw	t5,8(sp)
   15318:	000a8413          	mv	s0,s5
   1531c:	00160e13          	addi	t3,a2,1
   15320:	ed0ff06f          	j	149f0 <_vfiprintf_r+0x328>
   15324:	3e0f0063          	beqz	t5,15704 <_vfiprintf_r+0x103c>
   15328:	03c10793          	addi	a5,sp,60
   1532c:	04f12623          	sw	a5,76(sp)
   15330:	00200793          	li	a5,2
   15334:	04f12823          	sw	a5,80(sp)
   15338:	00100e13          	li	t3,1
   1533c:	000a8413          	mv	s0,s5
   15340:	000e0613          	mv	a2,t3
   15344:	00840413          	addi	s0,s0,8
   15348:	001e0e13          	addi	t3,t3,1
   1534c:	f14ff06f          	j	14a60 <_vfiprintf_r+0x398>
   15350:	00200713          	li	a4,2
   15354:	00058d93          	mv	s11,a1
   15358:	c40708e3          	beqz	a4,14fa8 <_vfiprintf_r+0x8e0>
   1535c:	00412583          	lw	a1,4(sp)
   15360:	0f010913          	addi	s2,sp,240
   15364:	00f7f713          	andi	a4,a5,15
   15368:	00e58733          	add	a4,a1,a4
   1536c:	00074683          	lbu	a3,0(a4)
   15370:	0047d793          	srli	a5,a5,0x4
   15374:	01c61713          	slli	a4,a2,0x1c
   15378:	00f767b3          	or	a5,a4,a5
   1537c:	00465613          	srli	a2,a2,0x4
   15380:	fed90fa3          	sb	a3,-1(s2)
   15384:	00c7e733          	or	a4,a5,a2
   15388:	fff90913          	addi	s2,s2,-1
   1538c:	fc071ce3          	bnez	a4,15364 <_vfiprintf_r+0xc9c>
   15390:	0f010793          	addi	a5,sp,240
   15394:	412788b3          	sub	a7,a5,s2
   15398:	00048693          	mv	a3,s1
   1539c:	9914d8e3          	bge	s1,a7,14d2c <_vfiprintf_r+0x664>
   153a0:	00088693          	mv	a3,a7
   153a4:	989ff06f          	j	14d2c <_vfiprintf_r+0x664>
   153a8:	01000513          	li	a0,16
   153ac:	62e55463          	bge	a0,a4,159d4 <_vfiprintf_r+0x130c>
   153b0:	0000ce17          	auipc	t3,0xc
   153b4:	abce0e13          	addi	t3,t3,-1348 # 20e6c <blanks.1>
   153b8:	02912023          	sw	s1,32(sp)
   153bc:	02d12223          	sw	a3,36(sp)
   153c0:	01000e93          	li	t4,16
   153c4:	00040693          	mv	a3,s0
   153c8:	00700293          	li	t0,7
   153cc:	01e12423          	sw	t5,8(sp)
   153d0:	01f12623          	sw	t6,12(sp)
   153d4:	01112823          	sw	a7,16(sp)
   153d8:	00070413          	mv	s0,a4
   153dc:	03012423          	sw	a6,40(sp)
   153e0:	000e0493          	mv	s1,t3
   153e4:	01c0006f          	j	15400 <_vfiprintf_r+0xd38>
   153e8:	00260513          	addi	a0,a2,2
   153ec:	00868693          	addi	a3,a3,8
   153f0:	00058613          	mv	a2,a1
   153f4:	ff040413          	addi	s0,s0,-16
   153f8:	048edc63          	bge	t4,s0,15450 <_vfiprintf_r+0xd88>
   153fc:	00160593          	addi	a1,a2,1
   15400:	01078793          	addi	a5,a5,16
   15404:	0096a023          	sw	s1,0(a3)
   15408:	01d6a223          	sw	t4,4(a3)
   1540c:	04f12423          	sw	a5,72(sp)
   15410:	04b12223          	sw	a1,68(sp)
   15414:	fcb2dae3          	bge	t0,a1,153e8 <_vfiprintf_r+0xd20>
   15418:	08078463          	beqz	a5,154a0 <_vfiprintf_r+0xdd8>
   1541c:	00012503          	lw	a0,0(sp)
   15420:	04010613          	addi	a2,sp,64
   15424:	000b0593          	mv	a1,s6
   15428:	964ff0ef          	jal	1458c <__sprint_r.part.0>
   1542c:	cc051263          	bnez	a0,148f0 <_vfiprintf_r+0x228>
   15430:	04412603          	lw	a2,68(sp)
   15434:	01000e93          	li	t4,16
   15438:	ff040413          	addi	s0,s0,-16
   1543c:	04812783          	lw	a5,72(sp)
   15440:	000a8693          	mv	a3,s5
   15444:	00160513          	addi	a0,a2,1
   15448:	00700293          	li	t0,7
   1544c:	fa8ec8e3          	blt	t4,s0,153fc <_vfiprintf_r+0xd34>
   15450:	00040713          	mv	a4,s0
   15454:	00048e13          	mv	t3,s1
   15458:	00068413          	mv	s0,a3
   1545c:	00812f03          	lw	t5,8(sp)
   15460:	00c12f83          	lw	t6,12(sp)
   15464:	01012883          	lw	a7,16(sp)
   15468:	02412683          	lw	a3,36(sp)
   1546c:	02812803          	lw	a6,40(sp)
   15470:	02012483          	lw	s1,32(sp)
   15474:	00e787b3          	add	a5,a5,a4
   15478:	00e42223          	sw	a4,4(s0)
   1547c:	01c42023          	sw	t3,0(s0)
   15480:	04f12423          	sw	a5,72(sp)
   15484:	04a12223          	sw	a0,68(sp)
   15488:	00700713          	li	a4,7
   1548c:	0ea74a63          	blt	a4,a0,15580 <_vfiprintf_r+0xeb8>
   15490:	00840413          	addi	s0,s0,8
   15494:	00150e13          	addi	t3,a0,1
   15498:	00050613          	mv	a2,a0
   1549c:	d1cff06f          	j	149b8 <_vfiprintf_r+0x2f0>
   154a0:	00000613          	li	a2,0
   154a4:	00100513          	li	a0,1
   154a8:	000a8693          	mv	a3,s5
   154ac:	f49ff06f          	j	153f4 <_vfiprintf_r+0xd2c>
   154b0:	00dbd463          	bge	s7,a3,154b8 <_vfiprintf_r+0xdf0>
   154b4:	00068b93          	mv	s7,a3
   154b8:	017c0c33          	add	s8,s8,s7
   154bc:	e20ff06f          	j	14adc <_vfiprintf_r+0x414>
   154c0:	bc0782e3          	beqz	a5,15084 <_vfiprintf_r+0x9bc>
   154c4:	00012503          	lw	a0,0(sp)
   154c8:	04010613          	addi	a2,sp,64
   154cc:	000b0593          	mv	a1,s6
   154d0:	01012823          	sw	a6,16(sp)
   154d4:	00d12623          	sw	a3,12(sp)
   154d8:	01112423          	sw	a7,8(sp)
   154dc:	8b0ff0ef          	jal	1458c <__sprint_r.part.0>
   154e0:	c0051863          	bnez	a0,148f0 <_vfiprintf_r+0x228>
   154e4:	04412e03          	lw	t3,68(sp)
   154e8:	04812783          	lw	a5,72(sp)
   154ec:	01012803          	lw	a6,16(sp)
   154f0:	00c12683          	lw	a3,12(sp)
   154f4:	00812883          	lw	a7,8(sp)
   154f8:	000a8413          	mv	s0,s5
   154fc:	001e0e13          	addi	t3,t3,1
   15500:	d70ff06f          	j	14a70 <_vfiprintf_r+0x3a8>
   15504:	04012223          	sw	zero,68(sp)
   15508:	000a8413          	mv	s0,s5
   1550c:	b24ff06f          	j	14830 <_vfiprintf_r+0x168>
   15510:	0106f713          	andi	a4,a3,16
   15514:	00082783          	lw	a5,0(a6)
   15518:	00480813          	addi	a6,a6,4
   1551c:	00071a63          	bnez	a4,15530 <_vfiprintf_r+0xe68>
   15520:	0406f713          	andi	a4,a3,64
   15524:	40070c63          	beqz	a4,1593c <_vfiprintf_r+0x1274>
   15528:	01079793          	slli	a5,a5,0x10
   1552c:	0107d793          	srli	a5,a5,0x10
   15530:	00000893          	li	a7,0
   15534:	a3dff06f          	j	14f70 <_vfiprintf_r+0x8a8>
   15538:	01077693          	andi	a3,a4,16
   1553c:	00082783          	lw	a5,0(a6)
   15540:	00480813          	addi	a6,a6,4
   15544:	02069263          	bnez	a3,15568 <_vfiprintf_r+0xea0>
   15548:	04077693          	andi	a3,a4,64
   1554c:	00068a63          	beqz	a3,15560 <_vfiprintf_r+0xe98>
   15550:	01079793          	slli	a5,a5,0x10
   15554:	0107d793          	srli	a5,a5,0x10
   15558:	00000613          	li	a2,0
   1555c:	985ff06f          	j	14ee0 <_vfiprintf_r+0x818>
   15560:	20077693          	andi	a3,a4,512
   15564:	44069863          	bnez	a3,159b4 <_vfiprintf_r+0x12ec>
   15568:	00000613          	li	a2,0
   1556c:	975ff06f          	j	14ee0 <_vfiprintf_r+0x818>
   15570:	00100e13          	li	t3,1
   15574:	00000613          	li	a2,0
   15578:	000a8413          	mv	s0,s5
   1557c:	ce4ff06f          	j	14a60 <_vfiprintf_r+0x398>
   15580:	24078e63          	beqz	a5,157dc <_vfiprintf_r+0x1114>
   15584:	00012503          	lw	a0,0(sp)
   15588:	04010613          	addi	a2,sp,64
   1558c:	000b0593          	mv	a1,s6
   15590:	03012223          	sw	a6,36(sp)
   15594:	02d12023          	sw	a3,32(sp)
   15598:	01112823          	sw	a7,16(sp)
   1559c:	01f12623          	sw	t6,12(sp)
   155a0:	01e12423          	sw	t5,8(sp)
   155a4:	fe9fe0ef          	jal	1458c <__sprint_r.part.0>
   155a8:	b4051463          	bnez	a0,148f0 <_vfiprintf_r+0x228>
   155ac:	04412603          	lw	a2,68(sp)
   155b0:	04812783          	lw	a5,72(sp)
   155b4:	02412803          	lw	a6,36(sp)
   155b8:	02012683          	lw	a3,32(sp)
   155bc:	01012883          	lw	a7,16(sp)
   155c0:	00c12f83          	lw	t6,12(sp)
   155c4:	00812f03          	lw	t5,8(sp)
   155c8:	000a8413          	mv	s0,s5
   155cc:	00160e13          	addi	t3,a2,1
   155d0:	be8ff06f          	j	149b8 <_vfiprintf_r+0x2f0>
   155d4:	ccccde37          	lui	t3,0xccccd
   155d8:	ccccd3b7          	lui	t2,0xccccd
   155dc:	01412303          	lw	t1,20(sp)
   155e0:	400dff13          	andi	t5,s11,1024
   155e4:	00000513          	li	a0,0
   155e8:	0f010593          	addi	a1,sp,240
   155ec:	00500e93          	li	t4,5
   155f0:	ccde0e13          	addi	t3,t3,-819 # cccccccd <__BSS_END__+0xccca9ecd>
   155f4:	ccc38393          	addi	t2,t2,-820 # cccccccc <__BSS_END__+0xccca9ecc>
   155f8:	0ff00f93          	li	t6,255
   155fc:	0540006f          	j	15650 <_vfiprintf_r+0xf88>
   15600:	00f93733          	sltu	a4,s2,a5
   15604:	00e90733          	add	a4,s2,a4
   15608:	03d77733          	remu	a4,a4,t4
   1560c:	40e78733          	sub	a4,a5,a4
   15610:	00e7b633          	sltu	a2,a5,a4
   15614:	40c88633          	sub	a2,a7,a2
   15618:	027702b3          	mul	t0,a4,t2
   1561c:	03c60633          	mul	a2,a2,t3
   15620:	03c735b3          	mulhu	a1,a4,t3
   15624:	00560633          	add	a2,a2,t0
   15628:	03c70733          	mul	a4,a4,t3
   1562c:	00b60633          	add	a2,a2,a1
   15630:	01f61593          	slli	a1,a2,0x1f
   15634:	00165613          	srli	a2,a2,0x1
   15638:	00175713          	srli	a4,a4,0x1
   1563c:	00e5e733          	or	a4,a1,a4
   15640:	32088663          	beqz	a7,1596c <_vfiprintf_r+0x12a4>
   15644:	00070793          	mv	a5,a4
   15648:	00060893          	mv	a7,a2
   1564c:	00068593          	mv	a1,a3
   15650:	01178933          	add	s2,a5,a7
   15654:	00f93733          	sltu	a4,s2,a5
   15658:	00e90733          	add	a4,s2,a4
   1565c:	03d77733          	remu	a4,a4,t4
   15660:	fff58693          	addi	a3,a1,-1
   15664:	00150513          	addi	a0,a0,1
   15668:	40e78733          	sub	a4,a5,a4
   1566c:	00e7b2b3          	sltu	t0,a5,a4
   15670:	405882b3          	sub	t0,a7,t0
   15674:	03c73633          	mulhu	a2,a4,t3
   15678:	03c282b3          	mul	t0,t0,t3
   1567c:	03c70733          	mul	a4,a4,t3
   15680:	00c282b3          	add	t0,t0,a2
   15684:	01f29293          	slli	t0,t0,0x1f
   15688:	00175613          	srli	a2,a4,0x1
   1568c:	00c2e633          	or	a2,t0,a2
   15690:	00261713          	slli	a4,a2,0x2
   15694:	00c70733          	add	a4,a4,a2
   15698:	00171713          	slli	a4,a4,0x1
   1569c:	40e78733          	sub	a4,a5,a4
   156a0:	03070713          	addi	a4,a4,48
   156a4:	fee58fa3          	sb	a4,-1(a1)
   156a8:	f40f0ce3          	beqz	t5,15600 <_vfiprintf_r+0xf38>
   156ac:	00034703          	lbu	a4,0(t1)
   156b0:	f4a718e3          	bne	a4,a0,15600 <_vfiprintf_r+0xf38>
   156b4:	f5f506e3          	beq	a0,t6,15600 <_vfiprintf_r+0xf38>
   156b8:	14089463          	bnez	a7,15800 <_vfiprintf_r+0x1138>
   156bc:	00900713          	li	a4,9
   156c0:	14f76063          	bltu	a4,a5,15800 <_vfiprintf_r+0x1138>
   156c4:	00612a23          	sw	t1,20(sp)
   156c8:	00068913          	mv	s2,a3
   156cc:	cc5ff06f          	j	15390 <_vfiprintf_r+0xcc8>
   156d0:	00000693          	li	a3,0
   156d4:	0f010913          	addi	s2,sp,240
   156d8:	e54ff06f          	j	14d2c <_vfiprintf_r+0x664>
   156dc:	00840413          	addi	s0,s0,8
   156e0:	00158e13          	addi	t3,a1,1
   156e4:	00058613          	mv	a2,a1
   156e8:	b80ff06f          	j	14a68 <_vfiprintf_r+0x3a0>
   156ec:	00082783          	lw	a5,0(a6)
   156f0:	41fc5713          	srai	a4,s8,0x1f
   156f4:	00480813          	addi	a6,a6,4
   156f8:	0187a023          	sw	s8,0(a5)
   156fc:	00e7a223          	sw	a4,4(a5)
   15700:	8b8ff06f          	j	147b8 <_vfiprintf_r+0xf0>
   15704:	00000613          	li	a2,0
   15708:	00100e13          	li	t3,1
   1570c:	000a8413          	mv	s0,s5
   15710:	b50ff06f          	j	14a60 <_vfiprintf_r+0x398>
   15714:	00088693          	mv	a3,a7
   15718:	00058913          	mv	s2,a1
   1571c:	e10ff06f          	j	14d2c <_vfiprintf_r+0x664>
   15720:	fff00493          	li	s1,-1
   15724:	00060993          	mv	s3,a2
   15728:	934ff06f          	j	1485c <_vfiprintf_r+0x194>
   1572c:	000d8693          	mv	a3,s11
   15730:	825ff06f          	j	14f54 <_vfiprintf_r+0x88c>
   15734:	000d8713          	mv	a4,s11
   15738:	f8cff06f          	j	14ec4 <_vfiprintf_r+0x7fc>
   1573c:	0000b797          	auipc	a5,0xb
   15740:	20478793          	addi	a5,a5,516 # 20940 <_exit+0x108>
   15744:	00f12223          	sw	a5,4(sp)
   15748:	020df793          	andi	a5,s11,32
   1574c:	04078a63          	beqz	a5,157a0 <_vfiprintf_r+0x10d8>
   15750:	00780813          	addi	a6,a6,7
   15754:	ff887813          	andi	a6,a6,-8
   15758:	00082783          	lw	a5,0(a6)
   1575c:	00482603          	lw	a2,4(a6)
   15760:	00880813          	addi	a6,a6,8
   15764:	001df693          	andi	a3,s11,1
   15768:	00068e63          	beqz	a3,15784 <_vfiprintf_r+0x10bc>
   1576c:	00c7e6b3          	or	a3,a5,a2
   15770:	00068a63          	beqz	a3,15784 <_vfiprintf_r+0x10bc>
   15774:	03000693          	li	a3,48
   15778:	02d10e23          	sb	a3,60(sp)
   1577c:	02e10ea3          	sb	a4,61(sp)
   15780:	002ded93          	ori	s11,s11,2
   15784:	bffdfd93          	andi	s11,s11,-1025
   15788:	8a9ff06f          	j	15030 <_vfiprintf_r+0x968>
   1578c:	0000b797          	auipc	a5,0xb
   15790:	1a078793          	addi	a5,a5,416 # 2092c <_exit+0xf4>
   15794:	00f12223          	sw	a5,4(sp)
   15798:	020df793          	andi	a5,s11,32
   1579c:	fa079ae3          	bnez	a5,15750 <_vfiprintf_r+0x1088>
   157a0:	010df693          	andi	a3,s11,16
   157a4:	00082783          	lw	a5,0(a6)
   157a8:	00480813          	addi	a6,a6,4
   157ac:	12069263          	bnez	a3,158d0 <_vfiprintf_r+0x1208>
   157b0:	040df693          	andi	a3,s11,64
   157b4:	10068a63          	beqz	a3,158c8 <_vfiprintf_r+0x1200>
   157b8:	01079793          	slli	a5,a5,0x10
   157bc:	0107d793          	srli	a5,a5,0x10
   157c0:	00000613          	li	a2,0
   157c4:	fa1ff06f          	j	15764 <_vfiprintf_r+0x109c>
   157c8:	00090513          	mv	a0,s2
   157cc:	e14fb0ef          	jal	10de0 <strlen>
   157d0:	00812803          	lw	a6,8(sp)
   157d4:	00050893          	mv	a7,a0
   157d8:	d98ff06f          	j	14d70 <_vfiprintf_r+0x6a8>
   157dc:	03b14703          	lbu	a4,59(sp)
   157e0:	1a070063          	beqz	a4,15980 <_vfiprintf_r+0x12b8>
   157e4:	03b10793          	addi	a5,sp,59
   157e8:	04f12623          	sw	a5,76(sp)
   157ec:	00100793          	li	a5,1
   157f0:	04f12823          	sw	a5,80(sp)
   157f4:	00100e13          	li	t3,1
   157f8:	000a8413          	mv	s0,s5
   157fc:	9e8ff06f          	j	149e4 <_vfiprintf_r+0x31c>
   15800:	02f12423          	sw	a5,40(sp)
   15804:	01812783          	lw	a5,24(sp)
   15808:	01c12583          	lw	a1,28(sp)
   1580c:	03112623          	sw	a7,44(sp)
   15810:	40f686b3          	sub	a3,a3,a5
   15814:	00078613          	mv	a2,a5
   15818:	00068513          	mv	a0,a3
   1581c:	02712223          	sw	t2,36(sp)
   15820:	03c12023          	sw	t3,32(sp)
   15824:	01012a23          	sw	a6,20(sp)
   15828:	01e12823          	sw	t5,16(sp)
   1582c:	00612623          	sw	t1,12(sp)
   15830:	00d12423          	sw	a3,8(sp)
   15834:	7c9000ef          	jal	167fc <strncpy>
   15838:	02812783          	lw	a5,40(sp)
   1583c:	00500613          	li	a2,5
   15840:	00c12303          	lw	t1,12(sp)
   15844:	00f93733          	sltu	a4,s2,a5
   15848:	00e90733          	add	a4,s2,a4
   1584c:	02c77733          	remu	a4,a4,a2
   15850:	00134603          	lbu	a2,1(t1)
   15854:	02c12883          	lw	a7,44(sp)
   15858:	ccccd5b7          	lui	a1,0xccccd
   1585c:	00c03633          	snez	a2,a2
   15860:	ccccd2b7          	lui	t0,0xccccd
   15864:	00c30333          	add	t1,t1,a2
   15868:	ccd58593          	addi	a1,a1,-819 # cccccccd <__BSS_END__+0xccca9ecd>
   1586c:	ccc28293          	addi	t0,t0,-820 # cccccccc <__BSS_END__+0xccca9ecc>
   15870:	00812683          	lw	a3,8(sp)
   15874:	01012f03          	lw	t5,16(sp)
   15878:	01412803          	lw	a6,20(sp)
   1587c:	02012e03          	lw	t3,32(sp)
   15880:	02412383          	lw	t2,36(sp)
   15884:	00000513          	li	a0,0
   15888:	00500e93          	li	t4,5
   1588c:	0ff00f93          	li	t6,255
   15890:	40e78733          	sub	a4,a5,a4
   15894:	00e7b633          	sltu	a2,a5,a4
   15898:	40c88633          	sub	a2,a7,a2
   1589c:	025702b3          	mul	t0,a4,t0
   158a0:	02b60633          	mul	a2,a2,a1
   158a4:	02b738b3          	mulhu	a7,a4,a1
   158a8:	00560633          	add	a2,a2,t0
   158ac:	02b707b3          	mul	a5,a4,a1
   158b0:	01160633          	add	a2,a2,a7
   158b4:	01f61713          	slli	a4,a2,0x1f
   158b8:	00165613          	srli	a2,a2,0x1
   158bc:	0017d793          	srli	a5,a5,0x1
   158c0:	00f76733          	or	a4,a4,a5
   158c4:	d81ff06f          	j	15644 <_vfiprintf_r+0xf7c>
   158c8:	200df693          	andi	a3,s11,512
   158cc:	0e069a63          	bnez	a3,159c0 <_vfiprintf_r+0x12f8>
   158d0:	00000613          	li	a2,0
   158d4:	e91ff06f          	j	15764 <_vfiprintf_r+0x109c>
   158d8:	00070d93          	mv	s11,a4
   158dc:	00000713          	li	a4,0
   158e0:	ec070463          	beqz	a4,14fa8 <_vfiprintf_r+0x8e0>
   158e4:	a79ff06f          	j	1535c <_vfiprintf_r+0xc94>
   158e8:	d7dfa0ef          	jal	10664 <__sinit>
   158ec:	e19fe06f          	j	14704 <_vfiprintf_r+0x3c>
   158f0:	0019c703          	lbu	a4,1(s3)
   158f4:	200ded93          	ori	s11,s11,512
   158f8:	00198993          	addi	s3,s3,1
   158fc:	f5dfe06f          	j	14858 <_vfiprintf_r+0x190>
   15900:	0019c703          	lbu	a4,1(s3)
   15904:	020ded93          	ori	s11,s11,32
   15908:	00198993          	addi	s3,s3,1
   1590c:	f4dfe06f          	j	14858 <_vfiprintf_r+0x190>
   15910:	00082783          	lw	a5,0(a6)
   15914:	00480813          	addi	a6,a6,4
   15918:	0187a023          	sw	s8,0(a5)
   1591c:	e9dfe06f          	j	147b8 <_vfiprintf_r+0xf0>
   15920:	00600793          	li	a5,6
   15924:	00048893          	mv	a7,s1
   15928:	0497e863          	bltu	a5,s1,15978 <_vfiprintf_r+0x12b0>
   1592c:	00088693          	mv	a3,a7
   15930:	0000b917          	auipc	s2,0xb
   15934:	02490913          	addi	s2,s2,36 # 20954 <_exit+0x11c>
   15938:	858ff06f          	j	14990 <_vfiprintf_r+0x2c8>
   1593c:	2006f713          	andi	a4,a3,512
   15940:	be0708e3          	beqz	a4,15530 <_vfiprintf_r+0xe68>
   15944:	0ff7f793          	zext.b	a5,a5
   15948:	00000893          	li	a7,0
   1594c:	e24ff06f          	j	14f70 <_vfiprintf_r+0x8a8>
   15950:	00000793          	li	a5,0
   15954:	00000613          	li	a2,0
   15958:	a05ff06f          	j	1535c <_vfiprintf_r+0xc94>
   1595c:	00100e13          	li	t3,1
   15960:	00000613          	li	a2,0
   15964:	000a8413          	mv	s0,s5
   15968:	900ff06f          	j	14a68 <_vfiprintf_r+0x3a0>
   1596c:	00900593          	li	a1,9
   15970:	ccf5eae3          	bltu	a1,a5,15644 <_vfiprintf_r+0xf7c>
   15974:	d51ff06f          	j	156c4 <_vfiprintf_r+0xffc>
   15978:	00600893          	li	a7,6
   1597c:	fb1ff06f          	j	1592c <_vfiprintf_r+0x1264>
   15980:	00000613          	li	a2,0
   15984:	00100e13          	li	t3,1
   15988:	000a8413          	mv	s0,s5
   1598c:	864ff06f          	j	149f0 <_vfiprintf_r+0x328>
   15990:	00082783          	lw	a5,0(a6)
   15994:	00480813          	addi	a6,a6,4
   15998:	01879023          	sh	s8,0(a5)
   1599c:	e1dfe06f          	j	147b8 <_vfiprintf_r+0xf0>
   159a0:	01879793          	slli	a5,a5,0x18
   159a4:	4187d793          	srai	a5,a5,0x18
   159a8:	41f7d893          	srai	a7,a5,0x1f
   159ac:	00088713          	mv	a4,a7
   159b0:	9a4ff06f          	j	14b54 <_vfiprintf_r+0x48c>
   159b4:	0ff7f793          	zext.b	a5,a5
   159b8:	00000613          	li	a2,0
   159bc:	d24ff06f          	j	14ee0 <_vfiprintf_r+0x818>
   159c0:	0ff7f793          	zext.b	a5,a5
   159c4:	00000613          	li	a2,0
   159c8:	d9dff06f          	j	15764 <_vfiprintf_r+0x109c>
   159cc:	03000793          	li	a5,48
   159d0:	b44ff06f          	j	14d14 <_vfiprintf_r+0x64c>
   159d4:	00058513          	mv	a0,a1
   159d8:	0000be17          	auipc	t3,0xb
   159dc:	494e0e13          	addi	t3,t3,1172 # 20e6c <blanks.1>
   159e0:	a95ff06f          	j	15474 <_vfiprintf_r+0xdac>
   159e4:	00160593          	addi	a1,a2,1
   159e8:	0000be17          	auipc	t3,0xb
   159ec:	484e0e13          	addi	t3,t3,1156 # 20e6c <blanks.1>
   159f0:	a78ff06f          	j	14c68 <_vfiprintf_r+0x5a0>
   159f4:	fff00d93          	li	s11,-1
   159f8:	f29fe06f          	j	14920 <_vfiprintf_r+0x258>
   159fc:	000e0593          	mv	a1,t3
   15a00:	0000be97          	auipc	t4,0xb
   15a04:	45ce8e93          	addi	t4,t4,1116 # 20e5c <zeroes.0>
   15a08:	f74ff06f          	j	1517c <_vfiprintf_r+0xab4>
   15a0c:	00082483          	lw	s1,0(a6)
   15a10:	00480813          	addi	a6,a6,4
   15a14:	0004d463          	bgez	s1,15a1c <_vfiprintf_r+0x1354>
   15a18:	fff00493          	li	s1,-1
   15a1c:	0019c703          	lbu	a4,1(s3)
   15a20:	00060993          	mv	s3,a2
   15a24:	e35fe06f          	j	14858 <_vfiprintf_r+0x190>

00015a28 <vfiprintf>:
   15a28:	00060693          	mv	a3,a2
   15a2c:	00058613          	mv	a2,a1
   15a30:	00050593          	mv	a1,a0
   15a34:	f5c1a503          	lw	a0,-164(gp) # 229dc <_impure_ptr>
   15a38:	c91fe06f          	j	146c8 <_vfiprintf_r>

00015a3c <__sbprintf>:
   15a3c:	b8010113          	addi	sp,sp,-1152
   15a40:	00c59783          	lh	a5,12(a1)
   15a44:	00e5d703          	lhu	a4,14(a1)
   15a48:	46812c23          	sw	s0,1144(sp)
   15a4c:	00058413          	mv	s0,a1
   15a50:	000105b7          	lui	a1,0x10
   15a54:	ffd58593          	addi	a1,a1,-3 # fffd <exit-0xb7>
   15a58:	06442e03          	lw	t3,100(s0)
   15a5c:	01c42303          	lw	t1,28(s0)
   15a60:	02442883          	lw	a7,36(s0)
   15a64:	01071713          	slli	a4,a4,0x10
   15a68:	00b7f7b3          	and	a5,a5,a1
   15a6c:	00e7e7b3          	or	a5,a5,a4
   15a70:	40000813          	li	a6,1024
   15a74:	00f12a23          	sw	a5,20(sp)
   15a78:	00810593          	addi	a1,sp,8
   15a7c:	07010793          	addi	a5,sp,112
   15a80:	46912a23          	sw	s1,1140(sp)
   15a84:	47212823          	sw	s2,1136(sp)
   15a88:	46112e23          	sw	ra,1148(sp)
   15a8c:	00050913          	mv	s2,a0
   15a90:	07c12623          	sw	t3,108(sp)
   15a94:	02612223          	sw	t1,36(sp)
   15a98:	03112623          	sw	a7,44(sp)
   15a9c:	00f12423          	sw	a5,8(sp)
   15aa0:	00f12c23          	sw	a5,24(sp)
   15aa4:	01012823          	sw	a6,16(sp)
   15aa8:	01012e23          	sw	a6,28(sp)
   15aac:	02012023          	sw	zero,32(sp)
   15ab0:	c19fe0ef          	jal	146c8 <_vfiprintf_r>
   15ab4:	00050493          	mv	s1,a0
   15ab8:	02055c63          	bgez	a0,15af0 <__sbprintf+0xb4>
   15abc:	01415783          	lhu	a5,20(sp)
   15ac0:	0407f793          	andi	a5,a5,64
   15ac4:	00078863          	beqz	a5,15ad4 <__sbprintf+0x98>
   15ac8:	00c45783          	lhu	a5,12(s0)
   15acc:	0407e793          	ori	a5,a5,64
   15ad0:	00f41623          	sh	a5,12(s0)
   15ad4:	47c12083          	lw	ra,1148(sp)
   15ad8:	47812403          	lw	s0,1144(sp)
   15adc:	47012903          	lw	s2,1136(sp)
   15ae0:	00048513          	mv	a0,s1
   15ae4:	47412483          	lw	s1,1140(sp)
   15ae8:	48010113          	addi	sp,sp,1152
   15aec:	00008067          	ret
   15af0:	00810593          	addi	a1,sp,8
   15af4:	00090513          	mv	a0,s2
   15af8:	370000ef          	jal	15e68 <_fflush_r>
   15afc:	fc0500e3          	beqz	a0,15abc <__sbprintf+0x80>
   15b00:	fff00493          	li	s1,-1
   15b04:	fb9ff06f          	j	15abc <__sbprintf+0x80>

00015b08 <_fclose_r>:
   15b08:	ff010113          	addi	sp,sp,-16
   15b0c:	00112623          	sw	ra,12(sp)
   15b10:	01212023          	sw	s2,0(sp)
   15b14:	02058863          	beqz	a1,15b44 <_fclose_r+0x3c>
   15b18:	00812423          	sw	s0,8(sp)
   15b1c:	00912223          	sw	s1,4(sp)
   15b20:	00058413          	mv	s0,a1
   15b24:	00050493          	mv	s1,a0
   15b28:	00050663          	beqz	a0,15b34 <_fclose_r+0x2c>
   15b2c:	03452783          	lw	a5,52(a0)
   15b30:	0c078c63          	beqz	a5,15c08 <_fclose_r+0x100>
   15b34:	00c41783          	lh	a5,12(s0)
   15b38:	02079263          	bnez	a5,15b5c <_fclose_r+0x54>
   15b3c:	00812403          	lw	s0,8(sp)
   15b40:	00412483          	lw	s1,4(sp)
   15b44:	00c12083          	lw	ra,12(sp)
   15b48:	00000913          	li	s2,0
   15b4c:	00090513          	mv	a0,s2
   15b50:	00012903          	lw	s2,0(sp)
   15b54:	01010113          	addi	sp,sp,16
   15b58:	00008067          	ret
   15b5c:	00040593          	mv	a1,s0
   15b60:	00048513          	mv	a0,s1
   15b64:	0b8000ef          	jal	15c1c <__sflush_r>
   15b68:	02c42783          	lw	a5,44(s0)
   15b6c:	00050913          	mv	s2,a0
   15b70:	00078a63          	beqz	a5,15b84 <_fclose_r+0x7c>
   15b74:	01c42583          	lw	a1,28(s0)
   15b78:	00048513          	mv	a0,s1
   15b7c:	000780e7          	jalr	a5
   15b80:	06054463          	bltz	a0,15be8 <_fclose_r+0xe0>
   15b84:	00c45783          	lhu	a5,12(s0)
   15b88:	0807f793          	andi	a5,a5,128
   15b8c:	06079663          	bnez	a5,15bf8 <_fclose_r+0xf0>
   15b90:	03042583          	lw	a1,48(s0)
   15b94:	00058c63          	beqz	a1,15bac <_fclose_r+0xa4>
   15b98:	04040793          	addi	a5,s0,64
   15b9c:	00f58663          	beq	a1,a5,15ba8 <_fclose_r+0xa0>
   15ba0:	00048513          	mv	a0,s1
   15ba4:	d4cfb0ef          	jal	110f0 <_free_r>
   15ba8:	02042823          	sw	zero,48(s0)
   15bac:	04442583          	lw	a1,68(s0)
   15bb0:	00058863          	beqz	a1,15bc0 <_fclose_r+0xb8>
   15bb4:	00048513          	mv	a0,s1
   15bb8:	d38fb0ef          	jal	110f0 <_free_r>
   15bbc:	04042223          	sw	zero,68(s0)
   15bc0:	ac9fa0ef          	jal	10688 <__sfp_lock_acquire>
   15bc4:	00041623          	sh	zero,12(s0)
   15bc8:	ac5fa0ef          	jal	1068c <__sfp_lock_release>
   15bcc:	00c12083          	lw	ra,12(sp)
   15bd0:	00812403          	lw	s0,8(sp)
   15bd4:	00412483          	lw	s1,4(sp)
   15bd8:	00090513          	mv	a0,s2
   15bdc:	00012903          	lw	s2,0(sp)
   15be0:	01010113          	addi	sp,sp,16
   15be4:	00008067          	ret
   15be8:	00c45783          	lhu	a5,12(s0)
   15bec:	fff00913          	li	s2,-1
   15bf0:	0807f793          	andi	a5,a5,128
   15bf4:	f8078ee3          	beqz	a5,15b90 <_fclose_r+0x88>
   15bf8:	01042583          	lw	a1,16(s0)
   15bfc:	00048513          	mv	a0,s1
   15c00:	cf0fb0ef          	jal	110f0 <_free_r>
   15c04:	f8dff06f          	j	15b90 <_fclose_r+0x88>
   15c08:	a5dfa0ef          	jal	10664 <__sinit>
   15c0c:	f29ff06f          	j	15b34 <_fclose_r+0x2c>

00015c10 <fclose>:
   15c10:	00050593          	mv	a1,a0
   15c14:	f5c1a503          	lw	a0,-164(gp) # 229dc <_impure_ptr>
   15c18:	ef1ff06f          	j	15b08 <_fclose_r>

00015c1c <__sflush_r>:
   15c1c:	00c59703          	lh	a4,12(a1)
   15c20:	fe010113          	addi	sp,sp,-32
   15c24:	00812c23          	sw	s0,24(sp)
   15c28:	01312623          	sw	s3,12(sp)
   15c2c:	00112e23          	sw	ra,28(sp)
   15c30:	00877793          	andi	a5,a4,8
   15c34:	00058413          	mv	s0,a1
   15c38:	00050993          	mv	s3,a0
   15c3c:	12079063          	bnez	a5,15d5c <__sflush_r+0x140>
   15c40:	000017b7          	lui	a5,0x1
   15c44:	80078793          	addi	a5,a5,-2048 # 800 <exit-0xf8b4>
   15c48:	0045a683          	lw	a3,4(a1)
   15c4c:	00f767b3          	or	a5,a4,a5
   15c50:	00f59623          	sh	a5,12(a1)
   15c54:	18d05263          	blez	a3,15dd8 <__sflush_r+0x1bc>
   15c58:	02842803          	lw	a6,40(s0)
   15c5c:	0e080463          	beqz	a6,15d44 <__sflush_r+0x128>
   15c60:	00912a23          	sw	s1,20(sp)
   15c64:	01371693          	slli	a3,a4,0x13
   15c68:	0009a483          	lw	s1,0(s3)
   15c6c:	0009a023          	sw	zero,0(s3)
   15c70:	01c42583          	lw	a1,28(s0)
   15c74:	1606ce63          	bltz	a3,15df0 <__sflush_r+0x1d4>
   15c78:	00000613          	li	a2,0
   15c7c:	00100693          	li	a3,1
   15c80:	00098513          	mv	a0,s3
   15c84:	000800e7          	jalr	a6
   15c88:	fff00793          	li	a5,-1
   15c8c:	00050613          	mv	a2,a0
   15c90:	1af50463          	beq	a0,a5,15e38 <__sflush_r+0x21c>
   15c94:	00c41783          	lh	a5,12(s0)
   15c98:	02842803          	lw	a6,40(s0)
   15c9c:	01c42583          	lw	a1,28(s0)
   15ca0:	0047f793          	andi	a5,a5,4
   15ca4:	00078e63          	beqz	a5,15cc0 <__sflush_r+0xa4>
   15ca8:	00442703          	lw	a4,4(s0)
   15cac:	03042783          	lw	a5,48(s0)
   15cb0:	40e60633          	sub	a2,a2,a4
   15cb4:	00078663          	beqz	a5,15cc0 <__sflush_r+0xa4>
   15cb8:	03c42783          	lw	a5,60(s0)
   15cbc:	40f60633          	sub	a2,a2,a5
   15cc0:	00000693          	li	a3,0
   15cc4:	00098513          	mv	a0,s3
   15cc8:	000800e7          	jalr	a6
   15ccc:	fff00793          	li	a5,-1
   15cd0:	12f51463          	bne	a0,a5,15df8 <__sflush_r+0x1dc>
   15cd4:	0009a683          	lw	a3,0(s3)
   15cd8:	01d00793          	li	a5,29
   15cdc:	00c41703          	lh	a4,12(s0)
   15ce0:	16d7ea63          	bltu	a5,a3,15e54 <__sflush_r+0x238>
   15ce4:	204007b7          	lui	a5,0x20400
   15ce8:	00178793          	addi	a5,a5,1 # 20400001 <__BSS_END__+0x203dd201>
   15cec:	00d7d7b3          	srl	a5,a5,a3
   15cf0:	0017f793          	andi	a5,a5,1
   15cf4:	16078063          	beqz	a5,15e54 <__sflush_r+0x238>
   15cf8:	01042603          	lw	a2,16(s0)
   15cfc:	fffff7b7          	lui	a5,0xfffff
   15d00:	7ff78793          	addi	a5,a5,2047 # fffff7ff <__BSS_END__+0xfffdc9ff>
   15d04:	00f777b3          	and	a5,a4,a5
   15d08:	00f41623          	sh	a5,12(s0)
   15d0c:	00042223          	sw	zero,4(s0)
   15d10:	00c42023          	sw	a2,0(s0)
   15d14:	01371793          	slli	a5,a4,0x13
   15d18:	0007d463          	bgez	a5,15d20 <__sflush_r+0x104>
   15d1c:	10068263          	beqz	a3,15e20 <__sflush_r+0x204>
   15d20:	03042583          	lw	a1,48(s0)
   15d24:	0099a023          	sw	s1,0(s3)
   15d28:	10058463          	beqz	a1,15e30 <__sflush_r+0x214>
   15d2c:	04040793          	addi	a5,s0,64
   15d30:	00f58663          	beq	a1,a5,15d3c <__sflush_r+0x120>
   15d34:	00098513          	mv	a0,s3
   15d38:	bb8fb0ef          	jal	110f0 <_free_r>
   15d3c:	01412483          	lw	s1,20(sp)
   15d40:	02042823          	sw	zero,48(s0)
   15d44:	00000513          	li	a0,0
   15d48:	01c12083          	lw	ra,28(sp)
   15d4c:	01812403          	lw	s0,24(sp)
   15d50:	00c12983          	lw	s3,12(sp)
   15d54:	02010113          	addi	sp,sp,32
   15d58:	00008067          	ret
   15d5c:	01212823          	sw	s2,16(sp)
   15d60:	0105a903          	lw	s2,16(a1)
   15d64:	08090263          	beqz	s2,15de8 <__sflush_r+0x1cc>
   15d68:	00912a23          	sw	s1,20(sp)
   15d6c:	0005a483          	lw	s1,0(a1)
   15d70:	00377713          	andi	a4,a4,3
   15d74:	0125a023          	sw	s2,0(a1)
   15d78:	412484b3          	sub	s1,s1,s2
   15d7c:	00000793          	li	a5,0
   15d80:	00071463          	bnez	a4,15d88 <__sflush_r+0x16c>
   15d84:	0145a783          	lw	a5,20(a1)
   15d88:	00f42423          	sw	a5,8(s0)
   15d8c:	00904863          	bgtz	s1,15d9c <__sflush_r+0x180>
   15d90:	0540006f          	j	15de4 <__sflush_r+0x1c8>
   15d94:	00a90933          	add	s2,s2,a0
   15d98:	04905663          	blez	s1,15de4 <__sflush_r+0x1c8>
   15d9c:	02442783          	lw	a5,36(s0)
   15da0:	01c42583          	lw	a1,28(s0)
   15da4:	00048693          	mv	a3,s1
   15da8:	00090613          	mv	a2,s2
   15dac:	00098513          	mv	a0,s3
   15db0:	000780e7          	jalr	a5
   15db4:	40a484b3          	sub	s1,s1,a0
   15db8:	fca04ee3          	bgtz	a0,15d94 <__sflush_r+0x178>
   15dbc:	00c41703          	lh	a4,12(s0)
   15dc0:	01012903          	lw	s2,16(sp)
   15dc4:	04076713          	ori	a4,a4,64
   15dc8:	01412483          	lw	s1,20(sp)
   15dcc:	00e41623          	sh	a4,12(s0)
   15dd0:	fff00513          	li	a0,-1
   15dd4:	f75ff06f          	j	15d48 <__sflush_r+0x12c>
   15dd8:	03c5a683          	lw	a3,60(a1)
   15ddc:	e6d04ee3          	bgtz	a3,15c58 <__sflush_r+0x3c>
   15de0:	f65ff06f          	j	15d44 <__sflush_r+0x128>
   15de4:	01412483          	lw	s1,20(sp)
   15de8:	01012903          	lw	s2,16(sp)
   15dec:	f59ff06f          	j	15d44 <__sflush_r+0x128>
   15df0:	05042603          	lw	a2,80(s0)
   15df4:	eadff06f          	j	15ca0 <__sflush_r+0x84>
   15df8:	00c41703          	lh	a4,12(s0)
   15dfc:	01042683          	lw	a3,16(s0)
   15e00:	fffff7b7          	lui	a5,0xfffff
   15e04:	7ff78793          	addi	a5,a5,2047 # fffff7ff <__BSS_END__+0xfffdc9ff>
   15e08:	00f777b3          	and	a5,a4,a5
   15e0c:	00f41623          	sh	a5,12(s0)
   15e10:	00042223          	sw	zero,4(s0)
   15e14:	00d42023          	sw	a3,0(s0)
   15e18:	01371793          	slli	a5,a4,0x13
   15e1c:	f007d2e3          	bgez	a5,15d20 <__sflush_r+0x104>
   15e20:	03042583          	lw	a1,48(s0)
   15e24:	04a42823          	sw	a0,80(s0)
   15e28:	0099a023          	sw	s1,0(s3)
   15e2c:	f00590e3          	bnez	a1,15d2c <__sflush_r+0x110>
   15e30:	01412483          	lw	s1,20(sp)
   15e34:	f11ff06f          	j	15d44 <__sflush_r+0x128>
   15e38:	0009a783          	lw	a5,0(s3)
   15e3c:	e4078ce3          	beqz	a5,15c94 <__sflush_r+0x78>
   15e40:	01d00713          	li	a4,29
   15e44:	00e78c63          	beq	a5,a4,15e5c <__sflush_r+0x240>
   15e48:	01600713          	li	a4,22
   15e4c:	00e78863          	beq	a5,a4,15e5c <__sflush_r+0x240>
   15e50:	00c41703          	lh	a4,12(s0)
   15e54:	04076713          	ori	a4,a4,64
   15e58:	f71ff06f          	j	15dc8 <__sflush_r+0x1ac>
   15e5c:	0099a023          	sw	s1,0(s3)
   15e60:	01412483          	lw	s1,20(sp)
   15e64:	ee1ff06f          	j	15d44 <__sflush_r+0x128>

00015e68 <_fflush_r>:
   15e68:	fe010113          	addi	sp,sp,-32
   15e6c:	00812c23          	sw	s0,24(sp)
   15e70:	00112e23          	sw	ra,28(sp)
   15e74:	00050413          	mv	s0,a0
   15e78:	00050663          	beqz	a0,15e84 <_fflush_r+0x1c>
   15e7c:	03452783          	lw	a5,52(a0)
   15e80:	02078a63          	beqz	a5,15eb4 <_fflush_r+0x4c>
   15e84:	00c59783          	lh	a5,12(a1)
   15e88:	00079c63          	bnez	a5,15ea0 <_fflush_r+0x38>
   15e8c:	01c12083          	lw	ra,28(sp)
   15e90:	01812403          	lw	s0,24(sp)
   15e94:	00000513          	li	a0,0
   15e98:	02010113          	addi	sp,sp,32
   15e9c:	00008067          	ret
   15ea0:	00040513          	mv	a0,s0
   15ea4:	01812403          	lw	s0,24(sp)
   15ea8:	01c12083          	lw	ra,28(sp)
   15eac:	02010113          	addi	sp,sp,32
   15eb0:	d6dff06f          	j	15c1c <__sflush_r>
   15eb4:	00b12623          	sw	a1,12(sp)
   15eb8:	facfa0ef          	jal	10664 <__sinit>
   15ebc:	00c12583          	lw	a1,12(sp)
   15ec0:	fc5ff06f          	j	15e84 <_fflush_r+0x1c>

00015ec4 <fflush>:
   15ec4:	06050063          	beqz	a0,15f24 <fflush+0x60>
   15ec8:	00050593          	mv	a1,a0
   15ecc:	f5c1a503          	lw	a0,-164(gp) # 229dc <_impure_ptr>
   15ed0:	00050663          	beqz	a0,15edc <fflush+0x18>
   15ed4:	03452783          	lw	a5,52(a0)
   15ed8:	00078c63          	beqz	a5,15ef0 <fflush+0x2c>
   15edc:	00c59783          	lh	a5,12(a1)
   15ee0:	00079663          	bnez	a5,15eec <fflush+0x28>
   15ee4:	00000513          	li	a0,0
   15ee8:	00008067          	ret
   15eec:	d31ff06f          	j	15c1c <__sflush_r>
   15ef0:	fe010113          	addi	sp,sp,-32
   15ef4:	00b12623          	sw	a1,12(sp)
   15ef8:	00a12423          	sw	a0,8(sp)
   15efc:	00112e23          	sw	ra,28(sp)
   15f00:	f64fa0ef          	jal	10664 <__sinit>
   15f04:	00c12583          	lw	a1,12(sp)
   15f08:	00812503          	lw	a0,8(sp)
   15f0c:	00c59783          	lh	a5,12(a1)
   15f10:	02079863          	bnez	a5,15f40 <fflush+0x7c>
   15f14:	01c12083          	lw	ra,28(sp)
   15f18:	00000513          	li	a0,0
   15f1c:	02010113          	addi	sp,sp,32
   15f20:	00008067          	ret
   15f24:	0000c617          	auipc	a2,0xc
   15f28:	3ac60613          	addi	a2,a2,940 # 222d0 <__sglue>
   15f2c:	00000597          	auipc	a1,0x0
   15f30:	f3c58593          	addi	a1,a1,-196 # 15e68 <_fflush_r>
   15f34:	0000c517          	auipc	a0,0xc
   15f38:	3ac50513          	addi	a0,a0,940 # 222e0 <_impure_data>
   15f3c:	f84fa06f          	j	106c0 <_fwalk_sglue>
   15f40:	01c12083          	lw	ra,28(sp)
   15f44:	02010113          	addi	sp,sp,32
   15f48:	cd5ff06f          	j	15c1c <__sflush_r>

00015f4c <__sfvwrite_r>:
   15f4c:	00862783          	lw	a5,8(a2)
   15f50:	2c078463          	beqz	a5,16218 <__sfvwrite_r+0x2cc>
   15f54:	00c59683          	lh	a3,12(a1)
   15f58:	fd010113          	addi	sp,sp,-48
   15f5c:	02812423          	sw	s0,40(sp)
   15f60:	01412c23          	sw	s4,24(sp)
   15f64:	01612823          	sw	s6,16(sp)
   15f68:	02112623          	sw	ra,44(sp)
   15f6c:	0086f793          	andi	a5,a3,8
   15f70:	00060b13          	mv	s6,a2
   15f74:	00050a13          	mv	s4,a0
   15f78:	00058413          	mv	s0,a1
   15f7c:	08078e63          	beqz	a5,16018 <__sfvwrite_r+0xcc>
   15f80:	0105a783          	lw	a5,16(a1)
   15f84:	08078a63          	beqz	a5,16018 <__sfvwrite_r+0xcc>
   15f88:	02912223          	sw	s1,36(sp)
   15f8c:	03212023          	sw	s2,32(sp)
   15f90:	01312e23          	sw	s3,28(sp)
   15f94:	01512a23          	sw	s5,20(sp)
   15f98:	0026f793          	andi	a5,a3,2
   15f9c:	000b2483          	lw	s1,0(s6)
   15fa0:	0a078463          	beqz	a5,16048 <__sfvwrite_r+0xfc>
   15fa4:	02442783          	lw	a5,36(s0)
   15fa8:	01c42583          	lw	a1,28(s0)
   15fac:	80000ab7          	lui	s5,0x80000
   15fb0:	00000993          	li	s3,0
   15fb4:	00000913          	li	s2,0
   15fb8:	c00a8a93          	addi	s5,s5,-1024 # 7ffffc00 <__BSS_END__+0x7ffdce00>
   15fbc:	00098613          	mv	a2,s3
   15fc0:	000a0513          	mv	a0,s4
   15fc4:	04090263          	beqz	s2,16008 <__sfvwrite_r+0xbc>
   15fc8:	00090693          	mv	a3,s2
   15fcc:	012af463          	bgeu	s5,s2,15fd4 <__sfvwrite_r+0x88>
   15fd0:	000a8693          	mv	a3,s5
   15fd4:	000780e7          	jalr	a5
   15fd8:	46a05063          	blez	a0,16438 <__sfvwrite_r+0x4ec>
   15fdc:	008b2783          	lw	a5,8(s6)
   15fe0:	00a989b3          	add	s3,s3,a0
   15fe4:	40a90933          	sub	s2,s2,a0
   15fe8:	40a787b3          	sub	a5,a5,a0
   15fec:	00fb2423          	sw	a5,8(s6)
   15ff0:	1a078663          	beqz	a5,1619c <__sfvwrite_r+0x250>
   15ff4:	02442783          	lw	a5,36(s0)
   15ff8:	01c42583          	lw	a1,28(s0)
   15ffc:	00098613          	mv	a2,s3
   16000:	000a0513          	mv	a0,s4
   16004:	fc0912e3          	bnez	s2,15fc8 <__sfvwrite_r+0x7c>
   16008:	0004a983          	lw	s3,0(s1)
   1600c:	0044a903          	lw	s2,4(s1)
   16010:	00848493          	addi	s1,s1,8
   16014:	fa9ff06f          	j	15fbc <__sfvwrite_r+0x70>
   16018:	00040593          	mv	a1,s0
   1601c:	000a0513          	mv	a0,s4
   16020:	43c000ef          	jal	1645c <__swsetup_r>
   16024:	1c051c63          	bnez	a0,161fc <__sfvwrite_r+0x2b0>
   16028:	00c41683          	lh	a3,12(s0)
   1602c:	02912223          	sw	s1,36(sp)
   16030:	03212023          	sw	s2,32(sp)
   16034:	01312e23          	sw	s3,28(sp)
   16038:	01512a23          	sw	s5,20(sp)
   1603c:	0026f793          	andi	a5,a3,2
   16040:	000b2483          	lw	s1,0(s6)
   16044:	f60790e3          	bnez	a5,15fa4 <__sfvwrite_r+0x58>
   16048:	01712623          	sw	s7,12(sp)
   1604c:	01812423          	sw	s8,8(sp)
   16050:	0016f793          	andi	a5,a3,1
   16054:	1c079663          	bnez	a5,16220 <__sfvwrite_r+0x2d4>
   16058:	00042783          	lw	a5,0(s0)
   1605c:	00842703          	lw	a4,8(s0)
   16060:	80000ab7          	lui	s5,0x80000
   16064:	01912223          	sw	s9,4(sp)
   16068:	00000b93          	li	s7,0
   1606c:	00000993          	li	s3,0
   16070:	fffa8a93          	addi	s5,s5,-1 # 7fffffff <__BSS_END__+0x7ffdd1ff>
   16074:	00078513          	mv	a0,a5
   16078:	00070c13          	mv	s8,a4
   1607c:	10098263          	beqz	s3,16180 <__sfvwrite_r+0x234>
   16080:	2006f613          	andi	a2,a3,512
   16084:	28060863          	beqz	a2,16314 <__sfvwrite_r+0x3c8>
   16088:	00070c93          	mv	s9,a4
   1608c:	32e9e663          	bltu	s3,a4,163b8 <__sfvwrite_r+0x46c>
   16090:	4806f713          	andi	a4,a3,1152
   16094:	08070a63          	beqz	a4,16128 <__sfvwrite_r+0x1dc>
   16098:	01442603          	lw	a2,20(s0)
   1609c:	01042583          	lw	a1,16(s0)
   160a0:	00161713          	slli	a4,a2,0x1
   160a4:	00c70733          	add	a4,a4,a2
   160a8:	40b78933          	sub	s2,a5,a1
   160ac:	01f75c13          	srli	s8,a4,0x1f
   160b0:	00ec0c33          	add	s8,s8,a4
   160b4:	00190793          	addi	a5,s2,1
   160b8:	401c5c13          	srai	s8,s8,0x1
   160bc:	013787b3          	add	a5,a5,s3
   160c0:	000c0613          	mv	a2,s8
   160c4:	00fc7663          	bgeu	s8,a5,160d0 <__sfvwrite_r+0x184>
   160c8:	00078c13          	mv	s8,a5
   160cc:	00078613          	mv	a2,a5
   160d0:	4006f693          	andi	a3,a3,1024
   160d4:	30068e63          	beqz	a3,163f0 <__sfvwrite_r+0x4a4>
   160d8:	00060593          	mv	a1,a2
   160dc:	000a0513          	mv	a0,s4
   160e0:	b14fb0ef          	jal	113f4 <_malloc_r>
   160e4:	00050c93          	mv	s9,a0
   160e8:	34050c63          	beqz	a0,16440 <__sfvwrite_r+0x4f4>
   160ec:	01042583          	lw	a1,16(s0)
   160f0:	00090613          	mv	a2,s2
   160f4:	285000ef          	jal	16b78 <memcpy>
   160f8:	00c45783          	lhu	a5,12(s0)
   160fc:	b7f7f793          	andi	a5,a5,-1153
   16100:	0807e793          	ori	a5,a5,128
   16104:	00f41623          	sh	a5,12(s0)
   16108:	012c8533          	add	a0,s9,s2
   1610c:	412c07b3          	sub	a5,s8,s2
   16110:	01942823          	sw	s9,16(s0)
   16114:	01842a23          	sw	s8,20(s0)
   16118:	00a42023          	sw	a0,0(s0)
   1611c:	00098c13          	mv	s8,s3
   16120:	00f42423          	sw	a5,8(s0)
   16124:	00098c93          	mv	s9,s3
   16128:	000c8613          	mv	a2,s9
   1612c:	000b8593          	mv	a1,s7
   16130:	13d000ef          	jal	16a6c <memmove>
   16134:	00842703          	lw	a4,8(s0)
   16138:	00042783          	lw	a5,0(s0)
   1613c:	00098913          	mv	s2,s3
   16140:	41870733          	sub	a4,a4,s8
   16144:	019787b3          	add	a5,a5,s9
   16148:	00e42423          	sw	a4,8(s0)
   1614c:	00f42023          	sw	a5,0(s0)
   16150:	00000993          	li	s3,0
   16154:	008b2783          	lw	a5,8(s6)
   16158:	012b8bb3          	add	s7,s7,s2
   1615c:	412787b3          	sub	a5,a5,s2
   16160:	00fb2423          	sw	a5,8(s6)
   16164:	02078663          	beqz	a5,16190 <__sfvwrite_r+0x244>
   16168:	00042783          	lw	a5,0(s0)
   1616c:	00842703          	lw	a4,8(s0)
   16170:	00c41683          	lh	a3,12(s0)
   16174:	00078513          	mv	a0,a5
   16178:	00070c13          	mv	s8,a4
   1617c:	f00992e3          	bnez	s3,16080 <__sfvwrite_r+0x134>
   16180:	0004ab83          	lw	s7,0(s1)
   16184:	0044a983          	lw	s3,4(s1)
   16188:	00848493          	addi	s1,s1,8
   1618c:	ee9ff06f          	j	16074 <__sfvwrite_r+0x128>
   16190:	00c12b83          	lw	s7,12(sp)
   16194:	00812c03          	lw	s8,8(sp)
   16198:	00412c83          	lw	s9,4(sp)
   1619c:	02c12083          	lw	ra,44(sp)
   161a0:	02812403          	lw	s0,40(sp)
   161a4:	02412483          	lw	s1,36(sp)
   161a8:	02012903          	lw	s2,32(sp)
   161ac:	01c12983          	lw	s3,28(sp)
   161b0:	01412a83          	lw	s5,20(sp)
   161b4:	01812a03          	lw	s4,24(sp)
   161b8:	01012b03          	lw	s6,16(sp)
   161bc:	00000513          	li	a0,0
   161c0:	03010113          	addi	sp,sp,48
   161c4:	00008067          	ret
   161c8:	00040593          	mv	a1,s0
   161cc:	000a0513          	mv	a0,s4
   161d0:	c99ff0ef          	jal	15e68 <_fflush_r>
   161d4:	0a050e63          	beqz	a0,16290 <__sfvwrite_r+0x344>
   161d8:	00c41783          	lh	a5,12(s0)
   161dc:	00c12b83          	lw	s7,12(sp)
   161e0:	00812c03          	lw	s8,8(sp)
   161e4:	02412483          	lw	s1,36(sp)
   161e8:	02012903          	lw	s2,32(sp)
   161ec:	01c12983          	lw	s3,28(sp)
   161f0:	01412a83          	lw	s5,20(sp)
   161f4:	0407e793          	ori	a5,a5,64
   161f8:	00f41623          	sh	a5,12(s0)
   161fc:	02c12083          	lw	ra,44(sp)
   16200:	02812403          	lw	s0,40(sp)
   16204:	01812a03          	lw	s4,24(sp)
   16208:	01012b03          	lw	s6,16(sp)
   1620c:	fff00513          	li	a0,-1
   16210:	03010113          	addi	sp,sp,48
   16214:	00008067          	ret
   16218:	00000513          	li	a0,0
   1621c:	00008067          	ret
   16220:	00000a93          	li	s5,0
   16224:	00000513          	li	a0,0
   16228:	00000c13          	li	s8,0
   1622c:	00000993          	li	s3,0
   16230:	08098263          	beqz	s3,162b4 <__sfvwrite_r+0x368>
   16234:	08050a63          	beqz	a0,162c8 <__sfvwrite_r+0x37c>
   16238:	000a8793          	mv	a5,s5
   1623c:	00098b93          	mv	s7,s3
   16240:	0137f463          	bgeu	a5,s3,16248 <__sfvwrite_r+0x2fc>
   16244:	00078b93          	mv	s7,a5
   16248:	00042503          	lw	a0,0(s0)
   1624c:	01042783          	lw	a5,16(s0)
   16250:	00842903          	lw	s2,8(s0)
   16254:	01442683          	lw	a3,20(s0)
   16258:	00a7f663          	bgeu	a5,a0,16264 <__sfvwrite_r+0x318>
   1625c:	00d90933          	add	s2,s2,a3
   16260:	09794463          	blt	s2,s7,162e8 <__sfvwrite_r+0x39c>
   16264:	16dbc063          	blt	s7,a3,163c4 <__sfvwrite_r+0x478>
   16268:	02442783          	lw	a5,36(s0)
   1626c:	01c42583          	lw	a1,28(s0)
   16270:	000c0613          	mv	a2,s8
   16274:	000a0513          	mv	a0,s4
   16278:	000780e7          	jalr	a5
   1627c:	00050913          	mv	s2,a0
   16280:	f4a05ce3          	blez	a0,161d8 <__sfvwrite_r+0x28c>
   16284:	412a8ab3          	sub	s5,s5,s2
   16288:	00100513          	li	a0,1
   1628c:	f20a8ee3          	beqz	s5,161c8 <__sfvwrite_r+0x27c>
   16290:	008b2783          	lw	a5,8(s6)
   16294:	012c0c33          	add	s8,s8,s2
   16298:	412989b3          	sub	s3,s3,s2
   1629c:	412787b3          	sub	a5,a5,s2
   162a0:	00fb2423          	sw	a5,8(s6)
   162a4:	f80796e3          	bnez	a5,16230 <__sfvwrite_r+0x2e4>
   162a8:	00c12b83          	lw	s7,12(sp)
   162ac:	00812c03          	lw	s8,8(sp)
   162b0:	eedff06f          	j	1619c <__sfvwrite_r+0x250>
   162b4:	0044a983          	lw	s3,4(s1)
   162b8:	00048793          	mv	a5,s1
   162bc:	00848493          	addi	s1,s1,8
   162c0:	fe098ae3          	beqz	s3,162b4 <__sfvwrite_r+0x368>
   162c4:	0007ac03          	lw	s8,0(a5)
   162c8:	00098613          	mv	a2,s3
   162cc:	00a00593          	li	a1,10
   162d0:	000c0513          	mv	a0,s8
   162d4:	464000ef          	jal	16738 <memchr>
   162d8:	14050a63          	beqz	a0,1642c <__sfvwrite_r+0x4e0>
   162dc:	00150513          	addi	a0,a0,1
   162e0:	41850ab3          	sub	s5,a0,s8
   162e4:	f55ff06f          	j	16238 <__sfvwrite_r+0x2ec>
   162e8:	000c0593          	mv	a1,s8
   162ec:	00090613          	mv	a2,s2
   162f0:	77c000ef          	jal	16a6c <memmove>
   162f4:	00042783          	lw	a5,0(s0)
   162f8:	00040593          	mv	a1,s0
   162fc:	000a0513          	mv	a0,s4
   16300:	012787b3          	add	a5,a5,s2
   16304:	00f42023          	sw	a5,0(s0)
   16308:	b61ff0ef          	jal	15e68 <_fflush_r>
   1630c:	f6050ce3          	beqz	a0,16284 <__sfvwrite_r+0x338>
   16310:	ec9ff06f          	j	161d8 <__sfvwrite_r+0x28c>
   16314:	01042683          	lw	a3,16(s0)
   16318:	04f6e263          	bltu	a3,a5,1635c <__sfvwrite_r+0x410>
   1631c:	01442603          	lw	a2,20(s0)
   16320:	02c9ee63          	bltu	s3,a2,1635c <__sfvwrite_r+0x410>
   16324:	00098793          	mv	a5,s3
   16328:	013af463          	bgeu	s5,s3,16330 <__sfvwrite_r+0x3e4>
   1632c:	000a8793          	mv	a5,s5
   16330:	02c7e6b3          	rem	a3,a5,a2
   16334:	02442703          	lw	a4,36(s0)
   16338:	01c42583          	lw	a1,28(s0)
   1633c:	000b8613          	mv	a2,s7
   16340:	000a0513          	mv	a0,s4
   16344:	40d786b3          	sub	a3,a5,a3
   16348:	000700e7          	jalr	a4
   1634c:	00050913          	mv	s2,a0
   16350:	04a05a63          	blez	a0,163a4 <__sfvwrite_r+0x458>
   16354:	412989b3          	sub	s3,s3,s2
   16358:	dfdff06f          	j	16154 <__sfvwrite_r+0x208>
   1635c:	00070913          	mv	s2,a4
   16360:	00e9f463          	bgeu	s3,a4,16368 <__sfvwrite_r+0x41c>
   16364:	00098913          	mv	s2,s3
   16368:	00078513          	mv	a0,a5
   1636c:	00090613          	mv	a2,s2
   16370:	000b8593          	mv	a1,s7
   16374:	6f8000ef          	jal	16a6c <memmove>
   16378:	00842703          	lw	a4,8(s0)
   1637c:	00042783          	lw	a5,0(s0)
   16380:	41270733          	sub	a4,a4,s2
   16384:	012787b3          	add	a5,a5,s2
   16388:	00e42423          	sw	a4,8(s0)
   1638c:	00f42023          	sw	a5,0(s0)
   16390:	fc0712e3          	bnez	a4,16354 <__sfvwrite_r+0x408>
   16394:	00040593          	mv	a1,s0
   16398:	000a0513          	mv	a0,s4
   1639c:	acdff0ef          	jal	15e68 <_fflush_r>
   163a0:	fa050ae3          	beqz	a0,16354 <__sfvwrite_r+0x408>
   163a4:	00c41783          	lh	a5,12(s0)
   163a8:	00c12b83          	lw	s7,12(sp)
   163ac:	00812c03          	lw	s8,8(sp)
   163b0:	00412c83          	lw	s9,4(sp)
   163b4:	e31ff06f          	j	161e4 <__sfvwrite_r+0x298>
   163b8:	00098c13          	mv	s8,s3
   163bc:	00098c93          	mv	s9,s3
   163c0:	d69ff06f          	j	16128 <__sfvwrite_r+0x1dc>
   163c4:	000b8613          	mv	a2,s7
   163c8:	000c0593          	mv	a1,s8
   163cc:	6a0000ef          	jal	16a6c <memmove>
   163d0:	00842703          	lw	a4,8(s0)
   163d4:	00042783          	lw	a5,0(s0)
   163d8:	000b8913          	mv	s2,s7
   163dc:	41770733          	sub	a4,a4,s7
   163e0:	017787b3          	add	a5,a5,s7
   163e4:	00e42423          	sw	a4,8(s0)
   163e8:	00f42023          	sw	a5,0(s0)
   163ec:	e99ff06f          	j	16284 <__sfvwrite_r+0x338>
   163f0:	000a0513          	mv	a0,s4
   163f4:	440040ef          	jal	1a834 <_realloc_r>
   163f8:	00050c93          	mv	s9,a0
   163fc:	d00516e3          	bnez	a0,16108 <__sfvwrite_r+0x1bc>
   16400:	01042583          	lw	a1,16(s0)
   16404:	000a0513          	mv	a0,s4
   16408:	ce9fa0ef          	jal	110f0 <_free_r>
   1640c:	00c41783          	lh	a5,12(s0)
   16410:	00c00713          	li	a4,12
   16414:	00c12b83          	lw	s7,12(sp)
   16418:	00812c03          	lw	s8,8(sp)
   1641c:	00412c83          	lw	s9,4(sp)
   16420:	00ea2023          	sw	a4,0(s4)
   16424:	f7f7f793          	andi	a5,a5,-129
   16428:	dbdff06f          	j	161e4 <__sfvwrite_r+0x298>
   1642c:	00198793          	addi	a5,s3,1
   16430:	00078a93          	mv	s5,a5
   16434:	e09ff06f          	j	1623c <__sfvwrite_r+0x2f0>
   16438:	00c41783          	lh	a5,12(s0)
   1643c:	da9ff06f          	j	161e4 <__sfvwrite_r+0x298>
   16440:	00c00713          	li	a4,12
   16444:	00c41783          	lh	a5,12(s0)
   16448:	00c12b83          	lw	s7,12(sp)
   1644c:	00812c03          	lw	s8,8(sp)
   16450:	00412c83          	lw	s9,4(sp)
   16454:	00ea2023          	sw	a4,0(s4)
   16458:	d8dff06f          	j	161e4 <__sfvwrite_r+0x298>

0001645c <__swsetup_r>:
   1645c:	ff010113          	addi	sp,sp,-16
   16460:	00812423          	sw	s0,8(sp)
   16464:	00912223          	sw	s1,4(sp)
   16468:	00112623          	sw	ra,12(sp)
   1646c:	f5c1a783          	lw	a5,-164(gp) # 229dc <_impure_ptr>
   16470:	00050493          	mv	s1,a0
   16474:	00058413          	mv	s0,a1
   16478:	00078663          	beqz	a5,16484 <__swsetup_r+0x28>
   1647c:	0347a703          	lw	a4,52(a5)
   16480:	0e070c63          	beqz	a4,16578 <__swsetup_r+0x11c>
   16484:	00c41783          	lh	a5,12(s0)
   16488:	0087f713          	andi	a4,a5,8
   1648c:	06070a63          	beqz	a4,16500 <__swsetup_r+0xa4>
   16490:	01042703          	lw	a4,16(s0)
   16494:	08070663          	beqz	a4,16520 <__swsetup_r+0xc4>
   16498:	0017f693          	andi	a3,a5,1
   1649c:	02068863          	beqz	a3,164cc <__swsetup_r+0x70>
   164a0:	01442683          	lw	a3,20(s0)
   164a4:	00042423          	sw	zero,8(s0)
   164a8:	00000513          	li	a0,0
   164ac:	40d006b3          	neg	a3,a3
   164b0:	00d42c23          	sw	a3,24(s0)
   164b4:	02070a63          	beqz	a4,164e8 <__swsetup_r+0x8c>
   164b8:	00c12083          	lw	ra,12(sp)
   164bc:	00812403          	lw	s0,8(sp)
   164c0:	00412483          	lw	s1,4(sp)
   164c4:	01010113          	addi	sp,sp,16
   164c8:	00008067          	ret
   164cc:	0027f693          	andi	a3,a5,2
   164d0:	00000613          	li	a2,0
   164d4:	00069463          	bnez	a3,164dc <__swsetup_r+0x80>
   164d8:	01442603          	lw	a2,20(s0)
   164dc:	00c42423          	sw	a2,8(s0)
   164e0:	00000513          	li	a0,0
   164e4:	fc071ae3          	bnez	a4,164b8 <__swsetup_r+0x5c>
   164e8:	0807f713          	andi	a4,a5,128
   164ec:	fc0706e3          	beqz	a4,164b8 <__swsetup_r+0x5c>
   164f0:	0407e793          	ori	a5,a5,64
   164f4:	00f41623          	sh	a5,12(s0)
   164f8:	fff00513          	li	a0,-1
   164fc:	fbdff06f          	j	164b8 <__swsetup_r+0x5c>
   16500:	0107f713          	andi	a4,a5,16
   16504:	08070063          	beqz	a4,16584 <__swsetup_r+0x128>
   16508:	0047f713          	andi	a4,a5,4
   1650c:	02071c63          	bnez	a4,16544 <__swsetup_r+0xe8>
   16510:	01042703          	lw	a4,16(s0)
   16514:	0087e793          	ori	a5,a5,8
   16518:	00f41623          	sh	a5,12(s0)
   1651c:	f6071ee3          	bnez	a4,16498 <__swsetup_r+0x3c>
   16520:	2807f693          	andi	a3,a5,640
   16524:	20000613          	li	a2,512
   16528:	f6c688e3          	beq	a3,a2,16498 <__swsetup_r+0x3c>
   1652c:	00040593          	mv	a1,s0
   16530:	00048513          	mv	a0,s1
   16534:	1a9040ef          	jal	1aedc <__smakebuf_r>
   16538:	00c41783          	lh	a5,12(s0)
   1653c:	01042703          	lw	a4,16(s0)
   16540:	f59ff06f          	j	16498 <__swsetup_r+0x3c>
   16544:	03042583          	lw	a1,48(s0)
   16548:	00058e63          	beqz	a1,16564 <__swsetup_r+0x108>
   1654c:	04040713          	addi	a4,s0,64
   16550:	00e58863          	beq	a1,a4,16560 <__swsetup_r+0x104>
   16554:	00048513          	mv	a0,s1
   16558:	b99fa0ef          	jal	110f0 <_free_r>
   1655c:	00c41783          	lh	a5,12(s0)
   16560:	02042823          	sw	zero,48(s0)
   16564:	01042703          	lw	a4,16(s0)
   16568:	fdb7f793          	andi	a5,a5,-37
   1656c:	00042223          	sw	zero,4(s0)
   16570:	00e42023          	sw	a4,0(s0)
   16574:	fa1ff06f          	j	16514 <__swsetup_r+0xb8>
   16578:	00078513          	mv	a0,a5
   1657c:	8e8fa0ef          	jal	10664 <__sinit>
   16580:	f05ff06f          	j	16484 <__swsetup_r+0x28>
   16584:	00900713          	li	a4,9
   16588:	00e4a023          	sw	a4,0(s1)
   1658c:	0407e793          	ori	a5,a5,64
   16590:	00f41623          	sh	a5,12(s0)
   16594:	fff00513          	li	a0,-1
   16598:	f21ff06f          	j	164b8 <__swsetup_r+0x5c>

0001659c <__fputwc>:
   1659c:	fe010113          	addi	sp,sp,-32
   165a0:	00812c23          	sw	s0,24(sp)
   165a4:	00912a23          	sw	s1,20(sp)
   165a8:	01212823          	sw	s2,16(sp)
   165ac:	00112e23          	sw	ra,28(sp)
   165b0:	00050913          	mv	s2,a0
   165b4:	00058493          	mv	s1,a1
   165b8:	00060413          	mv	s0,a2
   165bc:	368000ef          	jal	16924 <__locale_mb_cur_max>
   165c0:	00100793          	li	a5,1
   165c4:	00f51c63          	bne	a0,a5,165dc <__fputwc+0x40>
   165c8:	fff48793          	addi	a5,s1,-1
   165cc:	0fe00713          	li	a4,254
   165d0:	00f76663          	bltu	a4,a5,165dc <__fputwc+0x40>
   165d4:	00910623          	sb	s1,12(sp)
   165d8:	0240006f          	j	165fc <__fputwc+0x60>
   165dc:	05c40693          	addi	a3,s0,92
   165e0:	00048613          	mv	a2,s1
   165e4:	00c10593          	addi	a1,sp,12
   165e8:	00090513          	mv	a0,s2
   165ec:	7f0040ef          	jal	1addc <_wcrtomb_r>
   165f0:	fff00793          	li	a5,-1
   165f4:	08f50463          	beq	a0,a5,1667c <__fputwc+0xe0>
   165f8:	02050c63          	beqz	a0,16630 <__fputwc+0x94>
   165fc:	00842783          	lw	a5,8(s0)
   16600:	00c14583          	lbu	a1,12(sp)
   16604:	fff78793          	addi	a5,a5,-1
   16608:	00f42423          	sw	a5,8(s0)
   1660c:	0007da63          	bgez	a5,16620 <__fputwc+0x84>
   16610:	01842703          	lw	a4,24(s0)
   16614:	02e7cc63          	blt	a5,a4,1664c <__fputwc+0xb0>
   16618:	00a00793          	li	a5,10
   1661c:	02f58863          	beq	a1,a5,1664c <__fputwc+0xb0>
   16620:	00042783          	lw	a5,0(s0)
   16624:	00178713          	addi	a4,a5,1
   16628:	00e42023          	sw	a4,0(s0)
   1662c:	00b78023          	sb	a1,0(a5)
   16630:	01c12083          	lw	ra,28(sp)
   16634:	01812403          	lw	s0,24(sp)
   16638:	01012903          	lw	s2,16(sp)
   1663c:	00048513          	mv	a0,s1
   16640:	01412483          	lw	s1,20(sp)
   16644:	02010113          	addi	sp,sp,32
   16648:	00008067          	ret
   1664c:	00040613          	mv	a2,s0
   16650:	00090513          	mv	a0,s2
   16654:	2f9040ef          	jal	1b14c <__swbuf_r>
   16658:	fff00793          	li	a5,-1
   1665c:	fcf51ae3          	bne	a0,a5,16630 <__fputwc+0x94>
   16660:	fff00513          	li	a0,-1
   16664:	01c12083          	lw	ra,28(sp)
   16668:	01812403          	lw	s0,24(sp)
   1666c:	01412483          	lw	s1,20(sp)
   16670:	01012903          	lw	s2,16(sp)
   16674:	02010113          	addi	sp,sp,32
   16678:	00008067          	ret
   1667c:	00c45783          	lhu	a5,12(s0)
   16680:	fff00513          	li	a0,-1
   16684:	0407e793          	ori	a5,a5,64
   16688:	00f41623          	sh	a5,12(s0)
   1668c:	fd9ff06f          	j	16664 <__fputwc+0xc8>

00016690 <_fputwc_r>:
   16690:	00c61783          	lh	a5,12(a2)
   16694:	01279713          	slli	a4,a5,0x12
   16698:	02074063          	bltz	a4,166b8 <_fputwc_r+0x28>
   1669c:	06462703          	lw	a4,100(a2)
   166a0:	000026b7          	lui	a3,0x2
   166a4:	00d7e7b3          	or	a5,a5,a3
   166a8:	000026b7          	lui	a3,0x2
   166ac:	00d76733          	or	a4,a4,a3
   166b0:	00f61623          	sh	a5,12(a2)
   166b4:	06e62223          	sw	a4,100(a2)
   166b8:	ee5ff06f          	j	1659c <__fputwc>

000166bc <fputwc>:
   166bc:	fe010113          	addi	sp,sp,-32
   166c0:	00812c23          	sw	s0,24(sp)
   166c4:	00112e23          	sw	ra,28(sp)
   166c8:	f5c1a403          	lw	s0,-164(gp) # 229dc <_impure_ptr>
   166cc:	00058613          	mv	a2,a1
   166d0:	00050593          	mv	a1,a0
   166d4:	00040663          	beqz	s0,166e0 <fputwc+0x24>
   166d8:	03442783          	lw	a5,52(s0)
   166dc:	04078063          	beqz	a5,1671c <fputwc+0x60>
   166e0:	00c61783          	lh	a5,12(a2)
   166e4:	01279713          	slli	a4,a5,0x12
   166e8:	02074063          	bltz	a4,16708 <fputwc+0x4c>
   166ec:	06462703          	lw	a4,100(a2)
   166f0:	000026b7          	lui	a3,0x2
   166f4:	00d7e7b3          	or	a5,a5,a3
   166f8:	000026b7          	lui	a3,0x2
   166fc:	00d76733          	or	a4,a4,a3
   16700:	00f61623          	sh	a5,12(a2)
   16704:	06e62223          	sw	a4,100(a2)
   16708:	00040513          	mv	a0,s0
   1670c:	01812403          	lw	s0,24(sp)
   16710:	01c12083          	lw	ra,28(sp)
   16714:	02010113          	addi	sp,sp,32
   16718:	e85ff06f          	j	1659c <__fputwc>
   1671c:	00a12423          	sw	a0,8(sp)
   16720:	00040513          	mv	a0,s0
   16724:	00c12623          	sw	a2,12(sp)
   16728:	f3df90ef          	jal	10664 <__sinit>
   1672c:	00c12603          	lw	a2,12(sp)
   16730:	00812583          	lw	a1,8(sp)
   16734:	fadff06f          	j	166e0 <fputwc+0x24>

00016738 <memchr>:
   16738:	00357793          	andi	a5,a0,3
   1673c:	0ff5f693          	zext.b	a3,a1
   16740:	02078a63          	beqz	a5,16774 <memchr+0x3c>
   16744:	fff60793          	addi	a5,a2,-1
   16748:	02060e63          	beqz	a2,16784 <memchr+0x4c>
   1674c:	fff00613          	li	a2,-1
   16750:	0180006f          	j	16768 <memchr+0x30>
   16754:	00150513          	addi	a0,a0,1
   16758:	00357713          	andi	a4,a0,3
   1675c:	00070e63          	beqz	a4,16778 <memchr+0x40>
   16760:	fff78793          	addi	a5,a5,-1
   16764:	02c78063          	beq	a5,a2,16784 <memchr+0x4c>
   16768:	00054703          	lbu	a4,0(a0)
   1676c:	fed714e3          	bne	a4,a3,16754 <memchr+0x1c>
   16770:	00008067          	ret
   16774:	00060793          	mv	a5,a2
   16778:	00300713          	li	a4,3
   1677c:	00f76863          	bltu	a4,a5,1678c <memchr+0x54>
   16780:	06079063          	bnez	a5,167e0 <memchr+0xa8>
   16784:	00000513          	li	a0,0
   16788:	00008067          	ret
   1678c:	0ff5f593          	zext.b	a1,a1
   16790:	00859713          	slli	a4,a1,0x8
   16794:	00b705b3          	add	a1,a4,a1
   16798:	01059713          	slli	a4,a1,0x10
   1679c:	feff08b7          	lui	a7,0xfeff0
   167a0:	80808837          	lui	a6,0x80808
   167a4:	00e585b3          	add	a1,a1,a4
   167a8:	eff88893          	addi	a7,a7,-257 # fefefeff <__BSS_END__+0xfefcd0ff>
   167ac:	08080813          	addi	a6,a6,128 # 80808080 <__BSS_END__+0x807e5280>
   167b0:	00300313          	li	t1,3
   167b4:	0100006f          	j	167c4 <memchr+0x8c>
   167b8:	ffc78793          	addi	a5,a5,-4
   167bc:	00450513          	addi	a0,a0,4
   167c0:	fcf370e3          	bgeu	t1,a5,16780 <memchr+0x48>
   167c4:	00052703          	lw	a4,0(a0)
   167c8:	00e5c733          	xor	a4,a1,a4
   167cc:	01170633          	add	a2,a4,a7
   167d0:	fff74713          	not	a4,a4
   167d4:	00e67733          	and	a4,a2,a4
   167d8:	01077733          	and	a4,a4,a6
   167dc:	fc070ee3          	beqz	a4,167b8 <memchr+0x80>
   167e0:	00f507b3          	add	a5,a0,a5
   167e4:	00c0006f          	j	167f0 <memchr+0xb8>
   167e8:	00150513          	addi	a0,a0,1
   167ec:	f8a78ce3          	beq	a5,a0,16784 <memchr+0x4c>
   167f0:	00054703          	lbu	a4,0(a0)
   167f4:	fed71ae3          	bne	a4,a3,167e8 <memchr+0xb0>
   167f8:	00008067          	ret

000167fc <strncpy>:
   167fc:	00a5e7b3          	or	a5,a1,a0
   16800:	0037f793          	andi	a5,a5,3
   16804:	00079663          	bnez	a5,16810 <strncpy+0x14>
   16808:	00300793          	li	a5,3
   1680c:	04c7e663          	bltu	a5,a2,16858 <strncpy+0x5c>
   16810:	00050713          	mv	a4,a0
   16814:	01c0006f          	j	16830 <strncpy+0x34>
   16818:	fff5c683          	lbu	a3,-1(a1)
   1681c:	fff60813          	addi	a6,a2,-1
   16820:	fed78fa3          	sb	a3,-1(a5)
   16824:	00068e63          	beqz	a3,16840 <strncpy+0x44>
   16828:	00078713          	mv	a4,a5
   1682c:	00080613          	mv	a2,a6
   16830:	00158593          	addi	a1,a1,1
   16834:	00170793          	addi	a5,a4,1
   16838:	fe0610e3          	bnez	a2,16818 <strncpy+0x1c>
   1683c:	00008067          	ret
   16840:	00c70733          	add	a4,a4,a2
   16844:	06080063          	beqz	a6,168a4 <strncpy+0xa8>
   16848:	00178793          	addi	a5,a5,1
   1684c:	fe078fa3          	sb	zero,-1(a5)
   16850:	fee79ce3          	bne	a5,a4,16848 <strncpy+0x4c>
   16854:	00008067          	ret
   16858:	feff0337          	lui	t1,0xfeff0
   1685c:	808088b7          	lui	a7,0x80808
   16860:	00050713          	mv	a4,a0
   16864:	eff30313          	addi	t1,t1,-257 # fefefeff <__BSS_END__+0xfefcd0ff>
   16868:	08088893          	addi	a7,a7,128 # 80808080 <__BSS_END__+0x807e5280>
   1686c:	00300e13          	li	t3,3
   16870:	0180006f          	j	16888 <strncpy+0x8c>
   16874:	00d72023          	sw	a3,0(a4)
   16878:	ffc60613          	addi	a2,a2,-4
   1687c:	00470713          	addi	a4,a4,4
   16880:	00458593          	addi	a1,a1,4
   16884:	face76e3          	bgeu	t3,a2,16830 <strncpy+0x34>
   16888:	0005a683          	lw	a3,0(a1)
   1688c:	006687b3          	add	a5,a3,t1
   16890:	fff6c813          	not	a6,a3
   16894:	0107f7b3          	and	a5,a5,a6
   16898:	0117f7b3          	and	a5,a5,a7
   1689c:	fc078ce3          	beqz	a5,16874 <strncpy+0x78>
   168a0:	f91ff06f          	j	16830 <strncpy+0x34>
   168a4:	00008067          	ret

000168a8 <_setlocale_r>:
   168a8:	04060063          	beqz	a2,168e8 <_setlocale_r+0x40>
   168ac:	ff010113          	addi	sp,sp,-16
   168b0:	0000a597          	auipc	a1,0xa
   168b4:	0b458593          	addi	a1,a1,180 # 20964 <_exit+0x12c>
   168b8:	00060513          	mv	a0,a2
   168bc:	00812423          	sw	s0,8(sp)
   168c0:	00112623          	sw	ra,12(sp)
   168c4:	00060413          	mv	s0,a2
   168c8:	454000ef          	jal	16d1c <strcmp>
   168cc:	02051463          	bnez	a0,168f4 <_setlocale_r+0x4c>
   168d0:	0000a517          	auipc	a0,0xa
   168d4:	09050513          	addi	a0,a0,144 # 20960 <_exit+0x128>
   168d8:	00c12083          	lw	ra,12(sp)
   168dc:	00812403          	lw	s0,8(sp)
   168e0:	01010113          	addi	sp,sp,16
   168e4:	00008067          	ret
   168e8:	0000a517          	auipc	a0,0xa
   168ec:	07850513          	addi	a0,a0,120 # 20960 <_exit+0x128>
   168f0:	00008067          	ret
   168f4:	0000a597          	auipc	a1,0xa
   168f8:	06c58593          	addi	a1,a1,108 # 20960 <_exit+0x128>
   168fc:	00040513          	mv	a0,s0
   16900:	41c000ef          	jal	16d1c <strcmp>
   16904:	fc0506e3          	beqz	a0,168d0 <_setlocale_r+0x28>
   16908:	0000a597          	auipc	a1,0xa
   1690c:	13058593          	addi	a1,a1,304 # 20a38 <_exit+0x200>
   16910:	00040513          	mv	a0,s0
   16914:	408000ef          	jal	16d1c <strcmp>
   16918:	fa050ce3          	beqz	a0,168d0 <_setlocale_r+0x28>
   1691c:	00000513          	li	a0,0
   16920:	fb9ff06f          	j	168d8 <_setlocale_r+0x30>

00016924 <__locale_mb_cur_max>:
   16924:	eb01c503          	lbu	a0,-336(gp) # 22930 <__global_locale+0x128>
   16928:	00008067          	ret

0001692c <setlocale>:
   1692c:	04058063          	beqz	a1,1696c <setlocale+0x40>
   16930:	ff010113          	addi	sp,sp,-16
   16934:	00812423          	sw	s0,8(sp)
   16938:	00058413          	mv	s0,a1
   1693c:	00040513          	mv	a0,s0
   16940:	0000a597          	auipc	a1,0xa
   16944:	02458593          	addi	a1,a1,36 # 20964 <_exit+0x12c>
   16948:	00112623          	sw	ra,12(sp)
   1694c:	3d0000ef          	jal	16d1c <strcmp>
   16950:	02051463          	bnez	a0,16978 <setlocale+0x4c>
   16954:	0000a517          	auipc	a0,0xa
   16958:	00c50513          	addi	a0,a0,12 # 20960 <_exit+0x128>
   1695c:	00c12083          	lw	ra,12(sp)
   16960:	00812403          	lw	s0,8(sp)
   16964:	01010113          	addi	sp,sp,16
   16968:	00008067          	ret
   1696c:	0000a517          	auipc	a0,0xa
   16970:	ff450513          	addi	a0,a0,-12 # 20960 <_exit+0x128>
   16974:	00008067          	ret
   16978:	0000a597          	auipc	a1,0xa
   1697c:	fe858593          	addi	a1,a1,-24 # 20960 <_exit+0x128>
   16980:	00040513          	mv	a0,s0
   16984:	398000ef          	jal	16d1c <strcmp>
   16988:	fc0506e3          	beqz	a0,16954 <setlocale+0x28>
   1698c:	0000a597          	auipc	a1,0xa
   16990:	0ac58593          	addi	a1,a1,172 # 20a38 <_exit+0x200>
   16994:	00040513          	mv	a0,s0
   16998:	384000ef          	jal	16d1c <strcmp>
   1699c:	fa050ce3          	beqz	a0,16954 <setlocale+0x28>
   169a0:	00000513          	li	a0,0
   169a4:	fb9ff06f          	j	1695c <setlocale+0x30>

000169a8 <__localeconv_l>:
   169a8:	0f050513          	addi	a0,a0,240
   169ac:	00008067          	ret

000169b0 <_localeconv_r>:
   169b0:	e7818513          	addi	a0,gp,-392 # 228f8 <__global_locale+0xf0>
   169b4:	00008067          	ret

000169b8 <localeconv>:
   169b8:	e7818513          	addi	a0,gp,-392 # 228f8 <__global_locale+0xf0>
   169bc:	00008067          	ret

000169c0 <_sbrk_r>:
   169c0:	ff010113          	addi	sp,sp,-16
   169c4:	00812423          	sw	s0,8(sp)
   169c8:	00050413          	mv	s0,a0
   169cc:	00058513          	mv	a0,a1
   169d0:	f601a623          	sw	zero,-148(gp) # 229ec <errno>
   169d4:	00112623          	sw	ra,12(sp)
   169d8:	621090ef          	jal	207f8 <_sbrk>
   169dc:	fff00793          	li	a5,-1
   169e0:	00f50a63          	beq	a0,a5,169f4 <_sbrk_r+0x34>
   169e4:	00c12083          	lw	ra,12(sp)
   169e8:	00812403          	lw	s0,8(sp)
   169ec:	01010113          	addi	sp,sp,16
   169f0:	00008067          	ret
   169f4:	f6c1a783          	lw	a5,-148(gp) # 229ec <errno>
   169f8:	fe0786e3          	beqz	a5,169e4 <_sbrk_r+0x24>
   169fc:	00c12083          	lw	ra,12(sp)
   16a00:	00f42023          	sw	a5,0(s0)
   16a04:	00812403          	lw	s0,8(sp)
   16a08:	01010113          	addi	sp,sp,16
   16a0c:	00008067          	ret

00016a10 <__libc_fini_array>:
   16a10:	ff010113          	addi	sp,sp,-16
   16a14:	00812423          	sw	s0,8(sp)
   16a18:	0000c797          	auipc	a5,0xc
   16a1c:	84c78793          	addi	a5,a5,-1972 # 22264 <__do_global_dtors_aux_fini_array_entry>
   16a20:	0000c417          	auipc	s0,0xc
   16a24:	84840413          	addi	s0,s0,-1976 # 22268 <__fini_array_end>
   16a28:	40f40433          	sub	s0,s0,a5
   16a2c:	00912223          	sw	s1,4(sp)
   16a30:	00112623          	sw	ra,12(sp)
   16a34:	40245493          	srai	s1,s0,0x2
   16a38:	02048063          	beqz	s1,16a58 <__libc_fini_array+0x48>
   16a3c:	ffc40413          	addi	s0,s0,-4
   16a40:	00f40433          	add	s0,s0,a5
   16a44:	00042783          	lw	a5,0(s0)
   16a48:	fff48493          	addi	s1,s1,-1
   16a4c:	ffc40413          	addi	s0,s0,-4
   16a50:	000780e7          	jalr	a5
   16a54:	fe0498e3          	bnez	s1,16a44 <__libc_fini_array+0x34>
   16a58:	00c12083          	lw	ra,12(sp)
   16a5c:	00812403          	lw	s0,8(sp)
   16a60:	00412483          	lw	s1,4(sp)
   16a64:	01010113          	addi	sp,sp,16
   16a68:	00008067          	ret

00016a6c <memmove>:
   16a6c:	02a5f663          	bgeu	a1,a0,16a98 <memmove+0x2c>
   16a70:	00c58733          	add	a4,a1,a2
   16a74:	02e57263          	bgeu	a0,a4,16a98 <memmove+0x2c>
   16a78:	00c507b3          	add	a5,a0,a2
   16a7c:	04060663          	beqz	a2,16ac8 <memmove+0x5c>
   16a80:	fff74683          	lbu	a3,-1(a4)
   16a84:	fff78793          	addi	a5,a5,-1
   16a88:	fff70713          	addi	a4,a4,-1
   16a8c:	00d78023          	sb	a3,0(a5)
   16a90:	fef518e3          	bne	a0,a5,16a80 <memmove+0x14>
   16a94:	00008067          	ret
   16a98:	00f00793          	li	a5,15
   16a9c:	02c7e863          	bltu	a5,a2,16acc <memmove+0x60>
   16aa0:	00050793          	mv	a5,a0
   16aa4:	fff60693          	addi	a3,a2,-1
   16aa8:	0c060263          	beqz	a2,16b6c <memmove+0x100>
   16aac:	00168693          	addi	a3,a3,1 # 2001 <exit-0xe0b3>
   16ab0:	00d786b3          	add	a3,a5,a3
   16ab4:	0005c703          	lbu	a4,0(a1)
   16ab8:	00178793          	addi	a5,a5,1
   16abc:	00158593          	addi	a1,a1,1
   16ac0:	fee78fa3          	sb	a4,-1(a5)
   16ac4:	fed798e3          	bne	a5,a3,16ab4 <memmove+0x48>
   16ac8:	00008067          	ret
   16acc:	00b567b3          	or	a5,a0,a1
   16ad0:	0037f793          	andi	a5,a5,3
   16ad4:	08079663          	bnez	a5,16b60 <memmove+0xf4>
   16ad8:	ff060893          	addi	a7,a2,-16
   16adc:	ff08f893          	andi	a7,a7,-16
   16ae0:	01088893          	addi	a7,a7,16
   16ae4:	011506b3          	add	a3,a0,a7
   16ae8:	00058713          	mv	a4,a1
   16aec:	00050793          	mv	a5,a0
   16af0:	00072803          	lw	a6,0(a4)
   16af4:	01070713          	addi	a4,a4,16
   16af8:	01078793          	addi	a5,a5,16
   16afc:	ff07a823          	sw	a6,-16(a5)
   16b00:	ff472803          	lw	a6,-12(a4)
   16b04:	ff07aa23          	sw	a6,-12(a5)
   16b08:	ff872803          	lw	a6,-8(a4)
   16b0c:	ff07ac23          	sw	a6,-8(a5)
   16b10:	ffc72803          	lw	a6,-4(a4)
   16b14:	ff07ae23          	sw	a6,-4(a5)
   16b18:	fcd79ce3          	bne	a5,a3,16af0 <memmove+0x84>
   16b1c:	00c67813          	andi	a6,a2,12
   16b20:	011585b3          	add	a1,a1,a7
   16b24:	00f67713          	andi	a4,a2,15
   16b28:	04080463          	beqz	a6,16b70 <memmove+0x104>
   16b2c:	ffc70813          	addi	a6,a4,-4
   16b30:	ffc87813          	andi	a6,a6,-4
   16b34:	00480813          	addi	a6,a6,4
   16b38:	010687b3          	add	a5,a3,a6
   16b3c:	00058713          	mv	a4,a1
   16b40:	00072883          	lw	a7,0(a4)
   16b44:	00468693          	addi	a3,a3,4
   16b48:	00470713          	addi	a4,a4,4
   16b4c:	ff16ae23          	sw	a7,-4(a3)
   16b50:	fef698e3          	bne	a3,a5,16b40 <memmove+0xd4>
   16b54:	00367613          	andi	a2,a2,3
   16b58:	010585b3          	add	a1,a1,a6
   16b5c:	f49ff06f          	j	16aa4 <memmove+0x38>
   16b60:	fff60693          	addi	a3,a2,-1
   16b64:	00050793          	mv	a5,a0
   16b68:	f45ff06f          	j	16aac <memmove+0x40>
   16b6c:	00008067          	ret
   16b70:	00070613          	mv	a2,a4
   16b74:	f31ff06f          	j	16aa4 <memmove+0x38>

00016b78 <memcpy>:
   16b78:	00a5c7b3          	xor	a5,a1,a0
   16b7c:	0037f793          	andi	a5,a5,3
   16b80:	00c508b3          	add	a7,a0,a2
   16b84:	06079463          	bnez	a5,16bec <memcpy+0x74>
   16b88:	00300793          	li	a5,3
   16b8c:	06c7f063          	bgeu	a5,a2,16bec <memcpy+0x74>
   16b90:	00357793          	andi	a5,a0,3
   16b94:	00050713          	mv	a4,a0
   16b98:	06079a63          	bnez	a5,16c0c <memcpy+0x94>
   16b9c:	ffc8f613          	andi	a2,a7,-4
   16ba0:	40e606b3          	sub	a3,a2,a4
   16ba4:	02000793          	li	a5,32
   16ba8:	08d7ce63          	blt	a5,a3,16c44 <memcpy+0xcc>
   16bac:	00058693          	mv	a3,a1
   16bb0:	00070793          	mv	a5,a4
   16bb4:	02c77863          	bgeu	a4,a2,16be4 <memcpy+0x6c>
   16bb8:	0006a803          	lw	a6,0(a3)
   16bbc:	00478793          	addi	a5,a5,4
   16bc0:	00468693          	addi	a3,a3,4
   16bc4:	ff07ae23          	sw	a6,-4(a5)
   16bc8:	fec7e8e3          	bltu	a5,a2,16bb8 <memcpy+0x40>
   16bcc:	fff60793          	addi	a5,a2,-1
   16bd0:	40e787b3          	sub	a5,a5,a4
   16bd4:	ffc7f793          	andi	a5,a5,-4
   16bd8:	00478793          	addi	a5,a5,4
   16bdc:	00f70733          	add	a4,a4,a5
   16be0:	00f585b3          	add	a1,a1,a5
   16be4:	01176863          	bltu	a4,a7,16bf4 <memcpy+0x7c>
   16be8:	00008067          	ret
   16bec:	00050713          	mv	a4,a0
   16bf0:	05157863          	bgeu	a0,a7,16c40 <memcpy+0xc8>
   16bf4:	0005c783          	lbu	a5,0(a1)
   16bf8:	00170713          	addi	a4,a4,1
   16bfc:	00158593          	addi	a1,a1,1
   16c00:	fef70fa3          	sb	a5,-1(a4)
   16c04:	fee898e3          	bne	a7,a4,16bf4 <memcpy+0x7c>
   16c08:	00008067          	ret
   16c0c:	0005c683          	lbu	a3,0(a1)
   16c10:	00170713          	addi	a4,a4,1
   16c14:	00377793          	andi	a5,a4,3
   16c18:	fed70fa3          	sb	a3,-1(a4)
   16c1c:	00158593          	addi	a1,a1,1
   16c20:	f6078ee3          	beqz	a5,16b9c <memcpy+0x24>
   16c24:	0005c683          	lbu	a3,0(a1)
   16c28:	00170713          	addi	a4,a4,1
   16c2c:	00377793          	andi	a5,a4,3
   16c30:	fed70fa3          	sb	a3,-1(a4)
   16c34:	00158593          	addi	a1,a1,1
   16c38:	fc079ae3          	bnez	a5,16c0c <memcpy+0x94>
   16c3c:	f61ff06f          	j	16b9c <memcpy+0x24>
   16c40:	00008067          	ret
   16c44:	ff010113          	addi	sp,sp,-16
   16c48:	00812623          	sw	s0,12(sp)
   16c4c:	02000413          	li	s0,32
   16c50:	0005a383          	lw	t2,0(a1)
   16c54:	0045a283          	lw	t0,4(a1)
   16c58:	0085af83          	lw	t6,8(a1)
   16c5c:	00c5af03          	lw	t5,12(a1)
   16c60:	0105ae83          	lw	t4,16(a1)
   16c64:	0145ae03          	lw	t3,20(a1)
   16c68:	0185a303          	lw	t1,24(a1)
   16c6c:	01c5a803          	lw	a6,28(a1)
   16c70:	0205a683          	lw	a3,32(a1)
   16c74:	02470713          	addi	a4,a4,36
   16c78:	40e607b3          	sub	a5,a2,a4
   16c7c:	fc772e23          	sw	t2,-36(a4)
   16c80:	fe572023          	sw	t0,-32(a4)
   16c84:	fff72223          	sw	t6,-28(a4)
   16c88:	ffe72423          	sw	t5,-24(a4)
   16c8c:	ffd72623          	sw	t4,-20(a4)
   16c90:	ffc72823          	sw	t3,-16(a4)
   16c94:	fe672a23          	sw	t1,-12(a4)
   16c98:	ff072c23          	sw	a6,-8(a4)
   16c9c:	fed72e23          	sw	a3,-4(a4)
   16ca0:	02458593          	addi	a1,a1,36
   16ca4:	faf446e3          	blt	s0,a5,16c50 <memcpy+0xd8>
   16ca8:	00058693          	mv	a3,a1
   16cac:	00070793          	mv	a5,a4
   16cb0:	02c77863          	bgeu	a4,a2,16ce0 <memcpy+0x168>
   16cb4:	0006a803          	lw	a6,0(a3)
   16cb8:	00478793          	addi	a5,a5,4
   16cbc:	00468693          	addi	a3,a3,4
   16cc0:	ff07ae23          	sw	a6,-4(a5)
   16cc4:	fec7e8e3          	bltu	a5,a2,16cb4 <memcpy+0x13c>
   16cc8:	fff60793          	addi	a5,a2,-1
   16ccc:	40e787b3          	sub	a5,a5,a4
   16cd0:	ffc7f793          	andi	a5,a5,-4
   16cd4:	00478793          	addi	a5,a5,4
   16cd8:	00f70733          	add	a4,a4,a5
   16cdc:	00f585b3          	add	a1,a1,a5
   16ce0:	01176863          	bltu	a4,a7,16cf0 <memcpy+0x178>
   16ce4:	00c12403          	lw	s0,12(sp)
   16ce8:	01010113          	addi	sp,sp,16
   16cec:	00008067          	ret
   16cf0:	0005c783          	lbu	a5,0(a1)
   16cf4:	00170713          	addi	a4,a4,1
   16cf8:	00158593          	addi	a1,a1,1
   16cfc:	fef70fa3          	sb	a5,-1(a4)
   16d00:	fee882e3          	beq	a7,a4,16ce4 <memcpy+0x16c>
   16d04:	0005c783          	lbu	a5,0(a1)
   16d08:	00170713          	addi	a4,a4,1
   16d0c:	00158593          	addi	a1,a1,1
   16d10:	fef70fa3          	sb	a5,-1(a4)
   16d14:	fce89ee3          	bne	a7,a4,16cf0 <memcpy+0x178>
   16d18:	fcdff06f          	j	16ce4 <memcpy+0x16c>

00016d1c <strcmp>:
   16d1c:	00b56733          	or	a4,a0,a1
   16d20:	fff00393          	li	t2,-1
   16d24:	00377713          	andi	a4,a4,3
   16d28:	10071063          	bnez	a4,16e28 <strcmp+0x10c>
   16d2c:	7f7f87b7          	lui	a5,0x7f7f8
   16d30:	f7f78793          	addi	a5,a5,-129 # 7f7f7f7f <__BSS_END__+0x7f7d517f>
   16d34:	00052603          	lw	a2,0(a0)
   16d38:	0005a683          	lw	a3,0(a1)
   16d3c:	00f672b3          	and	t0,a2,a5
   16d40:	00f66333          	or	t1,a2,a5
   16d44:	00f282b3          	add	t0,t0,a5
   16d48:	0062e2b3          	or	t0,t0,t1
   16d4c:	10729263          	bne	t0,t2,16e50 <strcmp+0x134>
   16d50:	08d61663          	bne	a2,a3,16ddc <strcmp+0xc0>
   16d54:	00452603          	lw	a2,4(a0)
   16d58:	0045a683          	lw	a3,4(a1)
   16d5c:	00f672b3          	and	t0,a2,a5
   16d60:	00f66333          	or	t1,a2,a5
   16d64:	00f282b3          	add	t0,t0,a5
   16d68:	0062e2b3          	or	t0,t0,t1
   16d6c:	0c729e63          	bne	t0,t2,16e48 <strcmp+0x12c>
   16d70:	06d61663          	bne	a2,a3,16ddc <strcmp+0xc0>
   16d74:	00852603          	lw	a2,8(a0)
   16d78:	0085a683          	lw	a3,8(a1)
   16d7c:	00f672b3          	and	t0,a2,a5
   16d80:	00f66333          	or	t1,a2,a5
   16d84:	00f282b3          	add	t0,t0,a5
   16d88:	0062e2b3          	or	t0,t0,t1
   16d8c:	0c729863          	bne	t0,t2,16e5c <strcmp+0x140>
   16d90:	04d61663          	bne	a2,a3,16ddc <strcmp+0xc0>
   16d94:	00c52603          	lw	a2,12(a0)
   16d98:	00c5a683          	lw	a3,12(a1)
   16d9c:	00f672b3          	and	t0,a2,a5
   16da0:	00f66333          	or	t1,a2,a5
   16da4:	00f282b3          	add	t0,t0,a5
   16da8:	0062e2b3          	or	t0,t0,t1
   16dac:	0c729263          	bne	t0,t2,16e70 <strcmp+0x154>
   16db0:	02d61663          	bne	a2,a3,16ddc <strcmp+0xc0>
   16db4:	01052603          	lw	a2,16(a0)
   16db8:	0105a683          	lw	a3,16(a1)
   16dbc:	00f672b3          	and	t0,a2,a5
   16dc0:	00f66333          	or	t1,a2,a5
   16dc4:	00f282b3          	add	t0,t0,a5
   16dc8:	0062e2b3          	or	t0,t0,t1
   16dcc:	0a729c63          	bne	t0,t2,16e84 <strcmp+0x168>
   16dd0:	01450513          	addi	a0,a0,20
   16dd4:	01458593          	addi	a1,a1,20
   16dd8:	f4d60ee3          	beq	a2,a3,16d34 <strcmp+0x18>
   16ddc:	01061713          	slli	a4,a2,0x10
   16de0:	01069793          	slli	a5,a3,0x10
   16de4:	00f71e63          	bne	a4,a5,16e00 <strcmp+0xe4>
   16de8:	01065713          	srli	a4,a2,0x10
   16dec:	0106d793          	srli	a5,a3,0x10
   16df0:	40f70533          	sub	a0,a4,a5
   16df4:	0ff57593          	zext.b	a1,a0
   16df8:	02059063          	bnez	a1,16e18 <strcmp+0xfc>
   16dfc:	00008067          	ret
   16e00:	01075713          	srli	a4,a4,0x10
   16e04:	0107d793          	srli	a5,a5,0x10
   16e08:	40f70533          	sub	a0,a4,a5
   16e0c:	0ff57593          	zext.b	a1,a0
   16e10:	00059463          	bnez	a1,16e18 <strcmp+0xfc>
   16e14:	00008067          	ret
   16e18:	0ff77713          	zext.b	a4,a4
   16e1c:	0ff7f793          	zext.b	a5,a5
   16e20:	40f70533          	sub	a0,a4,a5
   16e24:	00008067          	ret
   16e28:	00054603          	lbu	a2,0(a0)
   16e2c:	0005c683          	lbu	a3,0(a1)
   16e30:	00150513          	addi	a0,a0,1
   16e34:	00158593          	addi	a1,a1,1
   16e38:	00d61463          	bne	a2,a3,16e40 <strcmp+0x124>
   16e3c:	fe0616e3          	bnez	a2,16e28 <strcmp+0x10c>
   16e40:	40d60533          	sub	a0,a2,a3
   16e44:	00008067          	ret
   16e48:	00450513          	addi	a0,a0,4
   16e4c:	00458593          	addi	a1,a1,4
   16e50:	fcd61ce3          	bne	a2,a3,16e28 <strcmp+0x10c>
   16e54:	00000513          	li	a0,0
   16e58:	00008067          	ret
   16e5c:	00850513          	addi	a0,a0,8
   16e60:	00858593          	addi	a1,a1,8
   16e64:	fcd612e3          	bne	a2,a3,16e28 <strcmp+0x10c>
   16e68:	00000513          	li	a0,0
   16e6c:	00008067          	ret
   16e70:	00c50513          	addi	a0,a0,12
   16e74:	00c58593          	addi	a1,a1,12
   16e78:	fad618e3          	bne	a2,a3,16e28 <strcmp+0x10c>
   16e7c:	00000513          	li	a0,0
   16e80:	00008067          	ret
   16e84:	01050513          	addi	a0,a0,16
   16e88:	01058593          	addi	a1,a1,16
   16e8c:	f8d61ee3          	bne	a2,a3,16e28 <strcmp+0x10c>
   16e90:	00000513          	li	a0,0
   16e94:	00008067          	ret

00016e98 <frexpl>:
   16e98:	f9010113          	addi	sp,sp,-112
   16e9c:	07212023          	sw	s2,96(sp)
   16ea0:	00c5a903          	lw	s2,12(a1)
   16ea4:	05412c23          	sw	s4,88(sp)
   16ea8:	05512a23          	sw	s5,84(sp)
   16eac:	05612823          	sw	s6,80(sp)
   16eb0:	0045aa83          	lw	s5,4(a1)
   16eb4:	0005ab03          	lw	s6,0(a1)
   16eb8:	0085aa03          	lw	s4,8(a1)
   16ebc:	05312e23          	sw	s3,92(sp)
   16ec0:	000089b7          	lui	s3,0x8
   16ec4:	06812423          	sw	s0,104(sp)
   16ec8:	06912223          	sw	s1,100(sp)
   16ecc:	06112623          	sw	ra,108(sp)
   16ed0:	01095493          	srli	s1,s2,0x10
   16ed4:	fff98993          	addi	s3,s3,-1 # 7fff <exit-0x80b5>
   16ed8:	03612823          	sw	s6,48(sp)
   16edc:	03512a23          	sw	s5,52(sp)
   16ee0:	03412c23          	sw	s4,56(sp)
   16ee4:	03212e23          	sw	s2,60(sp)
   16ee8:	0134f4b3          	and	s1,s1,s3
   16eec:	00062023          	sw	zero,0(a2)
   16ef0:	00050413          	mv	s0,a0
   16ef4:	09348063          	beq	s1,s3,16f74 <frexpl+0xdc>
   16ef8:	01010593          	addi	a1,sp,16
   16efc:	02010513          	addi	a0,sp,32
   16f00:	05712623          	sw	s7,76(sp)
   16f04:	03612023          	sw	s6,32(sp)
   16f08:	00060b93          	mv	s7,a2
   16f0c:	03512223          	sw	s5,36(sp)
   16f10:	03412423          	sw	s4,40(sp)
   16f14:	03212623          	sw	s2,44(sp)
   16f18:	00012823          	sw	zero,16(sp)
   16f1c:	00012a23          	sw	zero,20(sp)
   16f20:	00012c23          	sw	zero,24(sp)
   16f24:	00012e23          	sw	zero,28(sp)
   16f28:	371060ef          	jal	1da98 <__eqtf2>
   16f2c:	0e050e63          	beqz	a0,17028 <frexpl+0x190>
   16f30:	00000693          	li	a3,0
   16f34:	06048e63          	beqz	s1,16fb0 <frexpl+0x118>
   16f38:	ffffc737          	lui	a4,0xffffc
   16f3c:	00270713          	addi	a4,a4,2 # ffffc002 <__BSS_END__+0xfffd9202>
   16f40:	03c12903          	lw	s2,60(sp)
   16f44:	00e484b3          	add	s1,s1,a4
   16f48:	800107b7          	lui	a5,0x80010
   16f4c:	00d484b3          	add	s1,s1,a3
   16f50:	fff78793          	addi	a5,a5,-1 # 8000ffff <__BSS_END__+0x7ffed1ff>
   16f54:	009ba023          	sw	s1,0(s7)
   16f58:	03012b03          	lw	s6,48(sp)
   16f5c:	03412a83          	lw	s5,52(sp)
   16f60:	03812a03          	lw	s4,56(sp)
   16f64:	04c12b83          	lw	s7,76(sp)
   16f68:	00f97933          	and	s2,s2,a5
   16f6c:	3ffe07b7          	lui	a5,0x3ffe0
   16f70:	00f96933          	or	s2,s2,a5
   16f74:	01642023          	sw	s6,0(s0)
   16f78:	01542223          	sw	s5,4(s0)
   16f7c:	01442423          	sw	s4,8(s0)
   16f80:	01242623          	sw	s2,12(s0)
   16f84:	06c12083          	lw	ra,108(sp)
   16f88:	00040513          	mv	a0,s0
   16f8c:	06812403          	lw	s0,104(sp)
   16f90:	06412483          	lw	s1,100(sp)
   16f94:	06012903          	lw	s2,96(sp)
   16f98:	05c12983          	lw	s3,92(sp)
   16f9c:	05812a03          	lw	s4,88(sp)
   16fa0:	05412a83          	lw	s5,84(sp)
   16fa4:	05012b03          	lw	s6,80(sp)
   16fa8:	07010113          	addi	sp,sp,112
   16fac:	00008067          	ret
   16fb0:	0000a797          	auipc	a5,0xa
   16fb4:	d1078793          	addi	a5,a5,-752 # 20cc0 <blanks.1+0x40>
   16fb8:	0007a603          	lw	a2,0(a5)
   16fbc:	0047a683          	lw	a3,4(a5)
   16fc0:	0087a703          	lw	a4,8(a5)
   16fc4:	00c7a783          	lw	a5,12(a5)
   16fc8:	00c12023          	sw	a2,0(sp)
   16fcc:	01010593          	addi	a1,sp,16
   16fd0:	00010613          	mv	a2,sp
   16fd4:	02010513          	addi	a0,sp,32
   16fd8:	00d12223          	sw	a3,4(sp)
   16fdc:	00e12423          	sw	a4,8(sp)
   16fe0:	00f12623          	sw	a5,12(sp)
   16fe4:	01612823          	sw	s6,16(sp)
   16fe8:	01512a23          	sw	s5,20(sp)
   16fec:	01412c23          	sw	s4,24(sp)
   16ff0:	01212e23          	sw	s2,28(sp)
   16ff4:	5d1060ef          	jal	1ddc4 <__multf3>
   16ff8:	02012703          	lw	a4,32(sp)
   16ffc:	02c12783          	lw	a5,44(sp)
   17000:	f8e00693          	li	a3,-114
   17004:	02e12823          	sw	a4,48(sp)
   17008:	02412703          	lw	a4,36(sp)
   1700c:	0107d493          	srli	s1,a5,0x10
   17010:	02f12e23          	sw	a5,60(sp)
   17014:	02e12a23          	sw	a4,52(sp)
   17018:	02812703          	lw	a4,40(sp)
   1701c:	0134f4b3          	and	s1,s1,s3
   17020:	02e12c23          	sw	a4,56(sp)
   17024:	f15ff06f          	j	16f38 <frexpl+0xa0>
   17028:	04c12b83          	lw	s7,76(sp)
   1702c:	f49ff06f          	j	16f74 <frexpl+0xdc>

00017030 <__register_exitproc>:
   17030:	f7018713          	addi	a4,gp,-144 # 229f0 <__atexit>
   17034:	00072783          	lw	a5,0(a4)
   17038:	04078c63          	beqz	a5,17090 <__register_exitproc+0x60>
   1703c:	0047a703          	lw	a4,4(a5)
   17040:	01f00813          	li	a6,31
   17044:	06e84e63          	blt	a6,a4,170c0 <__register_exitproc+0x90>
   17048:	00271813          	slli	a6,a4,0x2
   1704c:	02050663          	beqz	a0,17078 <__register_exitproc+0x48>
   17050:	01078333          	add	t1,a5,a6
   17054:	08c32423          	sw	a2,136(t1)
   17058:	1887a883          	lw	a7,392(a5)
   1705c:	00100613          	li	a2,1
   17060:	00e61633          	sll	a2,a2,a4
   17064:	00c8e8b3          	or	a7,a7,a2
   17068:	1917a423          	sw	a7,392(a5)
   1706c:	10d32423          	sw	a3,264(t1)
   17070:	00200693          	li	a3,2
   17074:	02d50463          	beq	a0,a3,1709c <__register_exitproc+0x6c>
   17078:	00170713          	addi	a4,a4,1
   1707c:	00e7a223          	sw	a4,4(a5)
   17080:	010787b3          	add	a5,a5,a6
   17084:	00b7a423          	sw	a1,8(a5)
   17088:	00000513          	li	a0,0
   1708c:	00008067          	ret
   17090:	1f018793          	addi	a5,gp,496 # 22c70 <__atexit0>
   17094:	00f72023          	sw	a5,0(a4)
   17098:	fa5ff06f          	j	1703c <__register_exitproc+0xc>
   1709c:	18c7a683          	lw	a3,396(a5)
   170a0:	00170713          	addi	a4,a4,1
   170a4:	00e7a223          	sw	a4,4(a5)
   170a8:	00c6e6b3          	or	a3,a3,a2
   170ac:	18d7a623          	sw	a3,396(a5)
   170b0:	010787b3          	add	a5,a5,a6
   170b4:	00b7a423          	sw	a1,8(a5)
   170b8:	00000513          	li	a0,0
   170bc:	00008067          	ret
   170c0:	fff00513          	li	a0,-1
   170c4:	00008067          	ret

000170c8 <_ldtoa_r>:
   170c8:	0000a897          	auipc	a7,0xa
   170cc:	db488893          	addi	a7,a7,-588 # 20e7c <blanks.1+0x10>
   170d0:	0008af83          	lw	t6,0(a7)
   170d4:	0048af03          	lw	t5,4(a7)
   170d8:	0088ae83          	lw	t4,8(a7)
   170dc:	00c8ae03          	lw	t3,12(a7)
   170e0:	0108a303          	lw	t1,16(a7)
   170e4:	03852883          	lw	a7,56(a0)
   170e8:	f4010113          	addi	sp,sp,-192
   170ec:	0b212823          	sw	s2,176(sp)
   170f0:	0b312623          	sw	s3,172(sp)
   170f4:	0b412423          	sw	s4,168(sp)
   170f8:	0b612023          	sw	s6,160(sp)
   170fc:	09712e23          	sw	s7,156(sp)
   17100:	09812c23          	sw	s8,152(sp)
   17104:	09912a23          	sw	s9,148(sp)
   17108:	09a12823          	sw	s10,144(sp)
   1710c:	0a112e23          	sw	ra,188(sp)
   17110:	0a812c23          	sw	s0,184(sp)
   17114:	0a912a23          	sw	s1,180(sp)
   17118:	0b512223          	sw	s5,164(sp)
   1711c:	09b12623          	sw	s11,140(sp)
   17120:	07f12623          	sw	t6,108(sp)
   17124:	07e12823          	sw	t5,112(sp)
   17128:	07d12a23          	sw	t4,116(sp)
   1712c:	07c12c23          	sw	t3,120(sp)
   17130:	06612e23          	sw	t1,124(sp)
   17134:	02c12023          	sw	a2,32(sp)
   17138:	02d12223          	sw	a3,36(sp)
   1713c:	0005aa03          	lw	s4,0(a1)
   17140:	0045a983          	lw	s3,4(a1)
   17144:	0085a903          	lw	s2,8(a1)
   17148:	00c5ac03          	lw	s8,12(a1)
   1714c:	00050b13          	mv	s6,a0
   17150:	00070b93          	mv	s7,a4
   17154:	00078c93          	mv	s9,a5
   17158:	00080d13          	mv	s10,a6
   1715c:	02088263          	beqz	a7,17180 <_ldtoa_r+0xb8>
   17160:	03c52683          	lw	a3,60(a0)
   17164:	00100713          	li	a4,1
   17168:	00088593          	mv	a1,a7
   1716c:	00d71733          	sll	a4,a4,a3
   17170:	00d8a223          	sw	a3,4(a7)
   17174:	00e8a423          	sw	a4,8(a7)
   17178:	54c020ef          	jal	196c4 <_Bfree>
   1717c:	020b2c23          	sw	zero,56(s6)
   17180:	07812603          	lw	a2,120(sp)
   17184:	01fc5693          	srli	a3,s8,0x1f
   17188:	001c1a93          	slli	s5,s8,0x1
   1718c:	40165713          	srai	a4,a2,0x1
   17190:	ffffc4b7          	lui	s1,0xffffc
   17194:	001c1413          	slli	s0,s8,0x1
   17198:	00d77733          	and	a4,a4,a3
   1719c:	011ada93          	srli	s5,s5,0x11
   171a0:	f9148493          	addi	s1,s1,-111 # ffffbf91 <__BSS_END__+0xfffd9191>
   171a4:	010c1d93          	slli	s11,s8,0x10
   171a8:	00dca023          	sw	a3,0(s9)
   171ac:	00145413          	srli	s0,s0,0x1
   171b0:	00c74733          	xor	a4,a4,a2
   171b4:	010ddd93          	srli	s11,s11,0x10
   171b8:	009a87b3          	add	a5,s5,s1
   171bc:	03010593          	addi	a1,sp,48
   171c0:	04010513          	addi	a0,sp,64
   171c4:	05412023          	sw	s4,64(sp)
   171c8:	05312223          	sw	s3,68(sp)
   171cc:	05212423          	sw	s2,72(sp)
   171d0:	04812623          	sw	s0,76(sp)
   171d4:	03412823          	sw	s4,48(sp)
   171d8:	03312a23          	sw	s3,52(sp)
   171dc:	03212c23          	sw	s2,56(sp)
   171e0:	02812e23          	sw	s0,60(sp)
   171e4:	06e12c23          	sw	a4,120(sp)
   171e8:	00f12e23          	sw	a5,28(sp)
   171ec:	05412e23          	sw	s4,92(sp)
   171f0:	07312023          	sw	s3,96(sp)
   171f4:	07212223          	sw	s2,100(sp)
   171f8:	07b12423          	sw	s11,104(sp)
   171fc:	05c090ef          	jal	20258 <__unordtf2>
   17200:	18051e63          	bnez	a0,1739c <_ldtoa_r+0x2d4>
   17204:	0000a797          	auipc	a5,0xa
   17208:	acc78793          	addi	a5,a5,-1332 # 20cd0 <blanks.1+0x50>
   1720c:	0007a603          	lw	a2,0(a5)
   17210:	0047a683          	lw	a3,4(a5)
   17214:	0087a483          	lw	s1,8(a5)
   17218:	00c7ac83          	lw	s9,12(a5)
   1721c:	03010593          	addi	a1,sp,48
   17220:	04010513          	addi	a0,sp,64
   17224:	05412023          	sw	s4,64(sp)
   17228:	05312223          	sw	s3,68(sp)
   1722c:	05212423          	sw	s2,72(sp)
   17230:	04812623          	sw	s0,76(sp)
   17234:	02c12823          	sw	a2,48(sp)
   17238:	02c12623          	sw	a2,44(sp)
   1723c:	02d12a23          	sw	a3,52(sp)
   17240:	02d12423          	sw	a3,40(sp)
   17244:	02912c23          	sw	s1,56(sp)
   17248:	03912e23          	sw	s9,60(sp)
   1724c:	00c090ef          	jal	20258 <__unordtf2>
   17250:	08051c63          	bnez	a0,172e8 <_ldtoa_r+0x220>
   17254:	03010593          	addi	a1,sp,48
   17258:	04010513          	addi	a0,sp,64
   1725c:	239060ef          	jal	1dc94 <__letf2>
   17260:	08a05463          	blez	a0,172e8 <_ldtoa_r+0x220>
   17264:	00300793          	li	a5,3
   17268:	04f12c23          	sw	a5,88(sp)
   1726c:	02012783          	lw	a5,32(sp)
   17270:	02412803          	lw	a6,36(sp)
   17274:	01c12603          	lw	a2,28(sp)
   17278:	05810713          	addi	a4,sp,88
   1727c:	01a12023          	sw	s10,0(sp)
   17280:	000b8893          	mv	a7,s7
   17284:	05c10693          	addi	a3,sp,92
   17288:	06c10593          	addi	a1,sp,108
   1728c:	000b0513          	mv	a0,s6
   17290:	238000ef          	jal	174c8 <__gdtoa>
   17294:	000ba703          	lw	a4,0(s7)
   17298:	ffff87b7          	lui	a5,0xffff8
   1729c:	00f71863          	bne	a4,a5,172ac <_ldtoa_r+0x1e4>
   172a0:	800007b7          	lui	a5,0x80000
   172a4:	fff78793          	addi	a5,a5,-1 # 7fffffff <__BSS_END__+0x7ffdd1ff>
   172a8:	00fba023          	sw	a5,0(s7)
   172ac:	0bc12083          	lw	ra,188(sp)
   172b0:	0b812403          	lw	s0,184(sp)
   172b4:	0b412483          	lw	s1,180(sp)
   172b8:	0b012903          	lw	s2,176(sp)
   172bc:	0ac12983          	lw	s3,172(sp)
   172c0:	0a812a03          	lw	s4,168(sp)
   172c4:	0a412a83          	lw	s5,164(sp)
   172c8:	0a012b03          	lw	s6,160(sp)
   172cc:	09c12b83          	lw	s7,156(sp)
   172d0:	09812c03          	lw	s8,152(sp)
   172d4:	09412c83          	lw	s9,148(sp)
   172d8:	09012d03          	lw	s10,144(sp)
   172dc:	08c12d83          	lw	s11,140(sp)
   172e0:	0c010113          	addi	sp,sp,192
   172e4:	00008067          	ret
   172e8:	0000a797          	auipc	a5,0xa
   172ec:	9f878793          	addi	a5,a5,-1544 # 20ce0 <blanks.1+0x60>
   172f0:	0007a603          	lw	a2,0(a5)
   172f4:	0047a683          	lw	a3,4(a5)
   172f8:	0087a703          	lw	a4,8(a5)
   172fc:	00c7a783          	lw	a5,12(a5)
   17300:	03010593          	addi	a1,sp,48
   17304:	04010513          	addi	a0,sp,64
   17308:	05412023          	sw	s4,64(sp)
   1730c:	05312223          	sw	s3,68(sp)
   17310:	05212423          	sw	s2,72(sp)
   17314:	04812623          	sw	s0,76(sp)
   17318:	02c12823          	sw	a2,48(sp)
   1731c:	02d12a23          	sw	a3,52(sp)
   17320:	02e12c23          	sw	a4,56(sp)
   17324:	02f12e23          	sw	a5,60(sp)
   17328:	03d060ef          	jal	1db64 <__getf2>
   1732c:	00054e63          	bltz	a0,17348 <_ldtoa_r+0x280>
   17330:	000107b7          	lui	a5,0x10
   17334:	00fdedb3          	or	s11,s11,a5
   17338:	00100793          	li	a5,1
   1733c:	04f12c23          	sw	a5,88(sp)
   17340:	07b12423          	sw	s11,104(sp)
   17344:	f29ff06f          	j	1726c <_ldtoa_r+0x1a4>
   17348:	03010593          	addi	a1,sp,48
   1734c:	04010513          	addi	a0,sp,64
   17350:	05412023          	sw	s4,64(sp)
   17354:	05312223          	sw	s3,68(sp)
   17358:	05212423          	sw	s2,72(sp)
   1735c:	05812623          	sw	s8,76(sp)
   17360:	02012823          	sw	zero,48(sp)
   17364:	02012a23          	sw	zero,52(sp)
   17368:	02012c23          	sw	zero,56(sp)
   1736c:	02012e23          	sw	zero,60(sp)
   17370:	728060ef          	jal	1da98 <__eqtf2>
   17374:	00051663          	bnez	a0,17380 <_ldtoa_r+0x2b8>
   17378:	04012c23          	sw	zero,88(sp)
   1737c:	ef1ff06f          	j	1726c <_ldtoa_r+0x1a4>
   17380:	ffffc4b7          	lui	s1,0xffffc
   17384:	00200793          	li	a5,2
   17388:	f9248493          	addi	s1,s1,-110 # ffffbf92 <__BSS_END__+0xfffd9192>
   1738c:	04f12c23          	sw	a5,88(sp)
   17390:	009a87b3          	add	a5,s5,s1
   17394:	00f12e23          	sw	a5,28(sp)
   17398:	ed5ff06f          	j	1726c <_ldtoa_r+0x1a4>
   1739c:	00400793          	li	a5,4
   173a0:	04f12c23          	sw	a5,88(sp)
   173a4:	ec9ff06f          	j	1726c <_ldtoa_r+0x1a4>

000173a8 <_ldcheck>:
   173a8:	fb010113          	addi	sp,sp,-80
   173ac:	04912223          	sw	s1,68(sp)
   173b0:	00c52483          	lw	s1,12(a0)
   173b4:	05212023          	sw	s2,64(sp)
   173b8:	03312e23          	sw	s3,60(sp)
   173bc:	00852903          	lw	s2,8(a0)
   173c0:	00452983          	lw	s3,4(a0)
   173c4:	03412c23          	sw	s4,56(sp)
   173c8:	00052a03          	lw	s4,0(a0)
   173cc:	00149493          	slli	s1,s1,0x1
   173d0:	0014d493          	srli	s1,s1,0x1
   173d4:	00010593          	mv	a1,sp
   173d8:	01010513          	addi	a0,sp,16
   173dc:	04112623          	sw	ra,76(sp)
   173e0:	01412823          	sw	s4,16(sp)
   173e4:	01312a23          	sw	s3,20(sp)
   173e8:	01212c23          	sw	s2,24(sp)
   173ec:	00912e23          	sw	s1,28(sp)
   173f0:	01412023          	sw	s4,0(sp)
   173f4:	01312223          	sw	s3,4(sp)
   173f8:	01212423          	sw	s2,8(sp)
   173fc:	00912623          	sw	s1,12(sp)
   17400:	659080ef          	jal	20258 <__unordtf2>
   17404:	0a051263          	bnez	a0,174a8 <_ldcheck+0x100>
   17408:	0000a797          	auipc	a5,0xa
   1740c:	8c878793          	addi	a5,a5,-1848 # 20cd0 <blanks.1+0x50>
   17410:	03512a23          	sw	s5,52(sp)
   17414:	03612823          	sw	s6,48(sp)
   17418:	03712623          	sw	s7,44(sp)
   1741c:	03812423          	sw	s8,40(sp)
   17420:	0047ab83          	lw	s7,4(a5)
   17424:	0007ac03          	lw	s8,0(a5)
   17428:	0087ab03          	lw	s6,8(a5)
   1742c:	00c7aa83          	lw	s5,12(a5)
   17430:	00010593          	mv	a1,sp
   17434:	01010513          	addi	a0,sp,16
   17438:	04812423          	sw	s0,72(sp)
   1743c:	01812023          	sw	s8,0(sp)
   17440:	01712223          	sw	s7,4(sp)
   17444:	01612423          	sw	s6,8(sp)
   17448:	01512623          	sw	s5,12(sp)
   1744c:	00100413          	li	s0,1
   17450:	609080ef          	jal	20258 <__unordtf2>
   17454:	04050063          	beqz	a0,17494 <_ldcheck+0xec>
   17458:	04c12083          	lw	ra,76(sp)
   1745c:	00144513          	xori	a0,s0,1
   17460:	04812403          	lw	s0,72(sp)
   17464:	0ff57513          	zext.b	a0,a0
   17468:	03412a83          	lw	s5,52(sp)
   1746c:	03012b03          	lw	s6,48(sp)
   17470:	02c12b83          	lw	s7,44(sp)
   17474:	02812c03          	lw	s8,40(sp)
   17478:	04412483          	lw	s1,68(sp)
   1747c:	04012903          	lw	s2,64(sp)
   17480:	03c12983          	lw	s3,60(sp)
   17484:	03812a03          	lw	s4,56(sp)
   17488:	00151513          	slli	a0,a0,0x1
   1748c:	05010113          	addi	sp,sp,80
   17490:	00008067          	ret
   17494:	00010593          	mv	a1,sp
   17498:	01010513          	addi	a0,sp,16
   1749c:	7f8060ef          	jal	1dc94 <__letf2>
   174a0:	00152413          	slti	s0,a0,1
   174a4:	fb5ff06f          	j	17458 <_ldcheck+0xb0>
   174a8:	04c12083          	lw	ra,76(sp)
   174ac:	04412483          	lw	s1,68(sp)
   174b0:	04012903          	lw	s2,64(sp)
   174b4:	03c12983          	lw	s3,60(sp)
   174b8:	03812a03          	lw	s4,56(sp)
   174bc:	00100513          	li	a0,1
   174c0:	05010113          	addi	sp,sp,80
   174c4:	00008067          	ret

000174c8 <__gdtoa>:
   174c8:	f4010113          	addi	sp,sp,-192
   174cc:	0b612023          	sw	s6,160(sp)
   174d0:	00072b03          	lw	s6,0(a4)
   174d4:	0b212823          	sw	s2,176(sp)
   174d8:	0a112e23          	sw	ra,188(sp)
   174dc:	fcfb7313          	andi	t1,s6,-49
   174e0:	00672023          	sw	t1,0(a4)
   174e4:	00070913          	mv	s2,a4
   174e8:	0c012703          	lw	a4,192(sp)
   174ec:	00fb7313          	andi	t1,s6,15
   174f0:	00300e13          	li	t3,3
   174f4:	00e12423          	sw	a4,8(sp)
   174f8:	00a12023          	sw	a0,0(sp)
   174fc:	00c12823          	sw	a2,16(sp)
   17500:	01012a23          	sw	a6,20(sp)
   17504:	01112223          	sw	a7,4(sp)
   17508:	7fc30663          	beq	t1,t3,17cf4 <__gdtoa+0x82c>
   1750c:	09812c23          	sw	s8,152(sp)
   17510:	00cb7c13          	andi	s8,s6,12
   17514:	720c1e63          	bnez	s8,17c50 <__gdtoa+0x788>
   17518:	78030c63          	beqz	t1,17cb0 <__gdtoa+0x7e8>
   1751c:	0b512223          	sw	s5,164(sp)
   17520:	0005aa83          	lw	s5,0(a1)
   17524:	0b312623          	sw	s3,172(sp)
   17528:	09a12823          	sw	s10,144(sp)
   1752c:	09b12623          	sw	s11,140(sp)
   17530:	00068993          	mv	s3,a3
   17534:	02000693          	li	a3,32
   17538:	00058d93          	mv	s11,a1
   1753c:	00078d13          	mv	s10,a5
   17540:	00000593          	li	a1,0
   17544:	02000793          	li	a5,32
   17548:	0156d863          	bge	a3,s5,17558 <__gdtoa+0x90>
   1754c:	00179793          	slli	a5,a5,0x1
   17550:	00158593          	addi	a1,a1,1
   17554:	ff57cce3          	blt	a5,s5,1754c <__gdtoa+0x84>
   17558:	00012503          	lw	a0,0(sp)
   1755c:	0b4020ef          	jal	19610 <_Balloc>
   17560:	00a12623          	sw	a0,12(sp)
   17564:	3c0508e3          	beqz	a0,18134 <__gdtoa+0xc6c>
   17568:	00c12783          	lw	a5,12(sp)
   1756c:	fffa8513          	addi	a0,s5,-1
   17570:	40555513          	srai	a0,a0,0x5
   17574:	00251593          	slli	a1,a0,0x2
   17578:	01478693          	addi	a3,a5,20
   1757c:	0b412423          	sw	s4,168(sp)
   17580:	09912a23          	sw	s9,148(sp)
   17584:	00b985b3          	add	a1,s3,a1
   17588:	00098793          	mv	a5,s3
   1758c:	0007a703          	lw	a4,0(a5)
   17590:	00478793          	addi	a5,a5,4
   17594:	00468693          	addi	a3,a3,4
   17598:	fee6ae23          	sw	a4,-4(a3)
   1759c:	fef5f8e3          	bgeu	a1,a5,1758c <__gdtoa+0xc4>
   175a0:	00158593          	addi	a1,a1,1
   175a4:	00198793          	addi	a5,s3,1
   175a8:	00400693          	li	a3,4
   175ac:	00f5e663          	bltu	a1,a5,175b8 <__gdtoa+0xf0>
   175b0:	00251513          	slli	a0,a0,0x2
   175b4:	00450693          	addi	a3,a0,4
   175b8:	00c12783          	lw	a5,12(sp)
   175bc:	4026dc93          	srai	s9,a3,0x2
   175c0:	00d786b3          	add	a3,a5,a3
   175c4:	00c0006f          	j	175d0 <__gdtoa+0x108>
   175c8:	ffc68693          	addi	a3,a3,-4
   175cc:	7a0c8063          	beqz	s9,17d6c <__gdtoa+0x8a4>
   175d0:	0106a783          	lw	a5,16(a3)
   175d4:	000c8a13          	mv	s4,s9
   175d8:	fffc8c93          	addi	s9,s9,-1
   175dc:	fe0786e3          	beqz	a5,175c8 <__gdtoa+0x100>
   175e0:	00c12703          	lw	a4,12(sp)
   175e4:	004c8793          	addi	a5,s9,4
   175e8:	00279793          	slli	a5,a5,0x2
   175ec:	00f707b3          	add	a5,a4,a5
   175f0:	0047a503          	lw	a0,4(a5)
   175f4:	01472823          	sw	s4,16(a4)
   175f8:	005a1a13          	slli	s4,s4,0x5
   175fc:	370020ef          	jal	1996c <__hi0bits>
   17600:	40aa0cb3          	sub	s9,s4,a0
   17604:	00c12503          	lw	a0,12(sp)
   17608:	735010ef          	jal	1953c <__trailz_D2A>
   1760c:	01012783          	lw	a5,16(sp)
   17610:	06a12e23          	sw	a0,124(sp)
   17614:	02f12023          	sw	a5,32(sp)
   17618:	78051e63          	bnez	a0,17db4 <__gdtoa+0x8ec>
   1761c:	00c12783          	lw	a5,12(sp)
   17620:	0107a683          	lw	a3,16(a5)
   17624:	66068463          	beqz	a3,17c8c <__gdtoa+0x7c4>
   17628:	00c12503          	lw	a0,12(sp)
   1762c:	07c10593          	addi	a1,sp,124
   17630:	0a812c23          	sw	s0,184(sp)
   17634:	0a912a23          	sw	s1,180(sp)
   17638:	09712e23          	sw	s7,156(sp)
   1763c:	56d020ef          	jal	1a3a8 <__b2d>
   17640:	00c59a13          	slli	s4,a1,0xc
   17644:	00ca5a13          	srli	s4,s4,0xc
   17648:	3ff006b7          	lui	a3,0x3ff00
   1764c:	00da64b3          	or	s1,s4,a3
   17650:	02012703          	lw	a4,32(sp)
   17654:	0000b697          	auipc	a3,0xb
   17658:	32468693          	addi	a3,a3,804 # 22978 <__SDATA_BEGIN__>
   1765c:	0006a603          	lw	a2,0(a3)
   17660:	0046a683          	lw	a3,4(a3)
   17664:	01970733          	add	a4,a4,s9
   17668:	00050793          	mv	a5,a0
   1766c:	00048593          	mv	a1,s1
   17670:	fff70a13          	addi	s4,a4,-1
   17674:	04f12423          	sw	a5,72(sp)
   17678:	00e12e23          	sw	a4,28(sp)
   1767c:	02a12c23          	sw	a0,56(sp)
   17680:	349050ef          	jal	1d1c8 <__subdf3>
   17684:	0000b697          	auipc	a3,0xb
   17688:	2fc68693          	addi	a3,a3,764 # 22980 <__SDATA_BEGIN__+0x8>
   1768c:	0006a603          	lw	a2,0(a3)
   17690:	0046a683          	lw	a3,4(a3)
   17694:	544050ef          	jal	1cbd8 <__muldf3>
   17698:	0000b697          	auipc	a3,0xb
   1769c:	2f068693          	addi	a3,a3,752 # 22988 <__SDATA_BEGIN__+0x10>
   176a0:	0006a603          	lw	a2,0(a3)
   176a4:	0046a683          	lw	a3,4(a3)
   176a8:	478040ef          	jal	1bb20 <__adddf3>
   176ac:	00050b93          	mv	s7,a0
   176b0:	000a0513          	mv	a0,s4
   176b4:	00058413          	mv	s0,a1
   176b8:	338060ef          	jal	1d9f0 <__floatsidf>
   176bc:	0000b697          	auipc	a3,0xb
   176c0:	2d468693          	addi	a3,a3,724 # 22990 <__SDATA_BEGIN__+0x18>
   176c4:	0006a603          	lw	a2,0(a3)
   176c8:	0046a683          	lw	a3,4(a3)
   176cc:	50c050ef          	jal	1cbd8 <__muldf3>
   176d0:	00050613          	mv	a2,a0
   176d4:	00058693          	mv	a3,a1
   176d8:	000b8513          	mv	a0,s7
   176dc:	00040593          	mv	a1,s0
   176e0:	440040ef          	jal	1bb20 <__adddf3>
   176e4:	00050b93          	mv	s7,a0
   176e8:	00058413          	mv	s0,a1
   176ec:	000a0513          	mv	a0,s4
   176f0:	000a5863          	bgez	s4,17700 <__gdtoa+0x238>
   176f4:	01c12703          	lw	a4,28(sp)
   176f8:	00100513          	li	a0,1
   176fc:	40e50533          	sub	a0,a0,a4
   17700:	bcb50513          	addi	a0,a0,-1077
   17704:	02a05c63          	blez	a0,1773c <__gdtoa+0x274>
   17708:	2e8060ef          	jal	1d9f0 <__floatsidf>
   1770c:	0000b697          	auipc	a3,0xb
   17710:	28c68693          	addi	a3,a3,652 # 22998 <__SDATA_BEGIN__+0x20>
   17714:	0006a603          	lw	a2,0(a3)
   17718:	0046a683          	lw	a3,4(a3)
   1771c:	4bc050ef          	jal	1cbd8 <__muldf3>
   17720:	00050613          	mv	a2,a0
   17724:	00058693          	mv	a3,a1
   17728:	000b8513          	mv	a0,s7
   1772c:	00040593          	mv	a1,s0
   17730:	3f0040ef          	jal	1bb20 <__adddf3>
   17734:	00050b93          	mv	s7,a0
   17738:	00058413          	mv	s0,a1
   1773c:	00040593          	mv	a1,s0
   17740:	000b8513          	mv	a0,s7
   17744:	22c060ef          	jal	1d970 <__fixdfsi>
   17748:	00050813          	mv	a6,a0
   1774c:	00040593          	mv	a1,s0
   17750:	000b8513          	mv	a0,s7
   17754:	00000613          	li	a2,0
   17758:	00000693          	li	a3,0
   1775c:	01012c23          	sw	a6,24(sp)
   17760:	39c050ef          	jal	1cafc <__ledf2>
   17764:	02055463          	bgez	a0,1778c <__gdtoa+0x2c4>
   17768:	01812503          	lw	a0,24(sp)
   1776c:	284060ef          	jal	1d9f0 <__floatsidf>
   17770:	000b8613          	mv	a2,s7
   17774:	00040693          	mv	a3,s0
   17778:	21c050ef          	jal	1c994 <__eqdf2>
   1777c:	01812783          	lw	a5,24(sp)
   17780:	00a03533          	snez	a0,a0
   17784:	40a787b3          	sub	a5,a5,a0
   17788:	00f12c23          	sw	a5,24(sp)
   1778c:	014a1613          	slli	a2,s4,0x14
   17790:	01812783          	lw	a5,24(sp)
   17794:	00960833          	add	a6,a2,s1
   17798:	414c8433          	sub	s0,s9,s4
   1779c:	01600693          	li	a3,22
   177a0:	05012023          	sw	a6,64(sp)
   177a4:	fff40b93          	addi	s7,s0,-1
   177a8:	5cf6e863          	bltu	a3,a5,17d78 <__gdtoa+0x8b0>
   177ac:	00009317          	auipc	t1,0x9
   177b0:	74430313          	addi	t1,t1,1860 # 20ef0 <__mprec_tens>
   177b4:	00379693          	slli	a3,a5,0x3
   177b8:	00d306b3          	add	a3,t1,a3
   177bc:	03812883          	lw	a7,56(sp)
   177c0:	0006a503          	lw	a0,0(a3)
   177c4:	0046a583          	lw	a1,4(a3)
   177c8:	00088613          	mv	a2,a7
   177cc:	00080693          	mv	a3,a6
   177d0:	250050ef          	jal	1ca20 <__gedf2>
   177d4:	1aa044e3          	bgtz	a0,1817c <__gdtoa+0xcb4>
   177d8:	02012e23          	sw	zero,60(sp)
   177dc:	02012223          	sw	zero,36(sp)
   177e0:	00804a63          	bgtz	s0,177f4 <__gdtoa+0x32c>
   177e4:	00100693          	li	a3,1
   177e8:	408687b3          	sub	a5,a3,s0
   177ec:	02f12223          	sw	a5,36(sp)
   177f0:	00000b93          	li	s7,0
   177f4:	01812783          	lw	a5,24(sp)
   177f8:	02012623          	sw	zero,44(sp)
   177fc:	00fb8bb3          	add	s7,s7,a5
   17800:	02f12823          	sw	a5,48(sp)
   17804:	00900693          	li	a3,9
   17808:	5da6e863          	bltu	a3,s10,17dd8 <__gdtoa+0x910>
   1780c:	00500693          	li	a3,5
   17810:	19a6c0e3          	blt	a3,s10,18190 <__gdtoa+0xcc8>
   17814:	01c12783          	lw	a5,28(sp)
   17818:	00400613          	li	a2,4
   1781c:	3fd78a13          	addi	s4,a5,1021
   17820:	7f8a3a13          	sltiu	s4,s4,2040
   17824:	00cd1463          	bne	s10,a2,1782c <__gdtoa+0x364>
   17828:	0a00106f          	j	188c8 <__gdtoa+0x1400>
   1782c:	00dd1463          	bne	s10,a3,17834 <__gdtoa+0x36c>
   17830:	08c0106f          	j	188bc <__gdtoa+0x13f4>
   17834:	00200693          	li	a3,2
   17838:	00dd1463          	bne	s10,a3,17840 <__gdtoa+0x378>
   1783c:	0980106f          	j	188d4 <__gdtoa+0x140c>
   17840:	00300693          	li	a3,3
   17844:	02012a23          	sw	zero,52(sp)
   17848:	5add1063          	bne	s10,a3,17de8 <__gdtoa+0x920>
   1784c:	03012783          	lw	a5,48(sp)
   17850:	01412703          	lw	a4,20(sp)
   17854:	00e787b3          	add	a5,a5,a4
   17858:	04f12623          	sw	a5,76(sp)
   1785c:	00178793          	addi	a5,a5,1
   17860:	00f12e23          	sw	a5,28(sp)
   17864:	62f058e3          	blez	a5,18694 <__gdtoa+0x11cc>
   17868:	00078693          	mv	a3,a5
   1786c:	00078593          	mv	a1,a5
   17870:	00012503          	lw	a0,0(sp)
   17874:	06d12e23          	sw	a3,124(sp)
   17878:	0cd010ef          	jal	19144 <__rv_alloc_D2A>
   1787c:	00050f13          	mv	t5,a0
   17880:	4c050063          	beqz	a0,17d40 <__gdtoa+0x878>
   17884:	00cda783          	lw	a5,12(s11)
   17888:	fff78793          	addi	a5,a5,-1
   1788c:	02f12423          	sw	a5,40(sp)
   17890:	5a078663          	beqz	a5,17e3c <__gdtoa+0x974>
   17894:	0c07cee3          	bltz	a5,18170 <__gdtoa+0xca8>
   17898:	100b7e93          	andi	t4,s6,256
   1789c:	580e9863          	bnez	t4,17e2c <__gdtoa+0x964>
   178a0:	02012783          	lw	a5,32(sp)
   178a4:	0007c863          	bltz	a5,178b4 <__gdtoa+0x3ec>
   178a8:	03012783          	lw	a5,48(sp)
   178ac:	00e00693          	li	a3,14
   178b0:	10f6d4e3          	bge	a3,a5,181b8 <__gdtoa+0xcf0>
   178b4:	03412783          	lw	a5,52(sp)
   178b8:	0a0784e3          	beqz	a5,18160 <__gdtoa+0xc98>
   178bc:	02012783          	lw	a5,32(sp)
   178c0:	419a8e33          	sub	t3,s5,s9
   178c4:	004da683          	lw	a3,4(s11)
   178c8:	001e0613          	addi	a2,t3,1
   178cc:	06c12e23          	sw	a2,124(sp)
   178d0:	41c78e33          	sub	t3,a5,t3
   178d4:	62de52e3          	bge	t3,a3,186f8 <__gdtoa+0x1230>
   178d8:	ffdd0613          	addi	a2,s10,-3
   178dc:	ffd67613          	andi	a2,a2,-3
   178e0:	500604e3          	beqz	a2,185e8 <__gdtoa+0x1120>
   178e4:	40d786b3          	sub	a3,a5,a3
   178e8:	00168693          	addi	a3,a3,1
   178ec:	06d12e23          	sw	a3,124(sp)
   178f0:	00100613          	li	a2,1
   178f4:	01a65c63          	bge	a2,s10,1790c <__gdtoa+0x444>
   178f8:	01c12783          	lw	a5,28(sp)
   178fc:	00f05863          	blez	a5,1790c <__gdtoa+0x444>
   17900:	01c12783          	lw	a5,28(sp)
   17904:	00d7d463          	bge	a5,a3,1790c <__gdtoa+0x444>
   17908:	6400106f          	j	18f48 <__gdtoa+0x1a80>
   1790c:	02412783          	lw	a5,36(sp)
   17910:	02c12483          	lw	s1,44(sp)
   17914:	00db8bb3          	add	s7,s7,a3
   17918:	00078a93          	mv	s5,a5
   1791c:	00f687b3          	add	a5,a3,a5
   17920:	02f12223          	sw	a5,36(sp)
   17924:	00012503          	lw	a0,0(sp)
   17928:	00100593          	li	a1,1
   1792c:	03e12023          	sw	t5,32(sp)
   17930:	194020ef          	jal	19ac4 <__i2b>
   17934:	02012f03          	lw	t5,32(sp)
   17938:	00050a13          	mv	s4,a0
   1793c:	40050263          	beqz	a0,17d40 <__gdtoa+0x878>
   17940:	020a8663          	beqz	s5,1796c <__gdtoa+0x4a4>
   17944:	03705463          	blez	s7,1796c <__gdtoa+0x4a4>
   17948:	000a8693          	mv	a3,s5
   1794c:	015bd463          	bge	s7,s5,17954 <__gdtoa+0x48c>
   17950:	000b8693          	mv	a3,s7
   17954:	02412783          	lw	a5,36(sp)
   17958:	06d12e23          	sw	a3,124(sp)
   1795c:	40da8ab3          	sub	s5,s5,a3
   17960:	40d787b3          	sub	a5,a5,a3
   17964:	02f12223          	sw	a5,36(sp)
   17968:	40db8bb3          	sub	s7,s7,a3
   1796c:	02c12783          	lw	a5,44(sp)
   17970:	02078863          	beqz	a5,179a0 <__gdtoa+0x4d8>
   17974:	03412783          	lw	a5,52(sp)
   17978:	00078463          	beqz	a5,17980 <__gdtoa+0x4b8>
   1797c:	660490e3          	bnez	s1,187dc <__gdtoa+0x1314>
   17980:	02c12603          	lw	a2,44(sp)
   17984:	00c12583          	lw	a1,12(sp)
   17988:	00012503          	lw	a0,0(sp)
   1798c:	03e12023          	sw	t5,32(sp)
   17990:	418020ef          	jal	19da8 <__pow5mult>
   17994:	00a12623          	sw	a0,12(sp)
   17998:	02012f03          	lw	t5,32(sp)
   1799c:	3a050263          	beqz	a0,17d40 <__gdtoa+0x878>
   179a0:	00012503          	lw	a0,0(sp)
   179a4:	00100593          	li	a1,1
   179a8:	03e12023          	sw	t5,32(sp)
   179ac:	118020ef          	jal	19ac4 <__i2b>
   179b0:	00050313          	mv	t1,a0
   179b4:	38050663          	beqz	a0,17d40 <__gdtoa+0x878>
   179b8:	01812783          	lw	a5,24(sp)
   179bc:	02012f03          	lw	t5,32(sp)
   179c0:	4e0790e3          	bnez	a5,186a0 <__gdtoa+0x11d8>
   179c4:	00100693          	li	a3,1
   179c8:	1da6d0e3          	bge	a3,s10,18388 <__gdtoa+0xec0>
   179cc:	01f00b13          	li	s6,31
   179d0:	02412783          	lw	a5,36(sp)
   179d4:	417b0b33          	sub	s6,s6,s7
   179d8:	ffcb0b13          	addi	s6,s6,-4
   179dc:	01fb7b13          	andi	s6,s6,31
   179e0:	00fb0633          	add	a2,s6,a5
   179e4:	07612e23          	sw	s6,124(sp)
   179e8:	000b0793          	mv	a5,s6
   179ec:	2ac044e3          	bgtz	a2,18494 <__gdtoa+0xfcc>
   179f0:	00fb8633          	add	a2,s7,a5
   179f4:	32c04863          	bgtz	a2,17d24 <__gdtoa+0x85c>
   179f8:	03c12783          	lw	a5,60(sp)
   179fc:	2c0792e3          	bnez	a5,184c0 <__gdtoa+0xff8>
   17a00:	01c12783          	lw	a5,28(sp)
   17a04:	40f058e3          	blez	a5,18614 <__gdtoa+0x114c>
   17a08:	03412783          	lw	a5,52(sp)
   17a0c:	30078ee3          	beqz	a5,18528 <__gdtoa+0x1060>
   17a10:	015b0633          	add	a2,s6,s5
   17a14:	66c048e3          	bgtz	a2,18884 <__gdtoa+0x13bc>
   17a18:	01812783          	lw	a5,24(sp)
   17a1c:	000a0a93          	mv	s5,s4
   17a20:	00078463          	beqz	a5,17a28 <__gdtoa+0x560>
   17a24:	6e50006f          	j	18908 <__gdtoa+0x1440>
   17a28:	01212a23          	sw	s2,20(sp)
   17a2c:	00c12483          	lw	s1,12(sp)
   17a30:	00012903          	lw	s2,0(sp)
   17a34:	000f0d93          	mv	s11,t5
   17a38:	00100793          	li	a5,1
   17a3c:	00200413          	li	s0,2
   17a40:	00030c13          	mv	s8,t1
   17a44:	01e12823          	sw	t5,16(sp)
   17a48:	0b00006f          	j	17af8 <__gdtoa+0x630>
   17a4c:	00090513          	mv	a0,s2
   17a50:	475010ef          	jal	196c4 <_Bfree>
   17a54:	000b5463          	bgez	s6,17a5c <__gdtoa+0x594>
   17a58:	06c0106f          	j	18ac4 <__gdtoa+0x15fc>
   17a5c:	016d6b33          	or	s6,s10,s6
   17a60:	000b1a63          	bnez	s6,17a74 <__gdtoa+0x5ac>
   17a64:	0009a783          	lw	a5,0(s3)
   17a68:	0017f793          	andi	a5,a5,1
   17a6c:	00079463          	bnez	a5,17a74 <__gdtoa+0x5ac>
   17a70:	0540106f          	j	18ac4 <__gdtoa+0x15fc>
   17a74:	02812783          	lw	a5,40(sp)
   17a78:	00878463          	beq	a5,s0,17a80 <__gdtoa+0x5b8>
   17a7c:	3c80106f          	j	18e44 <__gdtoa+0x197c>
   17a80:	019d8023          	sb	s9,0(s11)
   17a84:	07c12783          	lw	a5,124(sp)
   17a88:	01c12703          	lw	a4,28(sp)
   17a8c:	001d8d93          	addi	s11,s11,1
   17a90:	00f71463          	bne	a4,a5,17a98 <__gdtoa+0x5d0>
   17a94:	2d40106f          	j	18d68 <__gdtoa+0x18a0>
   17a98:	00048593          	mv	a1,s1
   17a9c:	00000693          	li	a3,0
   17aa0:	00a00613          	li	a2,10
   17aa4:	00090513          	mv	a0,s2
   17aa8:	441010ef          	jal	196e8 <__multadd>
   17aac:	00050493          	mv	s1,a0
   17ab0:	28050863          	beqz	a0,17d40 <__gdtoa+0x878>
   17ab4:	00000693          	li	a3,0
   17ab8:	00a00613          	li	a2,10
   17abc:	000a0593          	mv	a1,s4
   17ac0:	00090513          	mv	a0,s2
   17ac4:	135a00e3          	beq	s4,s5,183e4 <__gdtoa+0xf1c>
   17ac8:	421010ef          	jal	196e8 <__multadd>
   17acc:	00050a13          	mv	s4,a0
   17ad0:	26050863          	beqz	a0,17d40 <__gdtoa+0x878>
   17ad4:	000a8593          	mv	a1,s5
   17ad8:	00000693          	li	a3,0
   17adc:	00a00613          	li	a2,10
   17ae0:	00090513          	mv	a0,s2
   17ae4:	405010ef          	jal	196e8 <__multadd>
   17ae8:	00050a93          	mv	s5,a0
   17aec:	24050a63          	beqz	a0,17d40 <__gdtoa+0x878>
   17af0:	07c12783          	lw	a5,124(sp)
   17af4:	00178793          	addi	a5,a5,1
   17af8:	000c0593          	mv	a1,s8
   17afc:	00048513          	mv	a0,s1
   17b00:	06f12e23          	sw	a5,124(sp)
   17b04:	75c010ef          	jal	19260 <__quorem_D2A>
   17b08:	00050b93          	mv	s7,a0
   17b0c:	000a0593          	mv	a1,s4
   17b10:	00048513          	mv	a0,s1
   17b14:	574020ef          	jal	1a088 <__mcmp>
   17b18:	000c0593          	mv	a1,s8
   17b1c:	00050b13          	mv	s6,a0
   17b20:	000a8613          	mv	a2,s5
   17b24:	00090513          	mv	a0,s2
   17b28:	5b8020ef          	jal	1a0e0 <__mdiff>
   17b2c:	030b8c93          	addi	s9,s7,48
   17b30:	00050593          	mv	a1,a0
   17b34:	20050663          	beqz	a0,17d40 <__gdtoa+0x878>
   17b38:	00c52783          	lw	a5,12(a0)
   17b3c:	f00798e3          	bnez	a5,17a4c <__gdtoa+0x584>
   17b40:	00a12623          	sw	a0,12(sp)
   17b44:	00048513          	mv	a0,s1
   17b48:	540020ef          	jal	1a088 <__mcmp>
   17b4c:	00c12583          	lw	a1,12(sp)
   17b50:	00050693          	mv	a3,a0
   17b54:	00090513          	mv	a0,s2
   17b58:	00d12623          	sw	a3,12(sp)
   17b5c:	369010ef          	jal	196c4 <_Bfree>
   17b60:	00c12683          	lw	a3,12(sp)
   17b64:	00dd6733          	or	a4,s10,a3
   17b68:	00070463          	beqz	a4,17b70 <__gdtoa+0x6a8>
   17b6c:	5680106f          	j	190d4 <__gdtoa+0x1c0c>
   17b70:	0009a783          	lw	a5,0(s3)
   17b74:	0017f793          	andi	a5,a5,1
   17b78:	080790e3          	bnez	a5,183f8 <__gdtoa+0xf30>
   17b7c:	02812783          	lw	a5,40(sp)
   17b80:	00079463          	bnez	a5,17b88 <__gdtoa+0x6c0>
   17b84:	4280106f          	j	18fac <__gdtoa+0x1ae4>
   17b88:	ef604ce3          	bgtz	s6,17a80 <__gdtoa+0x5b8>
   17b8c:	0104a603          	lw	a2,16(s1)
   17b90:	00912623          	sw	s1,12(sp)
   17b94:	00100693          	li	a3,1
   17b98:	000c0313          	mv	t1,s8
   17b9c:	01012f03          	lw	t5,16(sp)
   17ba0:	01412903          	lw	s2,20(sp)
   17ba4:	00070c13          	mv	s8,a4
   17ba8:	00048793          	mv	a5,s1
   17bac:	00c6c463          	blt	a3,a2,17bb4 <__gdtoa+0x6ec>
   17bb0:	4ec0106f          	j	1909c <__gdtoa+0x1bd4>
   17bb4:	02812783          	lw	a5,40(sp)
   17bb8:	00200693          	li	a3,2
   17bbc:	00d79463          	bne	a5,a3,17bc4 <__gdtoa+0x6fc>
   17bc0:	4ac0106f          	j	1906c <__gdtoa+0x1ba4>
   17bc4:	00c12483          	lw	s1,12(sp)
   17bc8:	00012b83          	lw	s7,0(sp)
   17bcc:	00030b13          	mv	s6,t1
   17bd0:	000f0c13          	mv	s8,t5
   17bd4:	0240006f          	j	17bf8 <__gdtoa+0x730>
   17bd8:	311010ef          	jal	196e8 <__multadd>
   17bdc:	000b0593          	mv	a1,s6
   17be0:	00050493          	mv	s1,a0
   17be4:	14050e63          	beqz	a0,17d40 <__gdtoa+0x878>
   17be8:	678010ef          	jal	19260 <__quorem_D2A>
   17bec:	03050c93          	addi	s9,a0,48
   17bf0:	00098d93          	mv	s11,s3
   17bf4:	00040a93          	mv	s5,s0
   17bf8:	000a8593          	mv	a1,s5
   17bfc:	000b0513          	mv	a0,s6
   17c00:	488020ef          	jal	1a088 <__mcmp>
   17c04:	00050793          	mv	a5,a0
   17c08:	00000693          	li	a3,0
   17c0c:	00a00613          	li	a2,10
   17c10:	000a8593          	mv	a1,s5
   17c14:	000b8513          	mv	a0,s7
   17c18:	001d8993          	addi	s3,s11,1
   17c1c:	00f04463          	bgtz	a5,17c24 <__gdtoa+0x75c>
   17c20:	42c0106f          	j	1904c <__gdtoa+0x1b84>
   17c24:	ff998fa3          	sb	s9,-1(s3)
   17c28:	2c1010ef          	jal	196e8 <__multadd>
   17c2c:	00050413          	mv	s0,a0
   17c30:	00000693          	li	a3,0
   17c34:	00a00613          	li	a2,10
   17c38:	00048593          	mv	a1,s1
   17c3c:	000b8513          	mv	a0,s7
   17c40:	10040063          	beqz	s0,17d40 <__gdtoa+0x878>
   17c44:	f95a1ae3          	bne	s4,s5,17bd8 <__gdtoa+0x710>
   17c48:	00040a13          	mv	s4,s0
   17c4c:	f8dff06f          	j	17bd8 <__gdtoa+0x710>
   17c50:	00400793          	li	a5,4
   17c54:	10f31863          	bne	t1,a5,17d64 <__gdtoa+0x89c>
   17c58:	00412703          	lw	a4,4(sp)
   17c5c:	00812603          	lw	a2,8(sp)
   17c60:	09812c03          	lw	s8,152(sp)
   17c64:	0bc12083          	lw	ra,188(sp)
   17c68:	0b012903          	lw	s2,176(sp)
   17c6c:	0a012b03          	lw	s6,160(sp)
   17c70:	ffff87b7          	lui	a5,0xffff8
   17c74:	00f72023          	sw	a5,0(a4)
   17c78:	00300693          	li	a3,3
   17c7c:	00009597          	auipc	a1,0x9
   17c80:	d0058593          	addi	a1,a1,-768 # 2097c <_exit+0x144>
   17c84:	0c010113          	addi	sp,sp,192
   17c88:	50c0106f          	j	19194 <__nrv_alloc_D2A>
   17c8c:	00012503          	lw	a0,0(sp)
   17c90:	00078593          	mv	a1,a5
   17c94:	231010ef          	jal	196c4 <_Bfree>
   17c98:	0ac12983          	lw	s3,172(sp)
   17c9c:	0a812a03          	lw	s4,168(sp)
   17ca0:	0a412a83          	lw	s5,164(sp)
   17ca4:	09412c83          	lw	s9,148(sp)
   17ca8:	09012d03          	lw	s10,144(sp)
   17cac:	08c12d83          	lw	s11,140(sp)
   17cb0:	00412703          	lw	a4,4(sp)
   17cb4:	00812603          	lw	a2,8(sp)
   17cb8:	00012503          	lw	a0,0(sp)
   17cbc:	00100793          	li	a5,1
   17cc0:	00f72023          	sw	a5,0(a4)
   17cc4:	00100693          	li	a3,1
   17cc8:	00009597          	auipc	a1,0x9
   17ccc:	c9458593          	addi	a1,a1,-876 # 2095c <_exit+0x124>
   17cd0:	4c4010ef          	jal	19194 <__nrv_alloc_D2A>
   17cd4:	00050f13          	mv	t5,a0
   17cd8:	0bc12083          	lw	ra,188(sp)
   17cdc:	09812c03          	lw	s8,152(sp)
   17ce0:	0b012903          	lw	s2,176(sp)
   17ce4:	0a012b03          	lw	s6,160(sp)
   17ce8:	000f0513          	mv	a0,t5
   17cec:	0c010113          	addi	sp,sp,192
   17cf0:	00008067          	ret
   17cf4:	00412703          	lw	a4,4(sp)
   17cf8:	00812603          	lw	a2,8(sp)
   17cfc:	0bc12083          	lw	ra,188(sp)
   17d00:	0b012903          	lw	s2,176(sp)
   17d04:	0a012b03          	lw	s6,160(sp)
   17d08:	ffff87b7          	lui	a5,0xffff8
   17d0c:	00f72023          	sw	a5,0(a4)
   17d10:	00800693          	li	a3,8
   17d14:	00009597          	auipc	a1,0x9
   17d18:	c5c58593          	addi	a1,a1,-932 # 20970 <_exit+0x138>
   17d1c:	0c010113          	addi	sp,sp,192
   17d20:	4740106f          	j	19194 <__nrv_alloc_D2A>
   17d24:	00012503          	lw	a0,0(sp)
   17d28:	00030593          	mv	a1,t1
   17d2c:	01e12823          	sw	t5,16(sp)
   17d30:	1c8020ef          	jal	19ef8 <__lshift>
   17d34:	01012f03          	lw	t5,16(sp)
   17d38:	00050313          	mv	t1,a0
   17d3c:	ca051ee3          	bnez	a0,179f8 <__gdtoa+0x530>
   17d40:	0b812403          	lw	s0,184(sp)
   17d44:	0b412483          	lw	s1,180(sp)
   17d48:	0ac12983          	lw	s3,172(sp)
   17d4c:	0a812a03          	lw	s4,168(sp)
   17d50:	0a412a83          	lw	s5,164(sp)
   17d54:	09c12b83          	lw	s7,156(sp)
   17d58:	09412c83          	lw	s9,148(sp)
   17d5c:	09012d03          	lw	s10,144(sp)
   17d60:	08c12d83          	lw	s11,140(sp)
   17d64:	00000f13          	li	t5,0
   17d68:	f71ff06f          	j	17cd8 <__gdtoa+0x810>
   17d6c:	00c12783          	lw	a5,12(sp)
   17d70:	0007a823          	sw	zero,16(a5) # ffff8010 <__BSS_END__+0xfffd5210>
   17d74:	891ff06f          	j	17604 <__gdtoa+0x13c>
   17d78:	00100793          	li	a5,1
   17d7c:	02f12e23          	sw	a5,60(sp)
   17d80:	02012223          	sw	zero,36(sp)
   17d84:	3c0bc463          	bltz	s7,1814c <__gdtoa+0xc84>
   17d88:	01812783          	lw	a5,24(sp)
   17d8c:	a607d4e3          	bgez	a5,177f4 <__gdtoa+0x32c>
   17d90:	01812783          	lw	a5,24(sp)
   17d94:	02412703          	lw	a4,36(sp)
   17d98:	00012c23          	sw	zero,24(sp)
   17d9c:	02f12823          	sw	a5,48(sp)
   17da0:	40f70733          	sub	a4,a4,a5
   17da4:	02e12223          	sw	a4,36(sp)
   17da8:	40f00733          	neg	a4,a5
   17dac:	02e12623          	sw	a4,44(sp)
   17db0:	a55ff06f          	j	17804 <__gdtoa+0x33c>
   17db4:	00050593          	mv	a1,a0
   17db8:	00c12503          	lw	a0,12(sp)
   17dbc:	69c010ef          	jal	19458 <__rshift_D2A>
   17dc0:	07c12683          	lw	a3,124(sp)
   17dc4:	01012783          	lw	a5,16(sp)
   17dc8:	40dc8cb3          	sub	s9,s9,a3
   17dcc:	00f687b3          	add	a5,a3,a5
   17dd0:	02f12023          	sw	a5,32(sp)
   17dd4:	849ff06f          	j	1761c <__gdtoa+0x154>
   17dd8:	01c12783          	lw	a5,28(sp)
   17ddc:	00000d13          	li	s10,0
   17de0:	3fd78793          	addi	a5,a5,1021
   17de4:	7f87ba13          	sltiu	s4,a5,2040
   17de8:	000a8513          	mv	a0,s5
   17dec:	405050ef          	jal	1d9f0 <__floatsidf>
   17df0:	0000b697          	auipc	a3,0xb
   17df4:	bb068693          	addi	a3,a3,-1104 # 229a0 <__SDATA_BEGIN__+0x28>
   17df8:	0006a603          	lw	a2,0(a3)
   17dfc:	0046a683          	lw	a3,4(a3)
   17e00:	5d9040ef          	jal	1cbd8 <__muldf3>
   17e04:	36d050ef          	jal	1d970 <__fixdfsi>
   17e08:	00100793          	li	a5,1
   17e0c:	00350593          	addi	a1,a0,3
   17e10:	02f12a23          	sw	a5,52(sp)
   17e14:	fff00793          	li	a5,-1
   17e18:	00058693          	mv	a3,a1
   17e1c:	00012a23          	sw	zero,20(sp)
   17e20:	04f12623          	sw	a5,76(sp)
   17e24:	00f12e23          	sw	a5,28(sp)
   17e28:	a49ff06f          	j	17870 <__gdtoa+0x3a8>
   17e2c:	02812783          	lw	a5,40(sp)
   17e30:	00300693          	li	a3,3
   17e34:	40f687b3          	sub	a5,a3,a5
   17e38:	02f12423          	sw	a5,40(sp)
   17e3c:	01c12483          	lw	s1,28(sp)
   17e40:	00e00693          	li	a3,14
   17e44:	a496eee3          	bltu	a3,s1,178a0 <__gdtoa+0x3d8>
   17e48:	a40a0ce3          	beqz	s4,178a0 <__gdtoa+0x3d8>
   17e4c:	03012783          	lw	a5,48(sp)
   17e50:	02812703          	lw	a4,40(sp)
   17e54:	00e7e6b3          	or	a3,a5,a4
   17e58:	a40694e3          	bnez	a3,178a0 <__gdtoa+0x3d8>
   17e5c:	03812403          	lw	s0,56(sp)
   17e60:	04012a03          	lw	s4,64(sp)
   17e64:	03c12783          	lw	a5,60(sp)
   17e68:	06012e23          	sw	zero,124(sp)
   17e6c:	02812423          	sw	s0,40(sp)
   17e70:	03412c23          	sw	s4,56(sp)
   17e74:	02078863          	beqz	a5,17ea4 <__gdtoa+0x9dc>
   17e78:	0000b697          	auipc	a3,0xb
   17e7c:	b3068693          	addi	a3,a3,-1232 # 229a8 <__SDATA_BEGIN__+0x30>
   17e80:	0006a603          	lw	a2,0(a3)
   17e84:	0046a683          	lw	a3,4(a3)
   17e88:	00040513          	mv	a0,s0
   17e8c:	000a0593          	mv	a1,s4
   17e90:	05e12823          	sw	t5,80(sp)
   17e94:	469040ef          	jal	1cafc <__ledf2>
   17e98:	05012f03          	lw	t5,80(sp)
   17e9c:	00055463          	bgez	a0,17ea4 <__gdtoa+0x9dc>
   17ea0:	6ed0006f          	j	18d8c <__gdtoa+0x18c4>
   17ea4:	02812783          	lw	a5,40(sp)
   17ea8:	05e12823          	sw	t5,80(sp)
   17eac:	00078613          	mv	a2,a5
   17eb0:	00078513          	mv	a0,a5
   17eb4:	03812783          	lw	a5,56(sp)
   17eb8:	00078693          	mv	a3,a5
   17ebc:	00078593          	mv	a1,a5
   17ec0:	461030ef          	jal	1bb20 <__adddf3>
   17ec4:	0000b697          	auipc	a3,0xb
   17ec8:	afc68693          	addi	a3,a3,-1284 # 229c0 <__SDATA_BEGIN__+0x48>
   17ecc:	0006a603          	lw	a2,0(a3)
   17ed0:	0046a683          	lw	a3,4(a3)
   17ed4:	44d030ef          	jal	1bb20 <__adddf3>
   17ed8:	01c12783          	lw	a5,28(sp)
   17edc:	fcc00737          	lui	a4,0xfcc00
   17ee0:	05012f03          	lw	t5,80(sp)
   17ee4:	00050b13          	mv	s6,a0
   17ee8:	00b70a33          	add	s4,a4,a1
   17eec:	2a078ce3          	beqz	a5,189a4 <__gdtoa+0x14dc>
   17ef0:	01c12783          	lw	a5,28(sp)
   17ef4:	02812e83          	lw	t4,40(sp)
   17ef8:	03812803          	lw	a6,56(sp)
   17efc:	04012a23          	sw	zero,84(sp)
   17f00:	04f12823          	sw	a5,80(sp)
   17f04:	05012783          	lw	a5,80(sp)
   17f08:	03412703          	lw	a4,52(sp)
   17f0c:	00009317          	auipc	t1,0x9
   17f10:	fe430313          	addi	t1,t1,-28 # 20ef0 <__mprec_tens>
   17f14:	fff78693          	addi	a3,a5,-1
   17f18:	00369693          	slli	a3,a3,0x3
   17f1c:	00d306b3          	add	a3,t1,a3
   17f20:	05012e23          	sw	a6,92(sp)
   17f24:	05d12c23          	sw	t4,88(sp)
   17f28:	0006a603          	lw	a2,0(a3)
   17f2c:	000b0493          	mv	s1,s6
   17f30:	0046a683          	lw	a3,4(a3)
   17f34:	480704e3          	beqz	a4,18bbc <__gdtoa+0x16f4>
   17f38:	0000b597          	auipc	a1,0xb
   17f3c:	a9858593          	addi	a1,a1,-1384 # 229d0 <__SDATA_BEGIN__+0x58>
   17f40:	0005a503          	lw	a0,0(a1)
   17f44:	0045a583          	lw	a1,4(a1)
   17f48:	06612423          	sw	t1,104(sp)
   17f4c:	001f0b13          	addi	s6,t5,1
   17f50:	07e12023          	sw	t5,96(sp)
   17f54:	360040ef          	jal	1c2b4 <__divdf3>
   17f58:	00048613          	mv	a2,s1
   17f5c:	000a0693          	mv	a3,s4
   17f60:	268050ef          	jal	1d1c8 <__subdf3>
   17f64:	05812e83          	lw	t4,88(sp)
   17f68:	05c12803          	lw	a6,92(sp)
   17f6c:	00050613          	mv	a2,a0
   17f70:	00058693          	mv	a3,a1
   17f74:	000e8513          	mv	a0,t4
   17f78:	00080593          	mv	a1,a6
   17f7c:	05d12423          	sw	t4,72(sp)
   17f80:	05012023          	sw	a6,64(sp)
   17f84:	04c12c23          	sw	a2,88(sp)
   17f88:	04d12e23          	sw	a3,92(sp)
   17f8c:	1e5050ef          	jal	1d970 <__fixdfsi>
   17f90:	00050413          	mv	s0,a0
   17f94:	25d050ef          	jal	1d9f0 <__floatsidf>
   17f98:	04012803          	lw	a6,64(sp)
   17f9c:	04812e83          	lw	t4,72(sp)
   17fa0:	00050613          	mv	a2,a0
   17fa4:	00058693          	mv	a3,a1
   17fa8:	000e8513          	mv	a0,t4
   17fac:	00080593          	mv	a1,a6
   17fb0:	218050ef          	jal	1d1c8 <__subdf3>
   17fb4:	06012f03          	lw	t5,96(sp)
   17fb8:	00050613          	mv	a2,a0
   17fbc:	00058693          	mv	a3,a1
   17fc0:	00050493          	mv	s1,a0
   17fc4:	00058a13          	mv	s4,a1
   17fc8:	05812503          	lw	a0,88(sp)
   17fcc:	05c12583          	lw	a1,92(sp)
   17fd0:	03040793          	addi	a5,s0,48
   17fd4:	00ff0023          	sb	a5,0(t5)
   17fd8:	05e12023          	sw	t5,64(sp)
   17fdc:	245040ef          	jal	1ca20 <__gedf2>
   17fe0:	04012f03          	lw	t5,64(sp)
   17fe4:	00a05463          	blez	a0,17fec <__gdtoa+0xb24>
   17fe8:	71d0006f          	j	18f04 <__gdtoa+0x1a3c>
   17fec:	0000b697          	auipc	a3,0xb
   17ff0:	9bc68693          	addi	a3,a3,-1604 # 229a8 <__SDATA_BEGIN__+0x30>
   17ff4:	0006a783          	lw	a5,0(a3)
   17ff8:	0046a803          	lw	a6,4(a3)
   17ffc:	07712023          	sw	s7,96(sp)
   18000:	07512223          	sw	s5,100(sp)
   18004:	05012b83          	lw	s7,80(sp)
   18008:	05812403          	lw	s0,88(sp)
   1800c:	05c12a83          	lw	s5,92(sp)
   18010:	04f12023          	sw	a5,64(sp)
   18014:	05012223          	sw	a6,68(sp)
   18018:	05e12423          	sw	t5,72(sp)
   1801c:	05212823          	sw	s2,80(sp)
   18020:	0940006f          	j	180b4 <__gdtoa+0xbec>
   18024:	07c12783          	lw	a5,124(sp)
   18028:	00178793          	addi	a5,a5,1
   1802c:	06f12e23          	sw	a5,124(sp)
   18030:	0177c463          	blt	a5,s7,18038 <__gdtoa+0xb70>
   18034:	7450006f          	j	18f78 <__gdtoa+0x1ab0>
   18038:	0000b917          	auipc	s2,0xb
   1803c:	97890913          	addi	s2,s2,-1672 # 229b0 <__SDATA_BEGIN__+0x38>
   18040:	00092603          	lw	a2,0(s2)
   18044:	00492683          	lw	a3,4(s2)
   18048:	001b0b13          	addi	s6,s6,1
   1804c:	38d040ef          	jal	1cbd8 <__muldf3>
   18050:	00092603          	lw	a2,0(s2)
   18054:	00492683          	lw	a3,4(s2)
   18058:	00050413          	mv	s0,a0
   1805c:	00058a93          	mv	s5,a1
   18060:	00048513          	mv	a0,s1
   18064:	000a0593          	mv	a1,s4
   18068:	371040ef          	jal	1cbd8 <__muldf3>
   1806c:	00058a13          	mv	s4,a1
   18070:	00050913          	mv	s2,a0
   18074:	0fd050ef          	jal	1d970 <__fixdfsi>
   18078:	00050493          	mv	s1,a0
   1807c:	175050ef          	jal	1d9f0 <__floatsidf>
   18080:	00050613          	mv	a2,a0
   18084:	00058693          	mv	a3,a1
   18088:	00090513          	mv	a0,s2
   1808c:	000a0593          	mv	a1,s4
   18090:	138050ef          	jal	1d1c8 <__subdf3>
   18094:	03048793          	addi	a5,s1,48
   18098:	00040613          	mv	a2,s0
   1809c:	000a8693          	mv	a3,s5
   180a0:	fefb0fa3          	sb	a5,-1(s6)
   180a4:	00050493          	mv	s1,a0
   180a8:	00058a13          	mv	s4,a1
   180ac:	251040ef          	jal	1cafc <__ledf2>
   180b0:	640546e3          	bltz	a0,18efc <__gdtoa+0x1a34>
   180b4:	04012503          	lw	a0,64(sp)
   180b8:	04412583          	lw	a1,68(sp)
   180bc:	00048613          	mv	a2,s1
   180c0:	000a0693          	mv	a3,s4
   180c4:	104050ef          	jal	1d1c8 <__subdf3>
   180c8:	00050613          	mv	a2,a0
   180cc:	00058693          	mv	a3,a1
   180d0:	00040513          	mv	a0,s0
   180d4:	000a8593          	mv	a1,s5
   180d8:	149040ef          	jal	1ca20 <__gedf2>
   180dc:	00050793          	mv	a5,a0
   180e0:	000a8593          	mv	a1,s5
   180e4:	00040513          	mv	a0,s0
   180e8:	f2f05ee3          	blez	a5,18024 <__gdtoa+0xb5c>
   180ec:	05412783          	lw	a5,84(sp)
   180f0:	04812f03          	lw	t5,72(sp)
   180f4:	05012903          	lw	s2,80(sp)
   180f8:	fffb4703          	lbu	a4,-1(s6)
   180fc:	00178413          	addi	s0,a5,1
   18100:	03900693          	li	a3,57
   18104:	0100006f          	j	18114 <__gdtoa+0xc4c>
   18108:	2aff04e3          	beq	t5,a5,18bb0 <__gdtoa+0x16e8>
   1810c:	fff7c703          	lbu	a4,-1(a5)
   18110:	00078b13          	mv	s6,a5
   18114:	fffb0793          	addi	a5,s6,-1
   18118:	fed708e3          	beq	a4,a3,18108 <__gdtoa+0xc40>
   1811c:	00170693          	addi	a3,a4,1 # fcc00001 <__BSS_END__+0xfcbdd201>
   18120:	0ff6f693          	zext.b	a3,a3
   18124:	00d78023          	sb	a3,0(a5)
   18128:	00040493          	mv	s1,s0
   1812c:	02000c13          	li	s8,32
   18130:	1f80006f          	j	18328 <__gdtoa+0xe60>
   18134:	0ac12983          	lw	s3,172(sp)
   18138:	0a412a83          	lw	s5,164(sp)
   1813c:	09012d03          	lw	s10,144(sp)
   18140:	08c12d83          	lw	s11,140(sp)
   18144:	00000f13          	li	t5,0
   18148:	b91ff06f          	j	17cd8 <__gdtoa+0x810>
   1814c:	00100693          	li	a3,1
   18150:	408687b3          	sub	a5,a3,s0
   18154:	02f12223          	sw	a5,36(sp)
   18158:	00000b93          	li	s7,0
   1815c:	c2dff06f          	j	17d88 <__gdtoa+0x8c0>
   18160:	02c12483          	lw	s1,44(sp)
   18164:	02412a83          	lw	s5,36(sp)
   18168:	00000a13          	li	s4,0
   1816c:	fd4ff06f          	j	17940 <__gdtoa+0x478>
   18170:	00200793          	li	a5,2
   18174:	02f12423          	sw	a5,40(sp)
   18178:	f20ff06f          	j	17898 <__gdtoa+0x3d0>
   1817c:	01812783          	lw	a5,24(sp)
   18180:	02012e23          	sw	zero,60(sp)
   18184:	fff78793          	addi	a5,a5,-1
   18188:	00f12c23          	sw	a5,24(sp)
   1818c:	bf5ff06f          	j	17d80 <__gdtoa+0x8b8>
   18190:	ffcd0d13          	addi	s10,s10,-4
   18194:	00400613          	li	a2,4
   18198:	22cd0063          	beq	s10,a2,183b8 <__gdtoa+0xef0>
   1819c:	70dd0863          	beq	s10,a3,188ac <__gdtoa+0x13e4>
   181a0:	00200693          	li	a3,2
   181a4:	02012a23          	sw	zero,52(sp)
   181a8:	00000a13          	li	s4,0
   181ac:	20dd0c63          	beq	s10,a3,183c4 <__gdtoa+0xefc>
   181b0:	00300d13          	li	s10,3
   181b4:	e98ff06f          	j	1784c <__gdtoa+0x384>
   181b8:	00379693          	slli	a3,a5,0x3
   181bc:	00009797          	auipc	a5,0x9
   181c0:	d3478793          	addi	a5,a5,-716 # 20ef0 <__mprec_tens>
   181c4:	00d787b3          	add	a5,a5,a3
   181c8:	0007ad03          	lw	s10,0(a5)
   181cc:	0047ad83          	lw	s11,4(a5)
   181d0:	01412783          	lw	a5,20(sp)
   181d4:	5607c463          	bltz	a5,1873c <__gdtoa+0x1274>
   181d8:	04812403          	lw	s0,72(sp)
   181dc:	04012a03          	lw	s4,64(sp)
   181e0:	00100793          	li	a5,1
   181e4:	000d0613          	mv	a2,s10
   181e8:	000d8693          	mv	a3,s11
   181ec:	00040513          	mv	a0,s0
   181f0:	000a0593          	mv	a1,s4
   181f4:	01e12823          	sw	t5,16(sp)
   181f8:	06f12e23          	sw	a5,124(sp)
   181fc:	0b8040ef          	jal	1c2b4 <__divdf3>
   18200:	770050ef          	jal	1d970 <__fixdfsi>
   18204:	00050993          	mv	s3,a0
   18208:	7e8050ef          	jal	1d9f0 <__floatsidf>
   1820c:	000d0613          	mv	a2,s10
   18210:	000d8693          	mv	a3,s11
   18214:	1c5040ef          	jal	1cbd8 <__muldf3>
   18218:	00050613          	mv	a2,a0
   1821c:	00058693          	mv	a3,a1
   18220:	00040513          	mv	a0,s0
   18224:	000a0593          	mv	a1,s4
   18228:	7a1040ef          	jal	1d1c8 <__subdf3>
   1822c:	01012f03          	lw	t5,16(sp)
   18230:	03012703          	lw	a4,48(sp)
   18234:	03098793          	addi	a5,s3,48
   18238:	00ff0023          	sb	a5,0(t5)
   1823c:	00000613          	li	a2,0
   18240:	00000693          	li	a3,0
   18244:	00170413          	addi	s0,a4,1
   18248:	001f0b13          	addi	s6,t5,1
   1824c:	00050a93          	mv	s5,a0
   18250:	00058a13          	mv	s4,a1
   18254:	740040ef          	jal	1c994 <__eqdf2>
   18258:	01012f03          	lw	t5,16(sp)
   1825c:	00040493          	mv	s1,s0
   18260:	0c050463          	beqz	a0,18328 <__gdtoa+0xe60>
   18264:	01812823          	sw	s8,16(sp)
   18268:	01c12c83          	lw	s9,28(sp)
   1826c:	000a0c13          	mv	s8,s4
   18270:	0000ab97          	auipc	s7,0xa
   18274:	740b8b93          	addi	s7,s7,1856 # 229b0 <__SDATA_BEGIN__+0x38>
   18278:	00040a13          	mv	s4,s0
   1827c:	000f0413          	mv	s0,t5
   18280:	0780006f          	j	182f8 <__gdtoa+0xe30>
   18284:	000ba603          	lw	a2,0(s7)
   18288:	004ba683          	lw	a3,4(s7)
   1828c:	07012e23          	sw	a6,124(sp)
   18290:	001b0b13          	addi	s6,s6,1
   18294:	145040ef          	jal	1cbd8 <__muldf3>
   18298:	000d0613          	mv	a2,s10
   1829c:	000d8693          	mv	a3,s11
   182a0:	00050c13          	mv	s8,a0
   182a4:	00058a93          	mv	s5,a1
   182a8:	00c040ef          	jal	1c2b4 <__divdf3>
   182ac:	6c4050ef          	jal	1d970 <__fixdfsi>
   182b0:	00050993          	mv	s3,a0
   182b4:	73c050ef          	jal	1d9f0 <__floatsidf>
   182b8:	000d0613          	mv	a2,s10
   182bc:	000d8693          	mv	a3,s11
   182c0:	119040ef          	jal	1cbd8 <__muldf3>
   182c4:	00050613          	mv	a2,a0
   182c8:	00058693          	mv	a3,a1
   182cc:	000c0513          	mv	a0,s8
   182d0:	000a8593          	mv	a1,s5
   182d4:	6f5040ef          	jal	1d1c8 <__subdf3>
   182d8:	03098793          	addi	a5,s3,48
   182dc:	fefb0fa3          	sb	a5,-1(s6)
   182e0:	00000613          	li	a2,0
   182e4:	00000693          	li	a3,0
   182e8:	00050a93          	mv	s5,a0
   182ec:	00058c13          	mv	s8,a1
   182f0:	6a4040ef          	jal	1c994 <__eqdf2>
   182f4:	42050e63          	beqz	a0,18730 <__gdtoa+0x1268>
   182f8:	07c12703          	lw	a4,124(sp)
   182fc:	000a8513          	mv	a0,s5
   18300:	000c0593          	mv	a1,s8
   18304:	00170813          	addi	a6,a4,1
   18308:	f7971ee3          	bne	a4,s9,18284 <__gdtoa+0xdbc>
   1830c:	02812703          	lw	a4,40(sp)
   18310:	00040f13          	mv	t5,s0
   18314:	000a0413          	mv	s0,s4
   18318:	02070ae3          	beqz	a4,18b4c <__gdtoa+0x1684>
   1831c:	00100793          	li	a5,1
   18320:	01000c13          	li	s8,16
   18324:	26f700e3          	beq	a4,a5,18d84 <__gdtoa+0x18bc>
   18328:	00c12583          	lw	a1,12(sp)
   1832c:	00012503          	lw	a0,0(sp)
   18330:	01e12823          	sw	t5,16(sp)
   18334:	390010ef          	jal	196c4 <_Bfree>
   18338:	00412783          	lw	a5,4(sp)
   1833c:	000b0023          	sb	zero,0(s6)
   18340:	01012f03          	lw	t5,16(sp)
   18344:	0097a023          	sw	s1,0(a5)
   18348:	00812783          	lw	a5,8(sp)
   1834c:	00078463          	beqz	a5,18354 <__gdtoa+0xe8c>
   18350:	0167a023          	sw	s6,0(a5)
   18354:	00092783          	lw	a5,0(s2)
   18358:	0b812403          	lw	s0,184(sp)
   1835c:	0b412483          	lw	s1,180(sp)
   18360:	0187e7b3          	or	a5,a5,s8
   18364:	0ac12983          	lw	s3,172(sp)
   18368:	0a812a03          	lw	s4,168(sp)
   1836c:	0a412a83          	lw	s5,164(sp)
   18370:	09c12b83          	lw	s7,156(sp)
   18374:	09412c83          	lw	s9,148(sp)
   18378:	09012d03          	lw	s10,144(sp)
   1837c:	08c12d83          	lw	s11,140(sp)
   18380:	00f92023          	sw	a5,0(s2)
   18384:	955ff06f          	j	17cd8 <__gdtoa+0x810>
   18388:	e4dc9263          	bne	s9,a3,179cc <__gdtoa+0x504>
   1838c:	004da783          	lw	a5,4(s11)
   18390:	01012703          	lw	a4,16(sp)
   18394:	00178793          	addi	a5,a5,1
   18398:	e2e7da63          	bge	a5,a4,179cc <__gdtoa+0x504>
   1839c:	02412783          	lw	a5,36(sp)
   183a0:	001b8b93          	addi	s7,s7,1
   183a4:	00178793          	addi	a5,a5,1
   183a8:	02f12223          	sw	a5,36(sp)
   183ac:	00100793          	li	a5,1
   183b0:	00f12c23          	sw	a5,24(sp)
   183b4:	e18ff06f          	j	179cc <__gdtoa+0x504>
   183b8:	00100793          	li	a5,1
   183bc:	00000a13          	li	s4,0
   183c0:	02f12a23          	sw	a5,52(sp)
   183c4:	01412583          	lw	a1,20(sp)
   183c8:	00b04463          	bgtz	a1,183d0 <__gdtoa+0xf08>
   183cc:	00100593          	li	a1,1
   183d0:	00058693          	mv	a3,a1
   183d4:	04b12623          	sw	a1,76(sp)
   183d8:	00b12e23          	sw	a1,28(sp)
   183dc:	00b12a23          	sw	a1,20(sp)
   183e0:	c90ff06f          	j	17870 <__gdtoa+0x3a8>
   183e4:	304010ef          	jal	196e8 <__multadd>
   183e8:	00050a13          	mv	s4,a0
   183ec:	94050ae3          	beqz	a0,17d40 <__gdtoa+0x878>
   183f0:	00050a93          	mv	s5,a0
   183f4:	efcff06f          	j	17af0 <__gdtoa+0x628>
   183f8:	e80b5463          	bgez	s6,17a80 <__gdtoa+0x5b8>
   183fc:	02812783          	lw	a5,40(sp)
   18400:	00912623          	sw	s1,12(sp)
   18404:	000c0313          	mv	t1,s8
   18408:	01012f03          	lw	t5,16(sp)
   1840c:	01412903          	lw	s2,20(sp)
   18410:	00070c13          	mv	s8,a4
   18414:	46079ae3          	bnez	a5,19088 <__gdtoa+0x1bc0>
   18418:	00c12783          	lw	a5,12(sp)
   1841c:	00100693          	li	a3,1
   18420:	01000c13          	li	s8,16
   18424:	0107a603          	lw	a2,16(a5)
   18428:	001d8993          	addi	s3,s11,1
   1842c:	3ec6d6e3          	bge	a3,a2,19018 <__gdtoa+0x1b50>
   18430:	000a0413          	mv	s0,s4
   18434:	00098b13          	mv	s6,s3
   18438:	019d8023          	sb	s9,0(s11)
   1843c:	000a8a13          	mv	s4,s5
   18440:	00012983          	lw	s3,0(sp)
   18444:	00030593          	mv	a1,t1
   18448:	01e12823          	sw	t5,16(sp)
   1844c:	00098513          	mv	a0,s3
   18450:	274010ef          	jal	196c4 <_Bfree>
   18454:	03012783          	lw	a5,48(sp)
   18458:	01012f03          	lw	t5,16(sp)
   1845c:	00178493          	addi	s1,a5,1
   18460:	ec0a04e3          	beqz	s4,18328 <__gdtoa+0xe60>
   18464:	00040c63          	beqz	s0,1847c <__gdtoa+0xfb4>
   18468:	01440a63          	beq	s0,s4,1847c <__gdtoa+0xfb4>
   1846c:	00040593          	mv	a1,s0
   18470:	00098513          	mv	a0,s3
   18474:	250010ef          	jal	196c4 <_Bfree>
   18478:	01012f03          	lw	t5,16(sp)
   1847c:	00012503          	lw	a0,0(sp)
   18480:	000a0593          	mv	a1,s4
   18484:	01e12823          	sw	t5,16(sp)
   18488:	23c010ef          	jal	196c4 <_Bfree>
   1848c:	01012f03          	lw	t5,16(sp)
   18490:	e99ff06f          	j	18328 <__gdtoa+0xe60>
   18494:	00c12583          	lw	a1,12(sp)
   18498:	00012503          	lw	a0,0(sp)
   1849c:	03e12023          	sw	t5,32(sp)
   184a0:	00612823          	sw	t1,16(sp)
   184a4:	255010ef          	jal	19ef8 <__lshift>
   184a8:	00a12623          	sw	a0,12(sp)
   184ac:	88050ae3          	beqz	a0,17d40 <__gdtoa+0x878>
   184b0:	07c12783          	lw	a5,124(sp)
   184b4:	02012f03          	lw	t5,32(sp)
   184b8:	01012303          	lw	t1,16(sp)
   184bc:	d34ff06f          	j	179f0 <__gdtoa+0x528>
   184c0:	00c12503          	lw	a0,12(sp)
   184c4:	00030593          	mv	a1,t1
   184c8:	03e12023          	sw	t5,32(sp)
   184cc:	00612823          	sw	t1,16(sp)
   184d0:	3b9010ef          	jal	1a088 <__mcmp>
   184d4:	01012303          	lw	t1,16(sp)
   184d8:	02012f03          	lw	t5,32(sp)
   184dc:	d2055263          	bgez	a0,17a00 <__gdtoa+0x538>
   184e0:	03012783          	lw	a5,48(sp)
   184e4:	00c12583          	lw	a1,12(sp)
   184e8:	00012503          	lw	a0,0(sp)
   184ec:	fff78793          	addi	a5,a5,-1
   184f0:	00000693          	li	a3,0
   184f4:	00a00613          	li	a2,10
   184f8:	01e12e23          	sw	t5,28(sp)
   184fc:	02f12823          	sw	a5,48(sp)
   18500:	1e8010ef          	jal	196e8 <__multadd>
   18504:	00a12623          	sw	a0,12(sp)
   18508:	82050ce3          	beqz	a0,17d40 <__gdtoa+0x878>
   1850c:	03412783          	lw	a5,52(sp)
   18510:	01012303          	lw	t1,16(sp)
   18514:	01c12f03          	lw	t5,28(sp)
   18518:	18079ce3          	bnez	a5,18eb0 <__gdtoa+0x19e8>
   1851c:	04c12783          	lw	a5,76(sp)
   18520:	1ef05e63          	blez	a5,1871c <__gdtoa+0x1254>
   18524:	00f12e23          	sw	a5,28(sp)
   18528:	01c12483          	lw	s1,28(sp)
   1852c:	00c12403          	lw	s0,12(sp)
   18530:	00012a83          	lw	s5,0(sp)
   18534:	000f0d93          	mv	s11,t5
   18538:	00100793          	li	a5,1
   1853c:	00030993          	mv	s3,t1
   18540:	000f0b13          	mv	s6,t5
   18544:	0180006f          	j	1855c <__gdtoa+0x1094>
   18548:	1a0010ef          	jal	196e8 <__multadd>
   1854c:	00050413          	mv	s0,a0
   18550:	fe050863          	beqz	a0,17d40 <__gdtoa+0x878>
   18554:	07c12783          	lw	a5,124(sp)
   18558:	00178793          	addi	a5,a5,1
   1855c:	00098593          	mv	a1,s3
   18560:	00040513          	mv	a0,s0
   18564:	06f12e23          	sw	a5,124(sp)
   18568:	4f9000ef          	jal	19260 <__quorem_D2A>
   1856c:	03050c93          	addi	s9,a0,48
   18570:	019d8023          	sb	s9,0(s11)
   18574:	07c12783          	lw	a5,124(sp)
   18578:	00000693          	li	a3,0
   1857c:	00a00613          	li	a2,10
   18580:	00040593          	mv	a1,s0
   18584:	000a8513          	mv	a0,s5
   18588:	001d8d93          	addi	s11,s11,1
   1858c:	fa97cee3          	blt	a5,s1,18548 <__gdtoa+0x1080>
   18590:	00812623          	sw	s0,12(sp)
   18594:	00098313          	mv	t1,s3
   18598:	000b0f13          	mv	t5,s6
   1859c:	00000413          	li	s0,0
   185a0:	02812703          	lw	a4,40(sp)
   185a4:	4a070063          	beqz	a4,18a44 <__gdtoa+0x157c>
   185a8:	00c12603          	lw	a2,12(sp)
   185ac:	00200793          	li	a5,2
   185b0:	01062683          	lw	a3,16(a2)
   185b4:	4cf70a63          	beq	a4,a5,18a88 <__gdtoa+0x15c0>
   185b8:	00100793          	li	a5,1
   185bc:	28d7ce63          	blt	a5,a3,18858 <__gdtoa+0x1390>
   185c0:	01462783          	lw	a5,20(a2)
   185c4:	28079a63          	bnez	a5,18858 <__gdtoa+0x1390>
   185c8:	00f037b3          	snez	a5,a5
   185cc:	00479c13          	slli	s8,a5,0x4
   185d0:	03000693          	li	a3,48
   185d4:	fffdc783          	lbu	a5,-1(s11)
   185d8:	000d8b13          	mv	s6,s11
   185dc:	fffd8d93          	addi	s11,s11,-1
   185e0:	fed78ae3          	beq	a5,a3,185d4 <__gdtoa+0x110c>
   185e4:	e5dff06f          	j	18440 <__gdtoa+0xf78>
   185e8:	01c12783          	lw	a5,28(sp)
   185ec:	02c12703          	lw	a4,44(sp)
   185f0:	fff78693          	addi	a3,a5,-1
   185f4:	1ad74663          	blt	a4,a3,187a0 <__gdtoa+0x12d8>
   185f8:	40d704b3          	sub	s1,a4,a3
   185fc:	0007dee3          	bgez	a5,18e18 <__gdtoa+0x1950>
   18600:	02412783          	lw	a5,36(sp)
   18604:	01c12703          	lw	a4,28(sp)
   18608:	06012e23          	sw	zero,124(sp)
   1860c:	40e78ab3          	sub	s5,a5,a4
   18610:	b14ff06f          	j	17924 <__gdtoa+0x45c>
   18614:	00200793          	li	a5,2
   18618:	bfa7d863          	bge	a5,s10,17a08 <__gdtoa+0x540>
   1861c:	00012503          	lw	a0,0(sp)
   18620:	00030593          	mv	a1,t1
   18624:	00000693          	li	a3,0
   18628:	00500613          	li	a2,5
   1862c:	01e12823          	sw	t5,16(sp)
   18630:	0b8010ef          	jal	196e8 <__multadd>
   18634:	00050593          	mv	a1,a0
   18638:	f0050463          	beqz	a0,17d40 <__gdtoa+0x878>
   1863c:	01c12783          	lw	a5,28(sp)
   18640:	01012f03          	lw	t5,16(sp)
   18644:	14079463          	bnez	a5,1878c <__gdtoa+0x12c4>
   18648:	00a12823          	sw	a0,16(sp)
   1864c:	00c12503          	lw	a0,12(sp)
   18650:	01e12c23          	sw	t5,24(sp)
   18654:	235010ef          	jal	1a088 <__mcmp>
   18658:	01012583          	lw	a1,16(sp)
   1865c:	01812f03          	lw	t5,24(sp)
   18660:	12a05663          	blez	a0,1878c <__gdtoa+0x12c4>
   18664:	03012783          	lw	a5,48(sp)
   18668:	00278493          	addi	s1,a5,2
   1866c:	03100793          	li	a5,49
   18670:	001f0b13          	addi	s6,t5,1
   18674:	00ff0023          	sb	a5,0(t5)
   18678:	02000c13          	li	s8,32
   1867c:	00012503          	lw	a0,0(sp)
   18680:	01e12823          	sw	t5,16(sp)
   18684:	040010ef          	jal	196c4 <_Bfree>
   18688:	01012f03          	lw	t5,16(sp)
   1868c:	c80a0ee3          	beqz	s4,18328 <__gdtoa+0xe60>
   18690:	dedff06f          	j	1847c <__gdtoa+0xfb4>
   18694:	00100693          	li	a3,1
   18698:	00100593          	li	a1,1
   1869c:	9d4ff06f          	j	17870 <__gdtoa+0x3a8>
   186a0:	00050593          	mv	a1,a0
   186a4:	00012503          	lw	a0,0(sp)
   186a8:	00078613          	mv	a2,a5
   186ac:	6fc010ef          	jal	19da8 <__pow5mult>
   186b0:	00050313          	mv	t1,a0
   186b4:	e8050663          	beqz	a0,17d40 <__gdtoa+0x878>
   186b8:	00100693          	li	a3,1
   186bc:	02012f03          	lw	t5,32(sp)
   186c0:	21a6de63          	bge	a3,s10,188dc <__gdtoa+0x1414>
   186c4:	01032783          	lw	a5,16(t1)
   186c8:	03e12023          	sw	t5,32(sp)
   186cc:	00612823          	sw	t1,16(sp)
   186d0:	00378793          	addi	a5,a5,3
   186d4:	00279793          	slli	a5,a5,0x2
   186d8:	00f307b3          	add	a5,t1,a5
   186dc:	0047a503          	lw	a0,4(a5)
   186e0:	28c010ef          	jal	1996c <__hi0bits>
   186e4:	02012f03          	lw	t5,32(sp)
   186e8:	01012303          	lw	t1,16(sp)
   186ec:	00050b13          	mv	s6,a0
   186f0:	01812c23          	sw	s8,24(sp)
   186f4:	adcff06f          	j	179d0 <__gdtoa+0x508>
   186f8:	00100693          	li	a3,1
   186fc:	efa6c6e3          	blt	a3,s10,185e8 <__gdtoa+0x1120>
   18700:	02412783          	lw	a5,36(sp)
   18704:	02c12483          	lw	s1,44(sp)
   18708:	00cb8bb3          	add	s7,s7,a2
   1870c:	00078a93          	mv	s5,a5
   18710:	00f607b3          	add	a5,a2,a5
   18714:	02f12223          	sw	a5,36(sp)
   18718:	a0cff06f          	j	17924 <__gdtoa+0x45c>
   1871c:	00200793          	li	a5,2
   18720:	13a7c0e3          	blt	a5,s10,19040 <__gdtoa+0x1b78>
   18724:	04c12783          	lw	a5,76(sp)
   18728:	00f12e23          	sw	a5,28(sp)
   1872c:	dfdff06f          	j	18528 <__gdtoa+0x1060>
   18730:	01012c03          	lw	s8,16(sp)
   18734:	00040f13          	mv	t5,s0
   18738:	bf1ff06f          	j	18328 <__gdtoa+0xe60>
   1873c:	01c12783          	lw	a5,28(sp)
   18740:	a8f04ce3          	bgtz	a5,181d8 <__gdtoa+0xd10>
   18744:	04079063          	bnez	a5,18784 <__gdtoa+0x12bc>
   18748:	0000a797          	auipc	a5,0xa
   1874c:	28078793          	addi	a5,a5,640 # 229c8 <__SDATA_BEGIN__+0x50>
   18750:	0007a603          	lw	a2,0(a5)
   18754:	0047a683          	lw	a3,4(a5)
   18758:	000d0513          	mv	a0,s10
   1875c:	000d8593          	mv	a1,s11
   18760:	01e12823          	sw	t5,16(sp)
   18764:	474040ef          	jal	1cbd8 <__muldf3>
   18768:	04812883          	lw	a7,72(sp)
   1876c:	04012783          	lw	a5,64(sp)
   18770:	00088613          	mv	a2,a7
   18774:	00078693          	mv	a3,a5
   18778:	2a8040ef          	jal	1ca20 <__gedf2>
   1877c:	01012f03          	lw	t5,16(sp)
   18780:	6e054c63          	bltz	a0,18e78 <__gdtoa+0x19b0>
   18784:	00000593          	li	a1,0
   18788:	00000a13          	li	s4,0
   1878c:	01412783          	lw	a5,20(sp)
   18790:	000f0b13          	mv	s6,t5
   18794:	01000c13          	li	s8,16
   18798:	40f004b3          	neg	s1,a5
   1879c:	ee1ff06f          	j	1867c <__gdtoa+0x11b4>
   187a0:	02c12783          	lw	a5,44(sp)
   187a4:	02412703          	lw	a4,36(sp)
   187a8:	00000493          	li	s1,0
   187ac:	40f68633          	sub	a2,a3,a5
   187b0:	01812783          	lw	a5,24(sp)
   187b4:	00070a93          	mv	s5,a4
   187b8:	02d12623          	sw	a3,44(sp)
   187bc:	00c787b3          	add	a5,a5,a2
   187c0:	00f12c23          	sw	a5,24(sp)
   187c4:	01c12783          	lw	a5,28(sp)
   187c8:	06f12e23          	sw	a5,124(sp)
   187cc:	00fb8bb3          	add	s7,s7,a5
   187d0:	00f707b3          	add	a5,a4,a5
   187d4:	02f12223          	sw	a5,36(sp)
   187d8:	94cff06f          	j	17924 <__gdtoa+0x45c>
   187dc:	00012b03          	lw	s6,0(sp)
   187e0:	000a0593          	mv	a1,s4
   187e4:	00048613          	mv	a2,s1
   187e8:	000b0513          	mv	a0,s6
   187ec:	03e12023          	sw	t5,32(sp)
   187f0:	5b8010ef          	jal	19da8 <__pow5mult>
   187f4:	00050a13          	mv	s4,a0
   187f8:	d4050463          	beqz	a0,17d40 <__gdtoa+0x878>
   187fc:	00c12603          	lw	a2,12(sp)
   18800:	00050593          	mv	a1,a0
   18804:	000b0513          	mv	a0,s6
   18808:	36c010ef          	jal	19b74 <__multiply>
   1880c:	00050413          	mv	s0,a0
   18810:	d2050863          	beqz	a0,17d40 <__gdtoa+0x878>
   18814:	00c12583          	lw	a1,12(sp)
   18818:	000b0513          	mv	a0,s6
   1881c:	6a9000ef          	jal	196c4 <_Bfree>
   18820:	02c12783          	lw	a5,44(sp)
   18824:	00812623          	sw	s0,12(sp)
   18828:	02012f03          	lw	t5,32(sp)
   1882c:	409787b3          	sub	a5,a5,s1
   18830:	02f12623          	sw	a5,44(sp)
   18834:	96078663          	beqz	a5,179a0 <__gdtoa+0x4d8>
   18838:	948ff06f          	j	17980 <__gdtoa+0x4b8>
   1883c:	000a0413          	mv	s0,s4
   18840:	000d8793          	mv	a5,s11
   18844:	001d8993          	addi	s3,s11,1
   18848:	000a8a13          	mv	s4,s5
   1884c:	03900693          	li	a3,57
   18850:	00098d93          	mv	s11,s3
   18854:	00d78023          	sb	a3,0(a5)
   18858:	03900693          	li	a3,57
   1885c:	0080006f          	j	18864 <__gdtoa+0x139c>
   18860:	25bf0463          	beq	t5,s11,18aa8 <__gdtoa+0x15e0>
   18864:	fffdc783          	lbu	a5,-1(s11)
   18868:	000d8b13          	mv	s6,s11
   1886c:	fffd8d93          	addi	s11,s11,-1
   18870:	fed788e3          	beq	a5,a3,18860 <__gdtoa+0x1398>
   18874:	00178793          	addi	a5,a5,1
   18878:	00fd8023          	sb	a5,0(s11)
   1887c:	02000c13          	li	s8,32
   18880:	bc1ff06f          	j	18440 <__gdtoa+0xf78>
   18884:	00012503          	lw	a0,0(sp)
   18888:	000a0593          	mv	a1,s4
   1888c:	01e12a23          	sw	t5,20(sp)
   18890:	00612823          	sw	t1,16(sp)
   18894:	664010ef          	jal	19ef8 <__lshift>
   18898:	01012303          	lw	t1,16(sp)
   1889c:	01412f03          	lw	t5,20(sp)
   188a0:	00050a13          	mv	s4,a0
   188a4:	96051a63          	bnez	a0,17a18 <__gdtoa+0x550>
   188a8:	c98ff06f          	j	17d40 <__gdtoa+0x878>
   188ac:	00100793          	li	a5,1
   188b0:	00000a13          	li	s4,0
   188b4:	02f12a23          	sw	a5,52(sp)
   188b8:	f95fe06f          	j	1784c <__gdtoa+0x384>
   188bc:	00100793          	li	a5,1
   188c0:	02f12a23          	sw	a5,52(sp)
   188c4:	f89fe06f          	j	1784c <__gdtoa+0x384>
   188c8:	00100793          	li	a5,1
   188cc:	02f12a23          	sw	a5,52(sp)
   188d0:	af5ff06f          	j	183c4 <__gdtoa+0xefc>
   188d4:	02012a23          	sw	zero,52(sp)
   188d8:	aedff06f          	j	183c4 <__gdtoa+0xefc>
   188dc:	dedc94e3          	bne	s9,a3,186c4 <__gdtoa+0x11fc>
   188e0:	004da783          	lw	a5,4(s11)
   188e4:	01012703          	lw	a4,16(sp)
   188e8:	00178793          	addi	a5,a5,1
   188ec:	dce7dce3          	bge	a5,a4,186c4 <__gdtoa+0x11fc>
   188f0:	02412783          	lw	a5,36(sp)
   188f4:	001b8b93          	addi	s7,s7,1
   188f8:	00100c13          	li	s8,1
   188fc:	00178793          	addi	a5,a5,1
   18900:	02f12223          	sw	a5,36(sp)
   18904:	dc1ff06f          	j	186c4 <__gdtoa+0x11fc>
   18908:	00012403          	lw	s0,0(sp)
   1890c:	004a2583          	lw	a1,4(s4)
   18910:	01e12a23          	sw	t5,20(sp)
   18914:	00040513          	mv	a0,s0
   18918:	00612823          	sw	t1,16(sp)
   1891c:	4f5000ef          	jal	19610 <_Balloc>
   18920:	00050a93          	mv	s5,a0
   18924:	c0050e63          	beqz	a0,17d40 <__gdtoa+0x878>
   18928:	010a2603          	lw	a2,16(s4)
   1892c:	00ca0593          	addi	a1,s4,12
   18930:	00c50513          	addi	a0,a0,12
   18934:	00260613          	addi	a2,a2,2
   18938:	00261613          	slli	a2,a2,0x2
   1893c:	a3cfe0ef          	jal	16b78 <memcpy>
   18940:	000a8593          	mv	a1,s5
   18944:	00100613          	li	a2,1
   18948:	00040513          	mv	a0,s0
   1894c:	5ac010ef          	jal	19ef8 <__lshift>
   18950:	01012303          	lw	t1,16(sp)
   18954:	01412f03          	lw	t5,20(sp)
   18958:	00050a93          	mv	s5,a0
   1895c:	00050463          	beqz	a0,18964 <__gdtoa+0x149c>
   18960:	8c8ff06f          	j	17a28 <__gdtoa+0x560>
   18964:	bdcff06f          	j	17d40 <__gdtoa+0x878>
   18968:	000a0693          	mv	a3,s4
   1896c:	00040613          	mv	a2,s0
   18970:	000a0593          	mv	a1,s4
   18974:	00040513          	mv	a0,s0
   18978:	05e12023          	sw	t5,64(sp)
   1897c:	1a4030ef          	jal	1bb20 <__adddf3>
   18980:	0000a697          	auipc	a3,0xa
   18984:	04068693          	addi	a3,a3,64 # 229c0 <__SDATA_BEGIN__+0x48>
   18988:	0006a603          	lw	a2,0(a3)
   1898c:	0046a683          	lw	a3,4(a3)
   18990:	190030ef          	jal	1bb20 <__adddf3>
   18994:	04012f03          	lw	t5,64(sp)
   18998:	fcc00737          	lui	a4,0xfcc00
   1899c:	00050b13          	mv	s6,a0
   189a0:	00b70a33          	add	s4,a4,a1
   189a4:	0000a697          	auipc	a3,0xa
   189a8:	02468693          	addi	a3,a3,36 # 229c8 <__SDATA_BEGIN__+0x50>
   189ac:	0006a603          	lw	a2,0(a3)
   189b0:	02812503          	lw	a0,40(sp)
   189b4:	0046a683          	lw	a3,4(a3)
   189b8:	03812583          	lw	a1,56(sp)
   189bc:	05e12023          	sw	t5,64(sp)
   189c0:	009040ef          	jal	1d1c8 <__subdf3>
   189c4:	000b0613          	mv	a2,s6
   189c8:	000a0693          	mv	a3,s4
   189cc:	00050493          	mv	s1,a0
   189d0:	00058413          	mv	s0,a1
   189d4:	04c040ef          	jal	1ca20 <__gedf2>
   189d8:	04012f03          	lw	t5,64(sp)
   189dc:	44a04c63          	bgtz	a0,18e34 <__gdtoa+0x196c>
   189e0:	800008b7          	lui	a7,0x80000
   189e4:	0148c8b3          	xor	a7,a7,s4
   189e8:	000b0613          	mv	a2,s6
   189ec:	00048513          	mv	a0,s1
   189f0:	00088693          	mv	a3,a7
   189f4:	00040593          	mv	a1,s0
   189f8:	104040ef          	jal	1cafc <__ledf2>
   189fc:	04012f03          	lw	t5,64(sp)
   18a00:	d80542e3          	bltz	a0,18784 <__gdtoa+0x12bc>
   18a04:	02812783          	lw	a5,40(sp)
   18a08:	00008317          	auipc	t1,0x8
   18a0c:	4e830313          	addi	t1,t1,1256 # 20ef0 <__mprec_tens>
   18a10:	04f12423          	sw	a5,72(sp)
   18a14:	03812783          	lw	a5,56(sp)
   18a18:	04f12023          	sw	a5,64(sp)
   18a1c:	02012783          	lw	a5,32(sp)
   18a20:	5207c063          	bltz	a5,18f40 <__gdtoa+0x1a78>
   18a24:	01412783          	lw	a5,20(sp)
   18a28:	02012423          	sw	zero,40(sp)
   18a2c:	00032d03          	lw	s10,0(t1)
   18a30:	00432d83          	lw	s11,4(t1)
   18a34:	fa07d263          	bgez	a5,181d8 <__gdtoa+0xd10>
   18a38:	01c12783          	lw	a5,28(sp)
   18a3c:	f8079e63          	bnez	a5,181d8 <__gdtoa+0xd10>
   18a40:	d09ff06f          	j	18748 <__gdtoa+0x1280>
   18a44:	00c12583          	lw	a1,12(sp)
   18a48:	00012503          	lw	a0,0(sp)
   18a4c:	00100613          	li	a2,1
   18a50:	01e12a23          	sw	t5,20(sp)
   18a54:	00612823          	sw	t1,16(sp)
   18a58:	4a0010ef          	jal	19ef8 <__lshift>
   18a5c:	00a12623          	sw	a0,12(sp)
   18a60:	ae050063          	beqz	a0,17d40 <__gdtoa+0x878>
   18a64:	01012303          	lw	t1,16(sp)
   18a68:	00030593          	mv	a1,t1
   18a6c:	61c010ef          	jal	1a088 <__mcmp>
   18a70:	01012303          	lw	t1,16(sp)
   18a74:	01412f03          	lw	t5,20(sp)
   18a78:	dea040e3          	bgtz	a0,18858 <__gdtoa+0x1390>
   18a7c:	00051663          	bnez	a0,18a88 <__gdtoa+0x15c0>
   18a80:	001cf793          	andi	a5,s9,1
   18a84:	dc079ae3          	bnez	a5,18858 <__gdtoa+0x1390>
   18a88:	00c12783          	lw	a5,12(sp)
   18a8c:	01000c13          	li	s8,16
   18a90:	0107a683          	lw	a3,16(a5)
   18a94:	00100793          	li	a5,1
   18a98:	b2d7cce3          	blt	a5,a3,185d0 <__gdtoa+0x1108>
   18a9c:	00c12783          	lw	a5,12(sp)
   18aa0:	0147a783          	lw	a5,20(a5)
   18aa4:	b25ff06f          	j	185c8 <__gdtoa+0x1100>
   18aa8:	03012783          	lw	a5,48(sp)
   18aac:	02000c13          	li	s8,32
   18ab0:	00178793          	addi	a5,a5,1
   18ab4:	02f12823          	sw	a5,48(sp)
   18ab8:	03100793          	li	a5,49
   18abc:	00ff0023          	sb	a5,0(t5)
   18ac0:	981ff06f          	j	18440 <__gdtoa+0xf78>
   18ac4:	02812783          	lw	a5,40(sp)
   18ac8:	00912623          	sw	s1,12(sp)
   18acc:	01012f03          	lw	t5,16(sp)
   18ad0:	01412903          	lw	s2,20(sp)
   18ad4:	000c0313          	mv	t1,s8
   18ad8:	02078263          	beqz	a5,18afc <__gdtoa+0x1634>
   18adc:	00c12783          	lw	a5,12(sp)
   18ae0:	00100693          	li	a3,1
   18ae4:	0107a603          	lw	a2,16(a5)
   18ae8:	00c6d463          	bge	a3,a2,18af0 <__gdtoa+0x1628>
   18aec:	8c8ff06f          	j	17bb4 <__gdtoa+0x6ec>
   18af0:	0147a683          	lw	a3,20(a5)
   18af4:	00068463          	beqz	a3,18afc <__gdtoa+0x1634>
   18af8:	8bcff06f          	j	17bb4 <__gdtoa+0x6ec>
   18afc:	00c12583          	lw	a1,12(sp)
   18b00:	00012503          	lw	a0,0(sp)
   18b04:	00100613          	li	a2,1
   18b08:	01e12a23          	sw	t5,20(sp)
   18b0c:	00612823          	sw	t1,16(sp)
   18b10:	3e8010ef          	jal	19ef8 <__lshift>
   18b14:	00a12623          	sw	a0,12(sp)
   18b18:	a2050463          	beqz	a0,17d40 <__gdtoa+0x878>
   18b1c:	01012303          	lw	t1,16(sp)
   18b20:	00030593          	mv	a1,t1
   18b24:	564010ef          	jal	1a088 <__mcmp>
   18b28:	01012303          	lw	t1,16(sp)
   18b2c:	01412f03          	lw	t5,20(sp)
   18b30:	4ea05c63          	blez	a0,19028 <__gdtoa+0x1b60>
   18b34:	03900693          	li	a3,57
   18b38:	d0dc82e3          	beq	s9,a3,1883c <__gdtoa+0x1374>
   18b3c:	02000793          	li	a5,32
   18b40:	031b8c93          	addi	s9,s7,49
   18b44:	02f12423          	sw	a5,40(sp)
   18b48:	8d1ff06f          	j	18418 <__gdtoa+0xf50>
   18b4c:	000a8613          	mv	a2,s5
   18b50:	000c0693          	mv	a3,s8
   18b54:	01e12a23          	sw	t5,20(sp)
   18b58:	7c9020ef          	jal	1bb20 <__adddf3>
   18b5c:	fffb4703          	lbu	a4,-1(s6)
   18b60:	000d0613          	mv	a2,s10
   18b64:	000d8693          	mv	a3,s11
   18b68:	00e12823          	sw	a4,16(sp)
   18b6c:	00050a93          	mv	s5,a0
   18b70:	00058a13          	mv	s4,a1
   18b74:	6ad030ef          	jal	1ca20 <__gedf2>
   18b78:	01012703          	lw	a4,16(sp)
   18b7c:	01412f03          	lw	t5,20(sp)
   18b80:	d8a04063          	bgtz	a0,18100 <__gdtoa+0xc38>
   18b84:	000a8513          	mv	a0,s5
   18b88:	000a0593          	mv	a1,s4
   18b8c:	000d0613          	mv	a2,s10
   18b90:	000d8693          	mv	a3,s11
   18b94:	601030ef          	jal	1c994 <__eqdf2>
   18b98:	01412f03          	lw	t5,20(sp)
   18b9c:	2e051863          	bnez	a0,18e8c <__gdtoa+0x19c4>
   18ba0:	0019f993          	andi	s3,s3,1
   18ba4:	2e098463          	beqz	s3,18e8c <__gdtoa+0x19c4>
   18ba8:	01012703          	lw	a4,16(sp)
   18bac:	d54ff06f          	j	18100 <__gdtoa+0xc38>
   18bb0:	03100693          	li	a3,49
   18bb4:	00140413          	addi	s0,s0,1
   18bb8:	d6cff06f          	j	18124 <__gdtoa+0xc5c>
   18bbc:	000b0513          	mv	a0,s6
   18bc0:	000a0593          	mv	a1,s4
   18bc4:	06612623          	sw	t1,108(sp)
   18bc8:	07e12023          	sw	t5,96(sp)
   18bcc:	00c040ef          	jal	1cbd8 <__muldf3>
   18bd0:	04812603          	lw	a2,72(sp)
   18bd4:	04012683          	lw	a3,64(sp)
   18bd8:	06012f03          	lw	t5,96(sp)
   18bdc:	06c12303          	lw	t1,108(sp)
   18be0:	00060a93          	mv	s5,a2
   18be4:	04b12e23          	sw	a1,92(sp)
   18be8:	04c12023          	sw	a2,64(sp)
   18bec:	00100593          	li	a1,1
   18bf0:	00000613          	li	a2,0
   18bf4:	07312223          	sw	s3,100(sp)
   18bf8:	07212423          	sw	s2,104(sp)
   18bfc:	04a12c23          	sw	a0,88(sp)
   18c00:	05012903          	lw	s2,80(sp)
   18c04:	06b12e23          	sw	a1,124(sp)
   18c08:	04d12423          	sw	a3,72(sp)
   18c0c:	00068a13          	mv	s4,a3
   18c10:	000f0b13          	mv	s6,t5
   18c14:	0000a497          	auipc	s1,0xa
   18c18:	d9c48493          	addi	s1,s1,-612 # 229b0 <__SDATA_BEGIN__+0x38>
   18c1c:	00060413          	mv	s0,a2
   18c20:	04612823          	sw	t1,80(sp)
   18c24:	000a8993          	mv	s3,s5
   18c28:	0200006f          	j	18c48 <__gdtoa+0x1780>
   18c2c:	0004a603          	lw	a2,0(s1)
   18c30:	0044a683          	lw	a3,4(s1)
   18c34:	06612e23          	sw	t1,124(sp)
   18c38:	00100413          	li	s0,1
   18c3c:	79d030ef          	jal	1cbd8 <__muldf3>
   18c40:	00050993          	mv	s3,a0
   18c44:	00058a13          	mv	s4,a1
   18c48:	00098513          	mv	a0,s3
   18c4c:	000a0593          	mv	a1,s4
   18c50:	521040ef          	jal	1d970 <__fixdfsi>
   18c54:	00050a93          	mv	s5,a0
   18c58:	02050463          	beqz	a0,18c80 <__gdtoa+0x17b8>
   18c5c:	595040ef          	jal	1d9f0 <__floatsidf>
   18c60:	00050613          	mv	a2,a0
   18c64:	00058693          	mv	a3,a1
   18c68:	00098513          	mv	a0,s3
   18c6c:	000a0593          	mv	a1,s4
   18c70:	558040ef          	jal	1d1c8 <__subdf3>
   18c74:	00050993          	mv	s3,a0
   18c78:	00058a13          	mv	s4,a1
   18c7c:	00100413          	li	s0,1
   18c80:	030a8793          	addi	a5,s5,48
   18c84:	0ff7f713          	zext.b	a4,a5
   18c88:	00eb0023          	sb	a4,0(s6)
   18c8c:	07c12783          	lw	a5,124(sp)
   18c90:	001b0b13          	addi	s6,s6,1
   18c94:	00098513          	mv	a0,s3
   18c98:	000a0593          	mv	a1,s4
   18c9c:	00178313          	addi	t1,a5,1
   18ca0:	f92796e3          	bne	a5,s2,18c2c <__gdtoa+0x1764>
   18ca4:	06012f03          	lw	t5,96(sp)
   18ca8:	06812903          	lw	s2,104(sp)
   18cac:	05012303          	lw	t1,80(sp)
   18cb0:	06412983          	lw	s3,100(sp)
   18cb4:	00040663          	beqz	s0,18cc0 <__gdtoa+0x17f8>
   18cb8:	04a12023          	sw	a0,64(sp)
   18cbc:	05412423          	sw	s4,72(sp)
   18cc0:	0000aa17          	auipc	s4,0xa
   18cc4:	d10a0a13          	addi	s4,s4,-752 # 229d0 <__SDATA_BEGIN__+0x58>
   18cc8:	05c12a83          	lw	s5,92(sp)
   18ccc:	000a2603          	lw	a2,0(s4)
   18cd0:	004a2683          	lw	a3,4(s4)
   18cd4:	05812503          	lw	a0,88(sp)
   18cd8:	000a8593          	mv	a1,s5
   18cdc:	07e12023          	sw	t5,96(sp)
   18ce0:	04e12823          	sw	a4,80(sp)
   18ce4:	06612223          	sw	t1,100(sp)
   18ce8:	639020ef          	jal	1bb20 <__adddf3>
   18cec:	04012483          	lw	s1,64(sp)
   18cf0:	04812403          	lw	s0,72(sp)
   18cf4:	00048613          	mv	a2,s1
   18cf8:	00040693          	mv	a3,s0
   18cfc:	601030ef          	jal	1cafc <__ledf2>
   18d00:	05012703          	lw	a4,80(sp)
   18d04:	06012f03          	lw	t5,96(sp)
   18d08:	22054663          	bltz	a0,18f34 <__gdtoa+0x1a6c>
   18d0c:	05812603          	lw	a2,88(sp)
   18d10:	000a2503          	lw	a0,0(s4)
   18d14:	004a2583          	lw	a1,4(s4)
   18d18:	000a8693          	mv	a3,s5
   18d1c:	05e12823          	sw	t5,80(sp)
   18d20:	4a8040ef          	jal	1d1c8 <__subdf3>
   18d24:	00048613          	mv	a2,s1
   18d28:	00040693          	mv	a3,s0
   18d2c:	4f5030ef          	jal	1ca20 <__gedf2>
   18d30:	05012f03          	lw	t5,80(sp)
   18d34:	06412303          	lw	t1,100(sp)
   18d38:	2aa04663          	bgtz	a0,18fe4 <__gdtoa+0x1b1c>
   18d3c:	02812783          	lw	a5,40(sp)
   18d40:	04f12423          	sw	a5,72(sp)
   18d44:	03812783          	lw	a5,56(sp)
   18d48:	04f12023          	sw	a5,64(sp)
   18d4c:	02012783          	lw	a5,32(sp)
   18d50:	cc07dae3          	bgez	a5,18a24 <__gdtoa+0x155c>
   18d54:	02c12483          	lw	s1,44(sp)
   18d58:	02412a83          	lw	s5,36(sp)
   18d5c:	02012423          	sw	zero,40(sp)
   18d60:	00000a13          	li	s4,0
   18d64:	bddfe06f          	j	17940 <__gdtoa+0x478>
   18d68:	000a0413          	mv	s0,s4
   18d6c:	01012f03          	lw	t5,16(sp)
   18d70:	01412903          	lw	s2,20(sp)
   18d74:	00912623          	sw	s1,12(sp)
   18d78:	000c0313          	mv	t1,s8
   18d7c:	000a8a13          	mv	s4,s5
   18d80:	821ff06f          	j	185a0 <__gdtoa+0x10d8>
   18d84:	fffb4703          	lbu	a4,-1(s6)
   18d88:	b78ff06f          	j	18100 <__gdtoa+0xc38>
   18d8c:	bc048ee3          	beqz	s1,18968 <__gdtoa+0x14a0>
   18d90:	04c12483          	lw	s1,76(sp)
   18d94:	c69058e3          	blez	s1,18a04 <__gdtoa+0x153c>
   18d98:	0000a697          	auipc	a3,0xa
   18d9c:	c1868693          	addi	a3,a3,-1000 # 229b0 <__SDATA_BEGIN__+0x38>
   18da0:	0006a603          	lw	a2,0(a3)
   18da4:	0046a683          	lw	a3,4(a3)
   18da8:	000a0593          	mv	a1,s4
   18dac:	00040513          	mv	a0,s0
   18db0:	07e12023          	sw	t5,96(sp)
   18db4:	625030ef          	jal	1cbd8 <__muldf3>
   18db8:	0000a697          	auipc	a3,0xa
   18dbc:	c0068693          	addi	a3,a3,-1024 # 229b8 <__SDATA_BEGIN__+0x40>
   18dc0:	0006a603          	lw	a2,0(a3)
   18dc4:	0046a683          	lw	a3,4(a3)
   18dc8:	04a12423          	sw	a0,72(sp)
   18dcc:	04a12e23          	sw	a0,92(sp)
   18dd0:	04b12023          	sw	a1,64(sp)
   18dd4:	04b12c23          	sw	a1,88(sp)
   18dd8:	601030ef          	jal	1cbd8 <__muldf3>
   18ddc:	0000a697          	auipc	a3,0xa
   18de0:	be468693          	addi	a3,a3,-1052 # 229c0 <__SDATA_BEGIN__+0x48>
   18de4:	0006a603          	lw	a2,0(a3)
   18de8:	0046a683          	lw	a3,4(a3)
   18dec:	535020ef          	jal	1bb20 <__adddf3>
   18df0:	fcc00737          	lui	a4,0xfcc00
   18df4:	fff00793          	li	a5,-1
   18df8:	05812803          	lw	a6,88(sp)
   18dfc:	05c12e83          	lw	t4,92(sp)
   18e00:	06012f03          	lw	t5,96(sp)
   18e04:	00050b13          	mv	s6,a0
   18e08:	00b70a33          	add	s4,a4,a1
   18e0c:	04912823          	sw	s1,80(sp)
   18e10:	04f12a23          	sw	a5,84(sp)
   18e14:	8f0ff06f          	j	17f04 <__gdtoa+0xa3c>
   18e18:	02412703          	lw	a4,36(sp)
   18e1c:	06f12e23          	sw	a5,124(sp)
   18e20:	00fb8bb3          	add	s7,s7,a5
   18e24:	00f707b3          	add	a5,a4,a5
   18e28:	00070a93          	mv	s5,a4
   18e2c:	02f12223          	sw	a5,36(sp)
   18e30:	af5fe06f          	j	17924 <__gdtoa+0x45c>
   18e34:	00200493          	li	s1,2
   18e38:	00000593          	li	a1,0
   18e3c:	00000a13          	li	s4,0
   18e40:	82dff06f          	j	1866c <__gdtoa+0x11a4>
   18e44:	00912623          	sw	s1,12(sp)
   18e48:	03900693          	li	a3,57
   18e4c:	01012f03          	lw	t5,16(sp)
   18e50:	01412903          	lw	s2,20(sp)
   18e54:	000c0313          	mv	t1,s8
   18e58:	9edc82e3          	beq	s9,a3,1883c <__gdtoa+0x1374>
   18e5c:	001c8793          	addi	a5,s9,1
   18e60:	000a0413          	mv	s0,s4
   18e64:	001d8b13          	addi	s6,s11,1
   18e68:	00fd8023          	sb	a5,0(s11)
   18e6c:	000a8a13          	mv	s4,s5
   18e70:	02000c13          	li	s8,32
   18e74:	dccff06f          	j	18440 <__gdtoa+0xf78>
   18e78:	03012783          	lw	a5,48(sp)
   18e7c:	00000593          	li	a1,0
   18e80:	00000a13          	li	s4,0
   18e84:	00278493          	addi	s1,a5,2
   18e88:	fe4ff06f          	j	1866c <__gdtoa+0x11a4>
   18e8c:	000b0793          	mv	a5,s6
   18e90:	01000c13          	li	s8,16
   18e94:	03000613          	li	a2,48
   18e98:	fff7c703          	lbu	a4,-1(a5)
   18e9c:	00078b13          	mv	s6,a5
   18ea0:	fff78793          	addi	a5,a5,-1
   18ea4:	fec70ae3          	beq	a4,a2,18e98 <__gdtoa+0x19d0>
   18ea8:	00040493          	mv	s1,s0
   18eac:	c7cff06f          	j	18328 <__gdtoa+0xe60>
   18eb0:	00012503          	lw	a0,0(sp)
   18eb4:	000a0593          	mv	a1,s4
   18eb8:	00000693          	li	a3,0
   18ebc:	00a00613          	li	a2,10
   18ec0:	029000ef          	jal	196e8 <__multadd>
   18ec4:	00050a13          	mv	s4,a0
   18ec8:	00051463          	bnez	a0,18ed0 <__gdtoa+0x1a08>
   18ecc:	e75fe06f          	j	17d40 <__gdtoa+0x878>
   18ed0:	04c12783          	lw	a5,76(sp)
   18ed4:	01012303          	lw	t1,16(sp)
   18ed8:	01c12f03          	lw	t5,28(sp)
   18edc:	00f05663          	blez	a5,18ee8 <__gdtoa+0x1a20>
   18ee0:	00f12e23          	sw	a5,28(sp)
   18ee4:	b2dfe06f          	j	17a10 <__gdtoa+0x548>
   18ee8:	00200793          	li	a5,2
   18eec:	15a7ca63          	blt	a5,s10,19040 <__gdtoa+0x1b78>
   18ef0:	04c12783          	lw	a5,76(sp)
   18ef4:	00f12e23          	sw	a5,28(sp)
   18ef8:	b19fe06f          	j	17a10 <__gdtoa+0x548>
   18efc:	04812f03          	lw	t5,72(sp)
   18f00:	05012903          	lw	s2,80(sp)
   18f04:	00048513          	mv	a0,s1
   18f08:	000a0593          	mv	a1,s4
   18f0c:	00000613          	li	a2,0
   18f10:	00000693          	li	a3,0
   18f14:	01e12823          	sw	t5,16(sp)
   18f18:	27d030ef          	jal	1c994 <__eqdf2>
   18f1c:	00a037b3          	snez	a5,a0
   18f20:	00479c13          	slli	s8,a5,0x4
   18f24:	05412783          	lw	a5,84(sp)
   18f28:	01012f03          	lw	t5,16(sp)
   18f2c:	00178493          	addi	s1,a5,1
   18f30:	bf8ff06f          	j	18328 <__gdtoa+0xe60>
   18f34:	05412783          	lw	a5,84(sp)
   18f38:	00178413          	addi	s0,a5,1
   18f3c:	9c4ff06f          	j	18100 <__gdtoa+0xc38>
   18f40:	02012423          	sw	zero,40(sp)
   18f44:	971fe06f          	j	178b4 <__gdtoa+0x3ec>
   18f48:	01c12783          	lw	a5,28(sp)
   18f4c:	02c12703          	lw	a4,44(sp)
   18f50:	fff78693          	addi	a3,a5,-1
   18f54:	84d746e3          	blt	a4,a3,187a0 <__gdtoa+0x12d8>
   18f58:	02412603          	lw	a2,36(sp)
   18f5c:	00fb8bb3          	add	s7,s7,a5
   18f60:	06f12e23          	sw	a5,124(sp)
   18f64:	00f607b3          	add	a5,a2,a5
   18f68:	00060a93          	mv	s5,a2
   18f6c:	40d704b3          	sub	s1,a4,a3
   18f70:	02f12223          	sw	a5,36(sp)
   18f74:	9b1fe06f          	j	17924 <__gdtoa+0x45c>
   18f78:	02812783          	lw	a5,40(sp)
   18f7c:	04812f03          	lw	t5,72(sp)
   18f80:	06012b83          	lw	s7,96(sp)
   18f84:	04f12423          	sw	a5,72(sp)
   18f88:	03812783          	lw	a5,56(sp)
   18f8c:	06412a83          	lw	s5,100(sp)
   18f90:	05012903          	lw	s2,80(sp)
   18f94:	04f12023          	sw	a5,64(sp)
   18f98:	02012783          	lw	a5,32(sp)
   18f9c:	06812303          	lw	t1,104(sp)
   18fa0:	a807d2e3          	bgez	a5,18a24 <__gdtoa+0x155c>
   18fa4:	02012423          	sw	zero,40(sp)
   18fa8:	915fe06f          	j	178bc <__gdtoa+0x3f4>
   18fac:	00912623          	sw	s1,12(sp)
   18fb0:	03900693          	li	a3,57
   18fb4:	01012f03          	lw	t5,16(sp)
   18fb8:	01412903          	lw	s2,20(sp)
   18fbc:	000c0313          	mv	t1,s8
   18fc0:	86dc8ee3          	beq	s9,a3,1883c <__gdtoa+0x1374>
   18fc4:	0f605663          	blez	s6,190b0 <__gdtoa+0x1be8>
   18fc8:	031b8c93          	addi	s9,s7,49
   18fcc:	02000c13          	li	s8,32
   18fd0:	000a0413          	mv	s0,s4
   18fd4:	001d8b13          	addi	s6,s11,1
   18fd8:	019d8023          	sb	s9,0(s11)
   18fdc:	000a8a13          	mv	s4,s5
   18fe0:	c60ff06f          	j	18440 <__gdtoa+0xf78>
   18fe4:	00040593          	mv	a1,s0
   18fe8:	00048513          	mv	a0,s1
   18fec:	00000613          	li	a2,0
   18ff0:	00000693          	li	a3,0
   18ff4:	01e12823          	sw	t5,16(sp)
   18ff8:	19d030ef          	jal	1c994 <__eqdf2>
   18ffc:	05412703          	lw	a4,84(sp)
   19000:	00a037b3          	snez	a5,a0
   19004:	00479c13          	slli	s8,a5,0x4
   19008:	01012f03          	lw	t5,16(sp)
   1900c:	000b0793          	mv	a5,s6
   19010:	00170413          	addi	s0,a4,1 # fcc00001 <__BSS_END__+0xfcbdd201>
   19014:	e81ff06f          	j	18e94 <__gdtoa+0x19cc>
   19018:	0147a683          	lw	a3,20(a5)
   1901c:	c0069a63          	bnez	a3,18430 <__gdtoa+0xf68>
   19020:	02812c03          	lw	s8,40(sp)
   19024:	c0cff06f          	j	18430 <__gdtoa+0xf68>
   19028:	00051663          	bnez	a0,19034 <__gdtoa+0x1b6c>
   1902c:	001cf693          	andi	a3,s9,1
   19030:	b00692e3          	bnez	a3,18b34 <__gdtoa+0x166c>
   19034:	02000793          	li	a5,32
   19038:	02f12423          	sw	a5,40(sp)
   1903c:	bdcff06f          	j	18418 <__gdtoa+0xf50>
   19040:	04c12783          	lw	a5,76(sp)
   19044:	00f12e23          	sw	a5,28(sp)
   19048:	dd4ff06f          	j	1861c <__gdtoa+0x1154>
   1904c:	00912623          	sw	s1,12(sp)
   19050:	03900693          	li	a3,57
   19054:	000b0313          	mv	t1,s6
   19058:	000c0f13          	mv	t5,s8
   1905c:	00dc8e63          	beq	s9,a3,19078 <__gdtoa+0x1bb0>
   19060:	001c8c93          	addi	s9,s9,1
   19064:	02000c13          	li	s8,32
   19068:	bc8ff06f          	j	18430 <__gdtoa+0xf68>
   1906c:	01000c13          	li	s8,16
   19070:	001d8993          	addi	s3,s11,1
   19074:	bbcff06f          	j	18430 <__gdtoa+0xf68>
   19078:	000a0413          	mv	s0,s4
   1907c:	000d8793          	mv	a5,s11
   19080:	000a8a13          	mv	s4,s5
   19084:	fc8ff06f          	j	1884c <__gdtoa+0x1384>
   19088:	0104a603          	lw	a2,16(s1)
   1908c:	00100693          	li	a3,1
   19090:	00048793          	mv	a5,s1
   19094:	00c6d463          	bge	a3,a2,1909c <__gdtoa+0x1bd4>
   19098:	b1dfe06f          	j	17bb4 <__gdtoa+0x6ec>
   1909c:	0147a683          	lw	a3,20(a5)
   190a0:	00068463          	beqz	a3,190a8 <__gdtoa+0x1be0>
   190a4:	b11fe06f          	j	17bb4 <__gdtoa+0x6ec>
   190a8:	001d8993          	addi	s3,s11,1
   190ac:	b84ff06f          	j	18430 <__gdtoa+0xf68>
   190b0:	00c12783          	lw	a5,12(sp)
   190b4:	00100693          	li	a3,1
   190b8:	01000c13          	li	s8,16
   190bc:	0107a603          	lw	a2,16(a5)
   190c0:	f0c6c8e3          	blt	a3,a2,18fd0 <__gdtoa+0x1b08>
   190c4:	0147a683          	lw	a3,20(a5)
   190c8:	00d036b3          	snez	a3,a3
   190cc:	00469c13          	slli	s8,a3,0x4
   190d0:	f01ff06f          	j	18fd0 <__gdtoa+0x1b08>
   190d4:	020b4263          	bltz	s6,190f8 <__gdtoa+0x1c30>
   190d8:	016d6b33          	or	s6,s10,s6
   190dc:	000b1863          	bnez	s6,190ec <__gdtoa+0x1c24>
   190e0:	0009a783          	lw	a5,0(s3)
   190e4:	0017f793          	andi	a5,a5,1
   190e8:	00078863          	beqz	a5,190f8 <__gdtoa+0x1c30>
   190ec:	00d04463          	bgtz	a3,190f4 <__gdtoa+0x1c2c>
   190f0:	991fe06f          	j	17a80 <__gdtoa+0x5b8>
   190f4:	981fe06f          	j	17a74 <__gdtoa+0x5ac>
   190f8:	02812703          	lw	a4,40(sp)
   190fc:	00912623          	sw	s1,12(sp)
   19100:	01012f03          	lw	t5,16(sp)
   19104:	01412903          	lw	s2,20(sp)
   19108:	000c0313          	mv	t1,s8
   1910c:	02070863          	beqz	a4,1913c <__gdtoa+0x1c74>
   19110:	0104a583          	lw	a1,16(s1)
   19114:	00100613          	li	a2,1
   19118:	00b65463          	bge	a2,a1,19120 <__gdtoa+0x1c58>
   1911c:	a99fe06f          	j	17bb4 <__gdtoa+0x6ec>
   19120:	0144a603          	lw	a2,20(s1)
   19124:	00060463          	beqz	a2,1912c <__gdtoa+0x1c64>
   19128:	a8dfe06f          	j	17bb4 <__gdtoa+0x6ec>
   1912c:	9cd048e3          	bgtz	a3,18afc <__gdtoa+0x1634>
   19130:	00000c13          	li	s8,0
   19134:	001d8993          	addi	s3,s11,1
   19138:	af8ff06f          	j	18430 <__gdtoa+0xf68>
   1913c:	acd05e63          	blez	a3,18418 <__gdtoa+0xf50>
   19140:	9bdff06f          	j	18afc <__gdtoa+0x1634>

00019144 <__rv_alloc_D2A>:
   19144:	ff010113          	addi	sp,sp,-16
   19148:	00812423          	sw	s0,8(sp)
   1914c:	00112623          	sw	ra,12(sp)
   19150:	01300793          	li	a5,19
   19154:	00000413          	li	s0,0
   19158:	00b7fc63          	bgeu	a5,a1,19170 <__rv_alloc_D2A+0x2c>
   1915c:	00400793          	li	a5,4
   19160:	00179793          	slli	a5,a5,0x1
   19164:	01078713          	addi	a4,a5,16
   19168:	00140413          	addi	s0,s0,1
   1916c:	fee5fae3          	bgeu	a1,a4,19160 <__rv_alloc_D2A+0x1c>
   19170:	00040593          	mv	a1,s0
   19174:	49c000ef          	jal	19610 <_Balloc>
   19178:	00050663          	beqz	a0,19184 <__rv_alloc_D2A+0x40>
   1917c:	00852023          	sw	s0,0(a0)
   19180:	00450513          	addi	a0,a0,4
   19184:	00c12083          	lw	ra,12(sp)
   19188:	00812403          	lw	s0,8(sp)
   1918c:	01010113          	addi	sp,sp,16
   19190:	00008067          	ret

00019194 <__nrv_alloc_D2A>:
   19194:	ff010113          	addi	sp,sp,-16
   19198:	00812423          	sw	s0,8(sp)
   1919c:	01212023          	sw	s2,0(sp)
   191a0:	00112623          	sw	ra,12(sp)
   191a4:	00912223          	sw	s1,4(sp)
   191a8:	01300793          	li	a5,19
   191ac:	00058413          	mv	s0,a1
   191b0:	00060913          	mv	s2,a2
   191b4:	06d7fe63          	bgeu	a5,a3,19230 <__nrv_alloc_D2A+0x9c>
   191b8:	00400793          	li	a5,4
   191bc:	00000493          	li	s1,0
   191c0:	00179793          	slli	a5,a5,0x1
   191c4:	01078713          	addi	a4,a5,16
   191c8:	00148493          	addi	s1,s1,1
   191cc:	fee6fae3          	bgeu	a3,a4,191c0 <__nrv_alloc_D2A+0x2c>
   191d0:	00048593          	mv	a1,s1
   191d4:	43c000ef          	jal	19610 <_Balloc>
   191d8:	00050793          	mv	a5,a0
   191dc:	04050e63          	beqz	a0,19238 <__nrv_alloc_D2A+0xa4>
   191e0:	00952023          	sw	s1,0(a0)
   191e4:	00044703          	lbu	a4,0(s0)
   191e8:	00450513          	addi	a0,a0,4
   191ec:	00140593          	addi	a1,s0,1
   191f0:	00e78223          	sb	a4,4(a5)
   191f4:	00050793          	mv	a5,a0
   191f8:	00070c63          	beqz	a4,19210 <__nrv_alloc_D2A+0x7c>
   191fc:	0005c703          	lbu	a4,0(a1)
   19200:	00178793          	addi	a5,a5,1
   19204:	00158593          	addi	a1,a1,1
   19208:	00e78023          	sb	a4,0(a5)
   1920c:	fe0718e3          	bnez	a4,191fc <__nrv_alloc_D2A+0x68>
   19210:	00090463          	beqz	s2,19218 <__nrv_alloc_D2A+0x84>
   19214:	00f92023          	sw	a5,0(s2)
   19218:	00c12083          	lw	ra,12(sp)
   1921c:	00812403          	lw	s0,8(sp)
   19220:	00412483          	lw	s1,4(sp)
   19224:	00012903          	lw	s2,0(sp)
   19228:	01010113          	addi	sp,sp,16
   1922c:	00008067          	ret
   19230:	00000493          	li	s1,0
   19234:	f9dff06f          	j	191d0 <__nrv_alloc_D2A+0x3c>
   19238:	00000513          	li	a0,0
   1923c:	fddff06f          	j	19218 <__nrv_alloc_D2A+0x84>

00019240 <__freedtoa>:
   19240:	ffc5a683          	lw	a3,-4(a1)
   19244:	00100713          	li	a4,1
   19248:	00058793          	mv	a5,a1
   1924c:	00d71733          	sll	a4,a4,a3
   19250:	ffc58593          	addi	a1,a1,-4
   19254:	00d7a023          	sw	a3,0(a5)
   19258:	00e7a223          	sw	a4,4(a5)
   1925c:	4680006f          	j	196c4 <_Bfree>

00019260 <__quorem_D2A>:
   19260:	fe010113          	addi	sp,sp,-32
   19264:	00912a23          	sw	s1,20(sp)
   19268:	01052783          	lw	a5,16(a0)
   1926c:	0105a483          	lw	s1,16(a1)
   19270:	00112e23          	sw	ra,28(sp)
   19274:	1c97c863          	blt	a5,s1,19444 <__quorem_D2A+0x1e4>
   19278:	fff48493          	addi	s1,s1,-1
   1927c:	00249313          	slli	t1,s1,0x2
   19280:	00812c23          	sw	s0,24(sp)
   19284:	01458413          	addi	s0,a1,20
   19288:	01312623          	sw	s3,12(sp)
   1928c:	01412423          	sw	s4,8(sp)
   19290:	006409b3          	add	s3,s0,t1
   19294:	01450a13          	addi	s4,a0,20
   19298:	0009a783          	lw	a5,0(s3)
   1929c:	006a0333          	add	t1,s4,t1
   192a0:	00032703          	lw	a4,0(t1)
   192a4:	01212823          	sw	s2,16(sp)
   192a8:	01512223          	sw	s5,4(sp)
   192ac:	00178793          	addi	a5,a5,1
   192b0:	02f75933          	divu	s2,a4,a5
   192b4:	00050a93          	mv	s5,a0
   192b8:	0af76e63          	bltu	a4,a5,19374 <__quorem_D2A+0x114>
   192bc:	00010537          	lui	a0,0x10
   192c0:	00040893          	mv	a7,s0
   192c4:	000a0813          	mv	a6,s4
   192c8:	00000f13          	li	t5,0
   192cc:	00000e93          	li	t4,0
   192d0:	fff50513          	addi	a0,a0,-1 # ffff <exit-0xb5>
   192d4:	0008a783          	lw	a5,0(a7) # 80000000 <__BSS_END__+0x7ffdd200>
   192d8:	00082603          	lw	a2,0(a6)
   192dc:	00480813          	addi	a6,a6,4
   192e0:	00a7f6b3          	and	a3,a5,a0
   192e4:	0107d793          	srli	a5,a5,0x10
   192e8:	00a67733          	and	a4,a2,a0
   192ec:	01065e13          	srli	t3,a2,0x10
   192f0:	00488893          	addi	a7,a7,4
   192f4:	032686b3          	mul	a3,a3,s2
   192f8:	032787b3          	mul	a5,a5,s2
   192fc:	01e686b3          	add	a3,a3,t5
   19300:	00a6f633          	and	a2,a3,a0
   19304:	40c70733          	sub	a4,a4,a2
   19308:	41d70733          	sub	a4,a4,t4
   1930c:	0106d693          	srli	a3,a3,0x10
   19310:	01075613          	srli	a2,a4,0x10
   19314:	00167613          	andi	a2,a2,1
   19318:	00a77733          	and	a4,a4,a0
   1931c:	00d787b3          	add	a5,a5,a3
   19320:	00a7f6b3          	and	a3,a5,a0
   19324:	00d60633          	add	a2,a2,a3
   19328:	40ce06b3          	sub	a3,t3,a2
   1932c:	01069613          	slli	a2,a3,0x10
   19330:	00e66733          	or	a4,a2,a4
   19334:	0106d693          	srli	a3,a3,0x10
   19338:	fee82e23          	sw	a4,-4(a6)
   1933c:	0107df13          	srli	t5,a5,0x10
   19340:	0016fe93          	andi	t4,a3,1
   19344:	f919f8e3          	bgeu	s3,a7,192d4 <__quorem_D2A+0x74>
   19348:	00032783          	lw	a5,0(t1)
   1934c:	02079463          	bnez	a5,19374 <__quorem_D2A+0x114>
   19350:	ffc30313          	addi	t1,t1,-4
   19354:	006a6863          	bltu	s4,t1,19364 <__quorem_D2A+0x104>
   19358:	0180006f          	j	19370 <__quorem_D2A+0x110>
   1935c:	fff48493          	addi	s1,s1,-1
   19360:	006a7863          	bgeu	s4,t1,19370 <__quorem_D2A+0x110>
   19364:	00032783          	lw	a5,0(t1)
   19368:	ffc30313          	addi	t1,t1,-4
   1936c:	fe0788e3          	beqz	a5,1935c <__quorem_D2A+0xfc>
   19370:	009aa823          	sw	s1,16(s5)
   19374:	000a8513          	mv	a0,s5
   19378:	511000ef          	jal	1a088 <__mcmp>
   1937c:	0a054063          	bltz	a0,1941c <__quorem_D2A+0x1bc>
   19380:	00010537          	lui	a0,0x10
   19384:	000a0593          	mv	a1,s4
   19388:	00000693          	li	a3,0
   1938c:	fff50513          	addi	a0,a0,-1 # ffff <exit-0xb5>
   19390:	0005a783          	lw	a5,0(a1)
   19394:	00042603          	lw	a2,0(s0)
   19398:	00458593          	addi	a1,a1,4
   1939c:	00a7f733          	and	a4,a5,a0
   193a0:	00a67833          	and	a6,a2,a0
   193a4:	41070733          	sub	a4,a4,a6
   193a8:	40d70733          	sub	a4,a4,a3
   193ac:	01075693          	srli	a3,a4,0x10
   193b0:	0016f693          	andi	a3,a3,1
   193b4:	01065613          	srli	a2,a2,0x10
   193b8:	00c686b3          	add	a3,a3,a2
   193bc:	0107d793          	srli	a5,a5,0x10
   193c0:	40d787b3          	sub	a5,a5,a3
   193c4:	01079693          	slli	a3,a5,0x10
   193c8:	00a77733          	and	a4,a4,a0
   193cc:	00e6e733          	or	a4,a3,a4
   193d0:	0107d793          	srli	a5,a5,0x10
   193d4:	00440413          	addi	s0,s0,4
   193d8:	fee5ae23          	sw	a4,-4(a1)
   193dc:	0017f693          	andi	a3,a5,1
   193e0:	fa89f8e3          	bgeu	s3,s0,19390 <__quorem_D2A+0x130>
   193e4:	00249793          	slli	a5,s1,0x2
   193e8:	00fa07b3          	add	a5,s4,a5
   193ec:	0007a703          	lw	a4,0(a5)
   193f0:	02071463          	bnez	a4,19418 <__quorem_D2A+0x1b8>
   193f4:	ffc78793          	addi	a5,a5,-4
   193f8:	00fa6863          	bltu	s4,a5,19408 <__quorem_D2A+0x1a8>
   193fc:	0180006f          	j	19414 <__quorem_D2A+0x1b4>
   19400:	fff48493          	addi	s1,s1,-1
   19404:	00fa7863          	bgeu	s4,a5,19414 <__quorem_D2A+0x1b4>
   19408:	0007a703          	lw	a4,0(a5)
   1940c:	ffc78793          	addi	a5,a5,-4
   19410:	fe0708e3          	beqz	a4,19400 <__quorem_D2A+0x1a0>
   19414:	009aa823          	sw	s1,16(s5)
   19418:	00190913          	addi	s2,s2,1
   1941c:	01812403          	lw	s0,24(sp)
   19420:	01c12083          	lw	ra,28(sp)
   19424:	00c12983          	lw	s3,12(sp)
   19428:	00812a03          	lw	s4,8(sp)
   1942c:	00412a83          	lw	s5,4(sp)
   19430:	01412483          	lw	s1,20(sp)
   19434:	00090513          	mv	a0,s2
   19438:	01012903          	lw	s2,16(sp)
   1943c:	02010113          	addi	sp,sp,32
   19440:	00008067          	ret
   19444:	01c12083          	lw	ra,28(sp)
   19448:	01412483          	lw	s1,20(sp)
   1944c:	00000513          	li	a0,0
   19450:	02010113          	addi	sp,sp,32
   19454:	00008067          	ret

00019458 <__rshift_D2A>:
   19458:	01052803          	lw	a6,16(a0)
   1945c:	4055de13          	srai	t3,a1,0x5
   19460:	010e4863          	blt	t3,a6,19470 <__rshift_D2A+0x18>
   19464:	00052823          	sw	zero,16(a0)
   19468:	00052a23          	sw	zero,20(a0)
   1946c:	00008067          	ret
   19470:	01450313          	addi	t1,a0,20
   19474:	00281613          	slli	a2,a6,0x2
   19478:	002e1793          	slli	a5,t3,0x2
   1947c:	01f5f593          	andi	a1,a1,31
   19480:	00c30633          	add	a2,t1,a2
   19484:	00f307b3          	add	a5,t1,a5
   19488:	06058263          	beqz	a1,194ec <__rshift_D2A+0x94>
   1948c:	0007a683          	lw	a3,0(a5)
   19490:	02000e93          	li	t4,32
   19494:	00478793          	addi	a5,a5,4
   19498:	40be8eb3          	sub	t4,t4,a1
   1949c:	00b6d6b3          	srl	a3,a3,a1
   194a0:	08c7f463          	bgeu	a5,a2,19528 <__rshift_D2A+0xd0>
   194a4:	00030893          	mv	a7,t1
   194a8:	0007a703          	lw	a4,0(a5)
   194ac:	00488893          	addi	a7,a7,4
   194b0:	00478793          	addi	a5,a5,4
   194b4:	01d71733          	sll	a4,a4,t4
   194b8:	00d76733          	or	a4,a4,a3
   194bc:	fee8ae23          	sw	a4,-4(a7)
   194c0:	ffc7a683          	lw	a3,-4(a5)
   194c4:	00b6d6b3          	srl	a3,a3,a1
   194c8:	fec7e0e3          	bltu	a5,a2,194a8 <__rshift_D2A+0x50>
   194cc:	41c80833          	sub	a6,a6,t3
   194d0:	00281813          	slli	a6,a6,0x2
   194d4:	ffc80813          	addi	a6,a6,-4
   194d8:	01030833          	add	a6,t1,a6
   194dc:	00d82023          	sw	a3,0(a6)
   194e0:	02068a63          	beqz	a3,19514 <__rshift_D2A+0xbc>
   194e4:	00480813          	addi	a6,a6,4
   194e8:	02c0006f          	j	19514 <__rshift_D2A+0xbc>
   194ec:	00030713          	mv	a4,t1
   194f0:	f6c7fae3          	bgeu	a5,a2,19464 <__rshift_D2A+0xc>
   194f4:	0007a683          	lw	a3,0(a5)
   194f8:	00478793          	addi	a5,a5,4
   194fc:	00470713          	addi	a4,a4,4
   19500:	fed72e23          	sw	a3,-4(a4)
   19504:	fec7e8e3          	bltu	a5,a2,194f4 <__rshift_D2A+0x9c>
   19508:	41c80833          	sub	a6,a6,t3
   1950c:	00281813          	slli	a6,a6,0x2
   19510:	01030833          	add	a6,t1,a6
   19514:	406807b3          	sub	a5,a6,t1
   19518:	4027d793          	srai	a5,a5,0x2
   1951c:	00f52823          	sw	a5,16(a0)
   19520:	f46804e3          	beq	a6,t1,19468 <__rshift_D2A+0x10>
   19524:	00008067          	ret
   19528:	00d52a23          	sw	a3,20(a0)
   1952c:	f2068ce3          	beqz	a3,19464 <__rshift_D2A+0xc>
   19530:	00030813          	mv	a6,t1
   19534:	00480813          	addi	a6,a6,4
   19538:	fddff06f          	j	19514 <__rshift_D2A+0xbc>

0001953c <__trailz_D2A>:
   1953c:	01052703          	lw	a4,16(a0)
   19540:	fe010113          	addi	sp,sp,-32
   19544:	01450513          	addi	a0,a0,20
   19548:	00271713          	slli	a4,a4,0x2
   1954c:	00812c23          	sw	s0,24(sp)
   19550:	00112e23          	sw	ra,28(sp)
   19554:	00e50733          	add	a4,a0,a4
   19558:	00000413          	li	s0,0
   1955c:	00e56a63          	bltu	a0,a4,19570 <__trailz_D2A+0x34>
   19560:	02c0006f          	j	1958c <__trailz_D2A+0x50>
   19564:	00450513          	addi	a0,a0,4
   19568:	02040413          	addi	s0,s0,32
   1956c:	02e57063          	bgeu	a0,a4,1958c <__trailz_D2A+0x50>
   19570:	00052783          	lw	a5,0(a0)
   19574:	fe0788e3          	beqz	a5,19564 <__trailz_D2A+0x28>
   19578:	00e57a63          	bgeu	a0,a4,1958c <__trailz_D2A+0x50>
   1957c:	00c10513          	addi	a0,sp,12
   19580:	00f12623          	sw	a5,12(sp)
   19584:	460000ef          	jal	199e4 <__lo0bits>
   19588:	00a40433          	add	s0,s0,a0
   1958c:	01c12083          	lw	ra,28(sp)
   19590:	00040513          	mv	a0,s0
   19594:	01812403          	lw	s0,24(sp)
   19598:	02010113          	addi	sp,sp,32
   1959c:	00008067          	ret

000195a0 <_mbtowc_r>:
   195a0:	e6c1a783          	lw	a5,-404(gp) # 228ec <__global_locale+0xe4>
   195a4:	00078067          	jr	a5

000195a8 <__ascii_mbtowc>:
   195a8:	02058063          	beqz	a1,195c8 <__ascii_mbtowc+0x20>
   195ac:	04060263          	beqz	a2,195f0 <__ascii_mbtowc+0x48>
   195b0:	04068863          	beqz	a3,19600 <__ascii_mbtowc+0x58>
   195b4:	00064783          	lbu	a5,0(a2)
   195b8:	00f5a023          	sw	a5,0(a1)
   195bc:	00064503          	lbu	a0,0(a2)
   195c0:	00a03533          	snez	a0,a0
   195c4:	00008067          	ret
   195c8:	ff010113          	addi	sp,sp,-16
   195cc:	00c10593          	addi	a1,sp,12
   195d0:	02060463          	beqz	a2,195f8 <__ascii_mbtowc+0x50>
   195d4:	02068a63          	beqz	a3,19608 <__ascii_mbtowc+0x60>
   195d8:	00064783          	lbu	a5,0(a2)
   195dc:	00f5a023          	sw	a5,0(a1)
   195e0:	00064503          	lbu	a0,0(a2)
   195e4:	00a03533          	snez	a0,a0
   195e8:	01010113          	addi	sp,sp,16
   195ec:	00008067          	ret
   195f0:	00000513          	li	a0,0
   195f4:	00008067          	ret
   195f8:	00000513          	li	a0,0
   195fc:	fedff06f          	j	195e8 <__ascii_mbtowc+0x40>
   19600:	ffe00513          	li	a0,-2
   19604:	00008067          	ret
   19608:	ffe00513          	li	a0,-2
   1960c:	fddff06f          	j	195e8 <__ascii_mbtowc+0x40>

00019610 <_Balloc>:
   19610:	04452783          	lw	a5,68(a0)
   19614:	ff010113          	addi	sp,sp,-16
   19618:	00812423          	sw	s0,8(sp)
   1961c:	00912223          	sw	s1,4(sp)
   19620:	00112623          	sw	ra,12(sp)
   19624:	00050413          	mv	s0,a0
   19628:	00058493          	mv	s1,a1
   1962c:	02078c63          	beqz	a5,19664 <_Balloc+0x54>
   19630:	00249713          	slli	a4,s1,0x2
   19634:	00e787b3          	add	a5,a5,a4
   19638:	0007a503          	lw	a0,0(a5)
   1963c:	04050463          	beqz	a0,19684 <_Balloc+0x74>
   19640:	00052703          	lw	a4,0(a0)
   19644:	00e7a023          	sw	a4,0(a5)
   19648:	00052823          	sw	zero,16(a0)
   1964c:	00052623          	sw	zero,12(a0)
   19650:	00c12083          	lw	ra,12(sp)
   19654:	00812403          	lw	s0,8(sp)
   19658:	00412483          	lw	s1,4(sp)
   1965c:	01010113          	addi	sp,sp,16
   19660:	00008067          	ret
   19664:	02100613          	li	a2,33
   19668:	00400593          	li	a1,4
   1966c:	591010ef          	jal	1b3fc <_calloc_r>
   19670:	04a42223          	sw	a0,68(s0)
   19674:	00050793          	mv	a5,a0
   19678:	fa051ce3          	bnez	a0,19630 <_Balloc+0x20>
   1967c:	00000513          	li	a0,0
   19680:	fd1ff06f          	j	19650 <_Balloc+0x40>
   19684:	01212023          	sw	s2,0(sp)
   19688:	00100913          	li	s2,1
   1968c:	00991933          	sll	s2,s2,s1
   19690:	00590613          	addi	a2,s2,5
   19694:	00261613          	slli	a2,a2,0x2
   19698:	00100593          	li	a1,1
   1969c:	00040513          	mv	a0,s0
   196a0:	55d010ef          	jal	1b3fc <_calloc_r>
   196a4:	00050a63          	beqz	a0,196b8 <_Balloc+0xa8>
   196a8:	01252423          	sw	s2,8(a0)
   196ac:	00952223          	sw	s1,4(a0)
   196b0:	00012903          	lw	s2,0(sp)
   196b4:	f95ff06f          	j	19648 <_Balloc+0x38>
   196b8:	00012903          	lw	s2,0(sp)
   196bc:	00000513          	li	a0,0
   196c0:	f91ff06f          	j	19650 <_Balloc+0x40>

000196c4 <_Bfree>:
   196c4:	02058063          	beqz	a1,196e4 <_Bfree+0x20>
   196c8:	0045a703          	lw	a4,4(a1)
   196cc:	04452783          	lw	a5,68(a0)
   196d0:	00271713          	slli	a4,a4,0x2
   196d4:	00e787b3          	add	a5,a5,a4
   196d8:	0007a703          	lw	a4,0(a5)
   196dc:	00e5a023          	sw	a4,0(a1)
   196e0:	00b7a023          	sw	a1,0(a5)
   196e4:	00008067          	ret

000196e8 <__multadd>:
   196e8:	fe010113          	addi	sp,sp,-32
   196ec:	00912a23          	sw	s1,20(sp)
   196f0:	0105a483          	lw	s1,16(a1)
   196f4:	00010337          	lui	t1,0x10
   196f8:	00812c23          	sw	s0,24(sp)
   196fc:	01212823          	sw	s2,16(sp)
   19700:	01312623          	sw	s3,12(sp)
   19704:	00112e23          	sw	ra,28(sp)
   19708:	00058913          	mv	s2,a1
   1970c:	00050993          	mv	s3,a0
   19710:	00068413          	mv	s0,a3
   19714:	01458813          	addi	a6,a1,20
   19718:	00000893          	li	a7,0
   1971c:	fff30313          	addi	t1,t1,-1 # ffff <exit-0xb5>
   19720:	00082783          	lw	a5,0(a6)
   19724:	00480813          	addi	a6,a6,4
   19728:	00188893          	addi	a7,a7,1
   1972c:	0067f733          	and	a4,a5,t1
   19730:	02c70733          	mul	a4,a4,a2
   19734:	0107d693          	srli	a3,a5,0x10
   19738:	02c686b3          	mul	a3,a3,a2
   1973c:	008707b3          	add	a5,a4,s0
   19740:	0107d713          	srli	a4,a5,0x10
   19744:	0067f7b3          	and	a5,a5,t1
   19748:	00e686b3          	add	a3,a3,a4
   1974c:	01069713          	slli	a4,a3,0x10
   19750:	00f707b3          	add	a5,a4,a5
   19754:	fef82e23          	sw	a5,-4(a6)
   19758:	0106d413          	srli	s0,a3,0x10
   1975c:	fc98c2e3          	blt	a7,s1,19720 <__multadd+0x38>
   19760:	02040263          	beqz	s0,19784 <__multadd+0x9c>
   19764:	00892783          	lw	a5,8(s2)
   19768:	02f4de63          	bge	s1,a5,197a4 <__multadd+0xbc>
   1976c:	00448793          	addi	a5,s1,4
   19770:	00279793          	slli	a5,a5,0x2
   19774:	00f907b3          	add	a5,s2,a5
   19778:	0087a223          	sw	s0,4(a5)
   1977c:	00148493          	addi	s1,s1,1
   19780:	00992823          	sw	s1,16(s2)
   19784:	01c12083          	lw	ra,28(sp)
   19788:	01812403          	lw	s0,24(sp)
   1978c:	01412483          	lw	s1,20(sp)
   19790:	00c12983          	lw	s3,12(sp)
   19794:	00090513          	mv	a0,s2
   19798:	01012903          	lw	s2,16(sp)
   1979c:	02010113          	addi	sp,sp,32
   197a0:	00008067          	ret
   197a4:	00492583          	lw	a1,4(s2)
   197a8:	00098513          	mv	a0,s3
   197ac:	01412423          	sw	s4,8(sp)
   197b0:	00158593          	addi	a1,a1,1
   197b4:	e5dff0ef          	jal	19610 <_Balloc>
   197b8:	00050a13          	mv	s4,a0
   197bc:	04050e63          	beqz	a0,19818 <__multadd+0x130>
   197c0:	01092603          	lw	a2,16(s2)
   197c4:	00c90593          	addi	a1,s2,12
   197c8:	00c50513          	addi	a0,a0,12
   197cc:	00260613          	addi	a2,a2,2
   197d0:	00261613          	slli	a2,a2,0x2
   197d4:	ba4fd0ef          	jal	16b78 <memcpy>
   197d8:	00492703          	lw	a4,4(s2)
   197dc:	0449a783          	lw	a5,68(s3)
   197e0:	00271713          	slli	a4,a4,0x2
   197e4:	00e787b3          	add	a5,a5,a4
   197e8:	0007a703          	lw	a4,0(a5)
   197ec:	00e92023          	sw	a4,0(s2)
   197f0:	0127a023          	sw	s2,0(a5)
   197f4:	00448793          	addi	a5,s1,4
   197f8:	000a0913          	mv	s2,s4
   197fc:	00279793          	slli	a5,a5,0x2
   19800:	00f907b3          	add	a5,s2,a5
   19804:	00812a03          	lw	s4,8(sp)
   19808:	00148493          	addi	s1,s1,1
   1980c:	0087a223          	sw	s0,4(a5)
   19810:	00992823          	sw	s1,16(s2)
   19814:	f71ff06f          	j	19784 <__multadd+0x9c>
   19818:	00007697          	auipc	a3,0x7
   1981c:	16868693          	addi	a3,a3,360 # 20980 <_exit+0x148>
   19820:	00000613          	li	a2,0
   19824:	0ba00593          	li	a1,186
   19828:	00007517          	auipc	a0,0x7
   1982c:	16c50513          	addi	a0,a0,364 # 20994 <_exit+0x15c>
   19830:	365010ef          	jal	1b394 <__assert_func>

00019834 <__s2b>:
   19834:	fe010113          	addi	sp,sp,-32
   19838:	00812c23          	sw	s0,24(sp)
   1983c:	00912a23          	sw	s1,20(sp)
   19840:	01212823          	sw	s2,16(sp)
   19844:	01312623          	sw	s3,12(sp)
   19848:	01412423          	sw	s4,8(sp)
   1984c:	00068993          	mv	s3,a3
   19850:	00900793          	li	a5,9
   19854:	00868693          	addi	a3,a3,8
   19858:	00112e23          	sw	ra,28(sp)
   1985c:	02f6c6b3          	div	a3,a3,a5
   19860:	00050913          	mv	s2,a0
   19864:	00058413          	mv	s0,a1
   19868:	00060a13          	mv	s4,a2
   1986c:	00070493          	mv	s1,a4
   19870:	0d37da63          	bge	a5,s3,19944 <__s2b+0x110>
   19874:	00100793          	li	a5,1
   19878:	00000593          	li	a1,0
   1987c:	00179793          	slli	a5,a5,0x1
   19880:	00158593          	addi	a1,a1,1
   19884:	fed7cce3          	blt	a5,a3,1987c <__s2b+0x48>
   19888:	00090513          	mv	a0,s2
   1988c:	d85ff0ef          	jal	19610 <_Balloc>
   19890:	00050593          	mv	a1,a0
   19894:	0a050c63          	beqz	a0,1994c <__s2b+0x118>
   19898:	00100793          	li	a5,1
   1989c:	00f52823          	sw	a5,16(a0)
   198a0:	00952a23          	sw	s1,20(a0)
   198a4:	00900793          	li	a5,9
   198a8:	0947d863          	bge	a5,s4,19938 <__s2b+0x104>
   198ac:	01512223          	sw	s5,4(sp)
   198b0:	00940a93          	addi	s5,s0,9
   198b4:	000a8493          	mv	s1,s5
   198b8:	01440433          	add	s0,s0,s4
   198bc:	0004c683          	lbu	a3,0(s1)
   198c0:	00a00613          	li	a2,10
   198c4:	00090513          	mv	a0,s2
   198c8:	fd068693          	addi	a3,a3,-48
   198cc:	e1dff0ef          	jal	196e8 <__multadd>
   198d0:	00148493          	addi	s1,s1,1
   198d4:	00050593          	mv	a1,a0
   198d8:	fe8492e3          	bne	s1,s0,198bc <__s2b+0x88>
   198dc:	ff8a0413          	addi	s0,s4,-8
   198e0:	008a8433          	add	s0,s5,s0
   198e4:	00412a83          	lw	s5,4(sp)
   198e8:	033a5663          	bge	s4,s3,19914 <__s2b+0xe0>
   198ec:	414989b3          	sub	s3,s3,s4
   198f0:	013409b3          	add	s3,s0,s3
   198f4:	00044683          	lbu	a3,0(s0)
   198f8:	00a00613          	li	a2,10
   198fc:	00090513          	mv	a0,s2
   19900:	fd068693          	addi	a3,a3,-48
   19904:	de5ff0ef          	jal	196e8 <__multadd>
   19908:	00140413          	addi	s0,s0,1
   1990c:	00050593          	mv	a1,a0
   19910:	ff3412e3          	bne	s0,s3,198f4 <__s2b+0xc0>
   19914:	01c12083          	lw	ra,28(sp)
   19918:	01812403          	lw	s0,24(sp)
   1991c:	01412483          	lw	s1,20(sp)
   19920:	01012903          	lw	s2,16(sp)
   19924:	00c12983          	lw	s3,12(sp)
   19928:	00812a03          	lw	s4,8(sp)
   1992c:	00058513          	mv	a0,a1
   19930:	02010113          	addi	sp,sp,32
   19934:	00008067          	ret
   19938:	00a40413          	addi	s0,s0,10
   1993c:	00900a13          	li	s4,9
   19940:	fa9ff06f          	j	198e8 <__s2b+0xb4>
   19944:	00000593          	li	a1,0
   19948:	f41ff06f          	j	19888 <__s2b+0x54>
   1994c:	00007697          	auipc	a3,0x7
   19950:	03468693          	addi	a3,a3,52 # 20980 <_exit+0x148>
   19954:	00000613          	li	a2,0
   19958:	0d300593          	li	a1,211
   1995c:	00007517          	auipc	a0,0x7
   19960:	03850513          	addi	a0,a0,56 # 20994 <_exit+0x15c>
   19964:	01512223          	sw	s5,4(sp)
   19968:	22d010ef          	jal	1b394 <__assert_func>

0001996c <__hi0bits>:
   1996c:	00050793          	mv	a5,a0
   19970:	00010737          	lui	a4,0x10
   19974:	00000513          	li	a0,0
   19978:	00e7f663          	bgeu	a5,a4,19984 <__hi0bits+0x18>
   1997c:	01079793          	slli	a5,a5,0x10
   19980:	01000513          	li	a0,16
   19984:	01000737          	lui	a4,0x1000
   19988:	00e7f663          	bgeu	a5,a4,19994 <__hi0bits+0x28>
   1998c:	00850513          	addi	a0,a0,8
   19990:	00879793          	slli	a5,a5,0x8
   19994:	10000737          	lui	a4,0x10000
   19998:	00e7f663          	bgeu	a5,a4,199a4 <__hi0bits+0x38>
   1999c:	00450513          	addi	a0,a0,4
   199a0:	00479793          	slli	a5,a5,0x4
   199a4:	40000737          	lui	a4,0x40000
   199a8:	00e7ea63          	bltu	a5,a4,199bc <__hi0bits+0x50>
   199ac:	fff7c793          	not	a5,a5
   199b0:	01f7d793          	srli	a5,a5,0x1f
   199b4:	00f50533          	add	a0,a0,a5
   199b8:	00008067          	ret
   199bc:	00279793          	slli	a5,a5,0x2
   199c0:	0007ca63          	bltz	a5,199d4 <__hi0bits+0x68>
   199c4:	00179713          	slli	a4,a5,0x1
   199c8:	00074a63          	bltz	a4,199dc <__hi0bits+0x70>
   199cc:	02000513          	li	a0,32
   199d0:	00008067          	ret
   199d4:	00250513          	addi	a0,a0,2
   199d8:	00008067          	ret
   199dc:	00350513          	addi	a0,a0,3
   199e0:	00008067          	ret

000199e4 <__lo0bits>:
   199e4:	00052783          	lw	a5,0(a0)
   199e8:	00050713          	mv	a4,a0
   199ec:	0077f693          	andi	a3,a5,7
   199f0:	02068463          	beqz	a3,19a18 <__lo0bits+0x34>
   199f4:	0017f693          	andi	a3,a5,1
   199f8:	00000513          	li	a0,0
   199fc:	04069e63          	bnez	a3,19a58 <__lo0bits+0x74>
   19a00:	0027f693          	andi	a3,a5,2
   19a04:	0a068863          	beqz	a3,19ab4 <__lo0bits+0xd0>
   19a08:	0017d793          	srli	a5,a5,0x1
   19a0c:	00f72023          	sw	a5,0(a4) # 40000000 <__BSS_END__+0x3ffdd200>
   19a10:	00100513          	li	a0,1
   19a14:	00008067          	ret
   19a18:	01079693          	slli	a3,a5,0x10
   19a1c:	0106d693          	srli	a3,a3,0x10
   19a20:	00000513          	li	a0,0
   19a24:	06068e63          	beqz	a3,19aa0 <__lo0bits+0xbc>
   19a28:	0ff7f693          	zext.b	a3,a5
   19a2c:	06068063          	beqz	a3,19a8c <__lo0bits+0xa8>
   19a30:	00f7f693          	andi	a3,a5,15
   19a34:	04068263          	beqz	a3,19a78 <__lo0bits+0x94>
   19a38:	0037f693          	andi	a3,a5,3
   19a3c:	02068463          	beqz	a3,19a64 <__lo0bits+0x80>
   19a40:	0017f693          	andi	a3,a5,1
   19a44:	00069c63          	bnez	a3,19a5c <__lo0bits+0x78>
   19a48:	0017d793          	srli	a5,a5,0x1
   19a4c:	00150513          	addi	a0,a0,1
   19a50:	00079663          	bnez	a5,19a5c <__lo0bits+0x78>
   19a54:	02000513          	li	a0,32
   19a58:	00008067          	ret
   19a5c:	00f72023          	sw	a5,0(a4)
   19a60:	00008067          	ret
   19a64:	0027d793          	srli	a5,a5,0x2
   19a68:	0017f693          	andi	a3,a5,1
   19a6c:	00250513          	addi	a0,a0,2
   19a70:	fe0696e3          	bnez	a3,19a5c <__lo0bits+0x78>
   19a74:	fd5ff06f          	j	19a48 <__lo0bits+0x64>
   19a78:	0047d793          	srli	a5,a5,0x4
   19a7c:	0037f693          	andi	a3,a5,3
   19a80:	00450513          	addi	a0,a0,4
   19a84:	fa069ee3          	bnez	a3,19a40 <__lo0bits+0x5c>
   19a88:	fddff06f          	j	19a64 <__lo0bits+0x80>
   19a8c:	0087d793          	srli	a5,a5,0x8
   19a90:	00f7f693          	andi	a3,a5,15
   19a94:	00850513          	addi	a0,a0,8
   19a98:	fa0690e3          	bnez	a3,19a38 <__lo0bits+0x54>
   19a9c:	fddff06f          	j	19a78 <__lo0bits+0x94>
   19aa0:	0107d793          	srli	a5,a5,0x10
   19aa4:	0ff7f693          	zext.b	a3,a5
   19aa8:	01000513          	li	a0,16
   19aac:	f80692e3          	bnez	a3,19a30 <__lo0bits+0x4c>
   19ab0:	fddff06f          	j	19a8c <__lo0bits+0xa8>
   19ab4:	0027d793          	srli	a5,a5,0x2
   19ab8:	00f72023          	sw	a5,0(a4)
   19abc:	00200513          	li	a0,2
   19ac0:	00008067          	ret

00019ac4 <__i2b>:
   19ac4:	04452783          	lw	a5,68(a0)
   19ac8:	ff010113          	addi	sp,sp,-16
   19acc:	00812423          	sw	s0,8(sp)
   19ad0:	00912223          	sw	s1,4(sp)
   19ad4:	00112623          	sw	ra,12(sp)
   19ad8:	00050413          	mv	s0,a0
   19adc:	00058493          	mv	s1,a1
   19ae0:	02078c63          	beqz	a5,19b18 <__i2b+0x54>
   19ae4:	0047a503          	lw	a0,4(a5)
   19ae8:	06050263          	beqz	a0,19b4c <__i2b+0x88>
   19aec:	00052703          	lw	a4,0(a0)
   19af0:	00e7a223          	sw	a4,4(a5)
   19af4:	00c12083          	lw	ra,12(sp)
   19af8:	00812403          	lw	s0,8(sp)
   19afc:	00100793          	li	a5,1
   19b00:	00952a23          	sw	s1,20(a0)
   19b04:	00052623          	sw	zero,12(a0)
   19b08:	00f52823          	sw	a5,16(a0)
   19b0c:	00412483          	lw	s1,4(sp)
   19b10:	01010113          	addi	sp,sp,16
   19b14:	00008067          	ret
   19b18:	02100613          	li	a2,33
   19b1c:	00400593          	li	a1,4
   19b20:	0dd010ef          	jal	1b3fc <_calloc_r>
   19b24:	04a42223          	sw	a0,68(s0)
   19b28:	00050793          	mv	a5,a0
   19b2c:	fa051ce3          	bnez	a0,19ae4 <__i2b+0x20>
   19b30:	00007697          	auipc	a3,0x7
   19b34:	e5068693          	addi	a3,a3,-432 # 20980 <_exit+0x148>
   19b38:	00000613          	li	a2,0
   19b3c:	14500593          	li	a1,325
   19b40:	00007517          	auipc	a0,0x7
   19b44:	e5450513          	addi	a0,a0,-428 # 20994 <_exit+0x15c>
   19b48:	04d010ef          	jal	1b394 <__assert_func>
   19b4c:	01c00613          	li	a2,28
   19b50:	00100593          	li	a1,1
   19b54:	00040513          	mv	a0,s0
   19b58:	0a5010ef          	jal	1b3fc <_calloc_r>
   19b5c:	fc050ae3          	beqz	a0,19b30 <__i2b+0x6c>
   19b60:	00100793          	li	a5,1
   19b64:	00f52223          	sw	a5,4(a0)
   19b68:	00200793          	li	a5,2
   19b6c:	00f52423          	sw	a5,8(a0)
   19b70:	f85ff06f          	j	19af4 <__i2b+0x30>

00019b74 <__multiply>:
   19b74:	fe010113          	addi	sp,sp,-32
   19b78:	01212823          	sw	s2,16(sp)
   19b7c:	01312623          	sw	s3,12(sp)
   19b80:	0105a903          	lw	s2,16(a1)
   19b84:	01062983          	lw	s3,16(a2)
   19b88:	00912a23          	sw	s1,20(sp)
   19b8c:	01412423          	sw	s4,8(sp)
   19b90:	00112e23          	sw	ra,28(sp)
   19b94:	00812c23          	sw	s0,24(sp)
   19b98:	00058a13          	mv	s4,a1
   19b9c:	00060493          	mv	s1,a2
   19ba0:	01394c63          	blt	s2,s3,19bb8 <__multiply+0x44>
   19ba4:	00098713          	mv	a4,s3
   19ba8:	00058493          	mv	s1,a1
   19bac:	00090993          	mv	s3,s2
   19bb0:	00060a13          	mv	s4,a2
   19bb4:	00070913          	mv	s2,a4
   19bb8:	0084a783          	lw	a5,8(s1)
   19bbc:	0044a583          	lw	a1,4(s1)
   19bc0:	01298433          	add	s0,s3,s2
   19bc4:	0087a7b3          	slt	a5,a5,s0
   19bc8:	00f585b3          	add	a1,a1,a5
   19bcc:	a45ff0ef          	jal	19610 <_Balloc>
   19bd0:	1a050e63          	beqz	a0,19d8c <__multiply+0x218>
   19bd4:	01450313          	addi	t1,a0,20
   19bd8:	00241893          	slli	a7,s0,0x2
   19bdc:	011308b3          	add	a7,t1,a7
   19be0:	00030793          	mv	a5,t1
   19be4:	01137863          	bgeu	t1,a7,19bf4 <__multiply+0x80>
   19be8:	0007a023          	sw	zero,0(a5)
   19bec:	00478793          	addi	a5,a5,4
   19bf0:	ff17ece3          	bltu	a5,a7,19be8 <__multiply+0x74>
   19bf4:	014a0813          	addi	a6,s4,20
   19bf8:	00291e13          	slli	t3,s2,0x2
   19bfc:	01448e93          	addi	t4,s1,20
   19c00:	00299593          	slli	a1,s3,0x2
   19c04:	01c80e33          	add	t3,a6,t3
   19c08:	00be85b3          	add	a1,t4,a1
   19c0c:	13c87663          	bgeu	a6,t3,19d38 <__multiply+0x1c4>
   19c10:	01548793          	addi	a5,s1,21
   19c14:	00400f13          	li	t5,4
   19c18:	16f5f063          	bgeu	a1,a5,19d78 <__multiply+0x204>
   19c1c:	000106b7          	lui	a3,0x10
   19c20:	fff68693          	addi	a3,a3,-1 # ffff <exit-0xb5>
   19c24:	0180006f          	j	19c3c <__multiply+0xc8>
   19c28:	010fdf93          	srli	t6,t6,0x10
   19c2c:	080f9863          	bnez	t6,19cbc <__multiply+0x148>
   19c30:	00480813          	addi	a6,a6,4
   19c34:	00430313          	addi	t1,t1,4
   19c38:	11c87063          	bgeu	a6,t3,19d38 <__multiply+0x1c4>
   19c3c:	00082f83          	lw	t6,0(a6)
   19c40:	00dff3b3          	and	t2,t6,a3
   19c44:	fe0382e3          	beqz	t2,19c28 <__multiply+0xb4>
   19c48:	00030293          	mv	t0,t1
   19c4c:	000e8f93          	mv	t6,t4
   19c50:	00000493          	li	s1,0
   19c54:	000fa783          	lw	a5,0(t6)
   19c58:	0002a603          	lw	a2,0(t0)
   19c5c:	00428293          	addi	t0,t0,4
   19c60:	00d7f733          	and	a4,a5,a3
   19c64:	02770733          	mul	a4,a4,t2
   19c68:	0107d793          	srli	a5,a5,0x10
   19c6c:	00d67933          	and	s2,a2,a3
   19c70:	01065613          	srli	a2,a2,0x10
   19c74:	004f8f93          	addi	t6,t6,4
   19c78:	027787b3          	mul	a5,a5,t2
   19c7c:	01270733          	add	a4,a4,s2
   19c80:	00970733          	add	a4,a4,s1
   19c84:	01075493          	srli	s1,a4,0x10
   19c88:	00d77733          	and	a4,a4,a3
   19c8c:	00c787b3          	add	a5,a5,a2
   19c90:	009787b3          	add	a5,a5,s1
   19c94:	01079613          	slli	a2,a5,0x10
   19c98:	00e66733          	or	a4,a2,a4
   19c9c:	fee2ae23          	sw	a4,-4(t0)
   19ca0:	0107d493          	srli	s1,a5,0x10
   19ca4:	fabfe8e3          	bltu	t6,a1,19c54 <__multiply+0xe0>
   19ca8:	01e307b3          	add	a5,t1,t5
   19cac:	0097a023          	sw	s1,0(a5)
   19cb0:	00082f83          	lw	t6,0(a6)
   19cb4:	010fdf93          	srli	t6,t6,0x10
   19cb8:	f60f8ce3          	beqz	t6,19c30 <__multiply+0xbc>
   19cbc:	00032703          	lw	a4,0(t1)
   19cc0:	00030293          	mv	t0,t1
   19cc4:	000e8613          	mv	a2,t4
   19cc8:	00070493          	mv	s1,a4
   19ccc:	00000393          	li	t2,0
   19cd0:	00062783          	lw	a5,0(a2)
   19cd4:	0104d993          	srli	s3,s1,0x10
   19cd8:	00d77733          	and	a4,a4,a3
   19cdc:	00d7f7b3          	and	a5,a5,a3
   19ce0:	03f787b3          	mul	a5,a5,t6
   19ce4:	0042a483          	lw	s1,4(t0)
   19ce8:	00428293          	addi	t0,t0,4
   19cec:	00460613          	addi	a2,a2,4
   19cf0:	00d4f933          	and	s2,s1,a3
   19cf4:	007787b3          	add	a5,a5,t2
   19cf8:	013787b3          	add	a5,a5,s3
   19cfc:	01079393          	slli	t2,a5,0x10
   19d00:	00e3e733          	or	a4,t2,a4
   19d04:	fee2ae23          	sw	a4,-4(t0)
   19d08:	ffe65703          	lhu	a4,-2(a2)
   19d0c:	0107d793          	srli	a5,a5,0x10
   19d10:	03f70733          	mul	a4,a4,t6
   19d14:	01270733          	add	a4,a4,s2
   19d18:	00f70733          	add	a4,a4,a5
   19d1c:	01075393          	srli	t2,a4,0x10
   19d20:	fab668e3          	bltu	a2,a1,19cd0 <__multiply+0x15c>
   19d24:	01e307b3          	add	a5,t1,t5
   19d28:	00e7a023          	sw	a4,0(a5)
   19d2c:	00480813          	addi	a6,a6,4
   19d30:	00430313          	addi	t1,t1,4
   19d34:	f1c864e3          	bltu	a6,t3,19c3c <__multiply+0xc8>
   19d38:	00804863          	bgtz	s0,19d48 <__multiply+0x1d4>
   19d3c:	0180006f          	j	19d54 <__multiply+0x1e0>
   19d40:	fff40413          	addi	s0,s0,-1
   19d44:	00040863          	beqz	s0,19d54 <__multiply+0x1e0>
   19d48:	ffc8a783          	lw	a5,-4(a7)
   19d4c:	ffc88893          	addi	a7,a7,-4
   19d50:	fe0788e3          	beqz	a5,19d40 <__multiply+0x1cc>
   19d54:	01c12083          	lw	ra,28(sp)
   19d58:	00852823          	sw	s0,16(a0)
   19d5c:	01812403          	lw	s0,24(sp)
   19d60:	01412483          	lw	s1,20(sp)
   19d64:	01012903          	lw	s2,16(sp)
   19d68:	00c12983          	lw	s3,12(sp)
   19d6c:	00812a03          	lw	s4,8(sp)
   19d70:	02010113          	addi	sp,sp,32
   19d74:	00008067          	ret
   19d78:	409587b3          	sub	a5,a1,s1
   19d7c:	feb78793          	addi	a5,a5,-21
   19d80:	ffc7f793          	andi	a5,a5,-4
   19d84:	00478f13          	addi	t5,a5,4
   19d88:	e95ff06f          	j	19c1c <__multiply+0xa8>
   19d8c:	00007697          	auipc	a3,0x7
   19d90:	bf468693          	addi	a3,a3,-1036 # 20980 <_exit+0x148>
   19d94:	00000613          	li	a2,0
   19d98:	16200593          	li	a1,354
   19d9c:	00007517          	auipc	a0,0x7
   19da0:	bf850513          	addi	a0,a0,-1032 # 20994 <_exit+0x15c>
   19da4:	5f0010ef          	jal	1b394 <__assert_func>

00019da8 <__pow5mult>:
   19da8:	fe010113          	addi	sp,sp,-32
   19dac:	00812c23          	sw	s0,24(sp)
   19db0:	01212823          	sw	s2,16(sp)
   19db4:	01312623          	sw	s3,12(sp)
   19db8:	00112e23          	sw	ra,28(sp)
   19dbc:	00367793          	andi	a5,a2,3
   19dc0:	00060413          	mv	s0,a2
   19dc4:	00050993          	mv	s3,a0
   19dc8:	00058913          	mv	s2,a1
   19dcc:	0a079c63          	bnez	a5,19e84 <__pow5mult+0xdc>
   19dd0:	40245413          	srai	s0,s0,0x2
   19dd4:	06040a63          	beqz	s0,19e48 <__pow5mult+0xa0>
   19dd8:	00912a23          	sw	s1,20(sp)
   19ddc:	0409a483          	lw	s1,64(s3)
   19de0:	0c048663          	beqz	s1,19eac <__pow5mult+0x104>
   19de4:	00147793          	andi	a5,s0,1
   19de8:	02079063          	bnez	a5,19e08 <__pow5mult+0x60>
   19dec:	40145413          	srai	s0,s0,0x1
   19df0:	04040a63          	beqz	s0,19e44 <__pow5mult+0x9c>
   19df4:	0004a503          	lw	a0,0(s1)
   19df8:	06050663          	beqz	a0,19e64 <__pow5mult+0xbc>
   19dfc:	00050493          	mv	s1,a0
   19e00:	00147793          	andi	a5,s0,1
   19e04:	fe0784e3          	beqz	a5,19dec <__pow5mult+0x44>
   19e08:	00048613          	mv	a2,s1
   19e0c:	00090593          	mv	a1,s2
   19e10:	00098513          	mv	a0,s3
   19e14:	d61ff0ef          	jal	19b74 <__multiply>
   19e18:	02090063          	beqz	s2,19e38 <__pow5mult+0x90>
   19e1c:	00492703          	lw	a4,4(s2)
   19e20:	0449a783          	lw	a5,68(s3)
   19e24:	00271713          	slli	a4,a4,0x2
   19e28:	00e787b3          	add	a5,a5,a4
   19e2c:	0007a703          	lw	a4,0(a5)
   19e30:	00e92023          	sw	a4,0(s2)
   19e34:	0127a023          	sw	s2,0(a5)
   19e38:	40145413          	srai	s0,s0,0x1
   19e3c:	00050913          	mv	s2,a0
   19e40:	fa041ae3          	bnez	s0,19df4 <__pow5mult+0x4c>
   19e44:	01412483          	lw	s1,20(sp)
   19e48:	01c12083          	lw	ra,28(sp)
   19e4c:	01812403          	lw	s0,24(sp)
   19e50:	00c12983          	lw	s3,12(sp)
   19e54:	00090513          	mv	a0,s2
   19e58:	01012903          	lw	s2,16(sp)
   19e5c:	02010113          	addi	sp,sp,32
   19e60:	00008067          	ret
   19e64:	00048613          	mv	a2,s1
   19e68:	00048593          	mv	a1,s1
   19e6c:	00098513          	mv	a0,s3
   19e70:	d05ff0ef          	jal	19b74 <__multiply>
   19e74:	00a4a023          	sw	a0,0(s1)
   19e78:	00052023          	sw	zero,0(a0)
   19e7c:	00050493          	mv	s1,a0
   19e80:	f81ff06f          	j	19e00 <__pow5mult+0x58>
   19e84:	fff78793          	addi	a5,a5,-1
   19e88:	00007717          	auipc	a4,0x7
   19e8c:	00870713          	addi	a4,a4,8 # 20e90 <p05.0>
   19e90:	00279793          	slli	a5,a5,0x2
   19e94:	00f707b3          	add	a5,a4,a5
   19e98:	0007a603          	lw	a2,0(a5)
   19e9c:	00000693          	li	a3,0
   19ea0:	849ff0ef          	jal	196e8 <__multadd>
   19ea4:	00050913          	mv	s2,a0
   19ea8:	f29ff06f          	j	19dd0 <__pow5mult+0x28>
   19eac:	00100593          	li	a1,1
   19eb0:	00098513          	mv	a0,s3
   19eb4:	f5cff0ef          	jal	19610 <_Balloc>
   19eb8:	00050493          	mv	s1,a0
   19ebc:	02050063          	beqz	a0,19edc <__pow5mult+0x134>
   19ec0:	27100793          	li	a5,625
   19ec4:	00f52a23          	sw	a5,20(a0)
   19ec8:	00100793          	li	a5,1
   19ecc:	00f52823          	sw	a5,16(a0)
   19ed0:	04a9a023          	sw	a0,64(s3)
   19ed4:	00052023          	sw	zero,0(a0)
   19ed8:	f0dff06f          	j	19de4 <__pow5mult+0x3c>
   19edc:	00007697          	auipc	a3,0x7
   19ee0:	aa468693          	addi	a3,a3,-1372 # 20980 <_exit+0x148>
   19ee4:	00000613          	li	a2,0
   19ee8:	14500593          	li	a1,325
   19eec:	00007517          	auipc	a0,0x7
   19ef0:	aa850513          	addi	a0,a0,-1368 # 20994 <_exit+0x15c>
   19ef4:	4a0010ef          	jal	1b394 <__assert_func>

00019ef8 <__lshift>:
   19ef8:	fe010113          	addi	sp,sp,-32
   19efc:	01512223          	sw	s5,4(sp)
   19f00:	0105aa83          	lw	s5,16(a1)
   19f04:	0085a783          	lw	a5,8(a1)
   19f08:	01312623          	sw	s3,12(sp)
   19f0c:	40565993          	srai	s3,a2,0x5
   19f10:	01598ab3          	add	s5,s3,s5
   19f14:	00812c23          	sw	s0,24(sp)
   19f18:	00912a23          	sw	s1,20(sp)
   19f1c:	01212823          	sw	s2,16(sp)
   19f20:	01412423          	sw	s4,8(sp)
   19f24:	00112e23          	sw	ra,28(sp)
   19f28:	001a8913          	addi	s2,s5,1
   19f2c:	00058493          	mv	s1,a1
   19f30:	00060413          	mv	s0,a2
   19f34:	0045a583          	lw	a1,4(a1)
   19f38:	00050a13          	mv	s4,a0
   19f3c:	0127d863          	bge	a5,s2,19f4c <__lshift+0x54>
   19f40:	00179793          	slli	a5,a5,0x1
   19f44:	00158593          	addi	a1,a1,1
   19f48:	ff27cce3          	blt	a5,s2,19f40 <__lshift+0x48>
   19f4c:	000a0513          	mv	a0,s4
   19f50:	ec0ff0ef          	jal	19610 <_Balloc>
   19f54:	10050c63          	beqz	a0,1a06c <__lshift+0x174>
   19f58:	01450813          	addi	a6,a0,20
   19f5c:	03305463          	blez	s3,19f84 <__lshift+0x8c>
   19f60:	00598993          	addi	s3,s3,5
   19f64:	00299993          	slli	s3,s3,0x2
   19f68:	01350733          	add	a4,a0,s3
   19f6c:	00080793          	mv	a5,a6
   19f70:	00478793          	addi	a5,a5,4
   19f74:	fe07ae23          	sw	zero,-4(a5)
   19f78:	fee79ce3          	bne	a5,a4,19f70 <__lshift+0x78>
   19f7c:	fec98993          	addi	s3,s3,-20
   19f80:	01380833          	add	a6,a6,s3
   19f84:	0104a883          	lw	a7,16(s1)
   19f88:	01448793          	addi	a5,s1,20
   19f8c:	01f47613          	andi	a2,s0,31
   19f90:	00289893          	slli	a7,a7,0x2
   19f94:	011788b3          	add	a7,a5,a7
   19f98:	0a060463          	beqz	a2,1a040 <__lshift+0x148>
   19f9c:	02000593          	li	a1,32
   19fa0:	40c585b3          	sub	a1,a1,a2
   19fa4:	00080313          	mv	t1,a6
   19fa8:	00000693          	li	a3,0
   19fac:	0007a703          	lw	a4,0(a5)
   19fb0:	00430313          	addi	t1,t1,4
   19fb4:	00478793          	addi	a5,a5,4
   19fb8:	00c71733          	sll	a4,a4,a2
   19fbc:	00d76733          	or	a4,a4,a3
   19fc0:	fee32e23          	sw	a4,-4(t1)
   19fc4:	ffc7a683          	lw	a3,-4(a5)
   19fc8:	00b6d6b3          	srl	a3,a3,a1
   19fcc:	ff17e0e3          	bltu	a5,a7,19fac <__lshift+0xb4>
   19fd0:	01548793          	addi	a5,s1,21
   19fd4:	00400713          	li	a4,4
   19fd8:	00f8ea63          	bltu	a7,a5,19fec <__lshift+0xf4>
   19fdc:	409887b3          	sub	a5,a7,s1
   19fe0:	feb78793          	addi	a5,a5,-21
   19fe4:	ffc7f793          	andi	a5,a5,-4
   19fe8:	00478713          	addi	a4,a5,4
   19fec:	00e80833          	add	a6,a6,a4
   19ff0:	00d82023          	sw	a3,0(a6)
   19ff4:	00069463          	bnez	a3,19ffc <__lshift+0x104>
   19ff8:	000a8913          	mv	s2,s5
   19ffc:	0044a703          	lw	a4,4(s1)
   1a000:	044a2783          	lw	a5,68(s4)
   1a004:	01c12083          	lw	ra,28(sp)
   1a008:	00271713          	slli	a4,a4,0x2
   1a00c:	00e787b3          	add	a5,a5,a4
   1a010:	0007a703          	lw	a4,0(a5)
   1a014:	01252823          	sw	s2,16(a0)
   1a018:	01812403          	lw	s0,24(sp)
   1a01c:	00e4a023          	sw	a4,0(s1)
   1a020:	0097a023          	sw	s1,0(a5)
   1a024:	01012903          	lw	s2,16(sp)
   1a028:	01412483          	lw	s1,20(sp)
   1a02c:	00c12983          	lw	s3,12(sp)
   1a030:	00812a03          	lw	s4,8(sp)
   1a034:	00412a83          	lw	s5,4(sp)
   1a038:	02010113          	addi	sp,sp,32
   1a03c:	00008067          	ret
   1a040:	0007a703          	lw	a4,0(a5)
   1a044:	00478793          	addi	a5,a5,4
   1a048:	00480813          	addi	a6,a6,4
   1a04c:	fee82e23          	sw	a4,-4(a6)
   1a050:	fb17f4e3          	bgeu	a5,a7,19ff8 <__lshift+0x100>
   1a054:	0007a703          	lw	a4,0(a5)
   1a058:	00478793          	addi	a5,a5,4
   1a05c:	00480813          	addi	a6,a6,4
   1a060:	fee82e23          	sw	a4,-4(a6)
   1a064:	fd17eee3          	bltu	a5,a7,1a040 <__lshift+0x148>
   1a068:	f91ff06f          	j	19ff8 <__lshift+0x100>
   1a06c:	00007697          	auipc	a3,0x7
   1a070:	91468693          	addi	a3,a3,-1772 # 20980 <_exit+0x148>
   1a074:	00000613          	li	a2,0
   1a078:	1de00593          	li	a1,478
   1a07c:	00007517          	auipc	a0,0x7
   1a080:	91850513          	addi	a0,a0,-1768 # 20994 <_exit+0x15c>
   1a084:	310010ef          	jal	1b394 <__assert_func>

0001a088 <__mcmp>:
   1a088:	01052703          	lw	a4,16(a0)
   1a08c:	0105a783          	lw	a5,16(a1)
   1a090:	00050813          	mv	a6,a0
   1a094:	40f70533          	sub	a0,a4,a5
   1a098:	04f71263          	bne	a4,a5,1a0dc <__mcmp+0x54>
   1a09c:	00279793          	slli	a5,a5,0x2
   1a0a0:	01480813          	addi	a6,a6,20
   1a0a4:	01458593          	addi	a1,a1,20
   1a0a8:	00f80733          	add	a4,a6,a5
   1a0ac:	00f587b3          	add	a5,a1,a5
   1a0b0:	0080006f          	j	1a0b8 <__mcmp+0x30>
   1a0b4:	02e87463          	bgeu	a6,a4,1a0dc <__mcmp+0x54>
   1a0b8:	ffc72603          	lw	a2,-4(a4)
   1a0bc:	ffc7a683          	lw	a3,-4(a5)
   1a0c0:	ffc70713          	addi	a4,a4,-4
   1a0c4:	ffc78793          	addi	a5,a5,-4
   1a0c8:	fed606e3          	beq	a2,a3,1a0b4 <__mcmp+0x2c>
   1a0cc:	00100513          	li	a0,1
   1a0d0:	00d67663          	bgeu	a2,a3,1a0dc <__mcmp+0x54>
   1a0d4:	fff00513          	li	a0,-1
   1a0d8:	00008067          	ret
   1a0dc:	00008067          	ret

0001a0e0 <__mdiff>:
   1a0e0:	0105a703          	lw	a4,16(a1)
   1a0e4:	01062783          	lw	a5,16(a2)
   1a0e8:	ff010113          	addi	sp,sp,-16
   1a0ec:	00812423          	sw	s0,8(sp)
   1a0f0:	00912223          	sw	s1,4(sp)
   1a0f4:	00112623          	sw	ra,12(sp)
   1a0f8:	01212023          	sw	s2,0(sp)
   1a0fc:	00058413          	mv	s0,a1
   1a100:	00060493          	mv	s1,a2
   1a104:	40f706b3          	sub	a3,a4,a5
   1a108:	18f71e63          	bne	a4,a5,1a2a4 <__mdiff+0x1c4>
   1a10c:	00279693          	slli	a3,a5,0x2
   1a110:	01458613          	addi	a2,a1,20
   1a114:	01448713          	addi	a4,s1,20
   1a118:	00d607b3          	add	a5,a2,a3
   1a11c:	00d70733          	add	a4,a4,a3
   1a120:	0080006f          	j	1a128 <__mdiff+0x48>
   1a124:	18f67c63          	bgeu	a2,a5,1a2bc <__mdiff+0x1dc>
   1a128:	ffc7a583          	lw	a1,-4(a5)
   1a12c:	ffc72683          	lw	a3,-4(a4)
   1a130:	ffc78793          	addi	a5,a5,-4
   1a134:	ffc70713          	addi	a4,a4,-4
   1a138:	fed586e3          	beq	a1,a3,1a124 <__mdiff+0x44>
   1a13c:	00100913          	li	s2,1
   1a140:	00d5ea63          	bltu	a1,a3,1a154 <__mdiff+0x74>
   1a144:	00048793          	mv	a5,s1
   1a148:	00000913          	li	s2,0
   1a14c:	00040493          	mv	s1,s0
   1a150:	00078413          	mv	s0,a5
   1a154:	0044a583          	lw	a1,4(s1)
   1a158:	cb8ff0ef          	jal	19610 <_Balloc>
   1a15c:	1a050663          	beqz	a0,1a308 <__mdiff+0x228>
   1a160:	0104a883          	lw	a7,16(s1)
   1a164:	01042283          	lw	t0,16(s0)
   1a168:	01448f93          	addi	t6,s1,20
   1a16c:	00289313          	slli	t1,a7,0x2
   1a170:	01440813          	addi	a6,s0,20
   1a174:	00229293          	slli	t0,t0,0x2
   1a178:	01450593          	addi	a1,a0,20
   1a17c:	00010e37          	lui	t3,0x10
   1a180:	01252623          	sw	s2,12(a0)
   1a184:	006f8333          	add	t1,t6,t1
   1a188:	005802b3          	add	t0,a6,t0
   1a18c:	00058f13          	mv	t5,a1
   1a190:	000f8e93          	mv	t4,t6
   1a194:	00000693          	li	a3,0
   1a198:	fffe0e13          	addi	t3,t3,-1 # ffff <exit-0xb5>
   1a19c:	000ea703          	lw	a4,0(t4)
   1a1a0:	00082603          	lw	a2,0(a6)
   1a1a4:	004f0f13          	addi	t5,t5,4
   1a1a8:	01c777b3          	and	a5,a4,t3
   1a1ac:	01c673b3          	and	t2,a2,t3
   1a1b0:	407787b3          	sub	a5,a5,t2
   1a1b4:	00d787b3          	add	a5,a5,a3
   1a1b8:	01075713          	srli	a4,a4,0x10
   1a1bc:	01065613          	srli	a2,a2,0x10
   1a1c0:	4107d693          	srai	a3,a5,0x10
   1a1c4:	40c70733          	sub	a4,a4,a2
   1a1c8:	00d70733          	add	a4,a4,a3
   1a1cc:	01071693          	slli	a3,a4,0x10
   1a1d0:	01c7f7b3          	and	a5,a5,t3
   1a1d4:	00d7e7b3          	or	a5,a5,a3
   1a1d8:	00480813          	addi	a6,a6,4
   1a1dc:	feff2e23          	sw	a5,-4(t5)
   1a1e0:	004e8e93          	addi	t4,t4,4
   1a1e4:	41075693          	srai	a3,a4,0x10
   1a1e8:	fa586ae3          	bltu	a6,t0,1a19c <__mdiff+0xbc>
   1a1ec:	01540713          	addi	a4,s0,21
   1a1f0:	40828433          	sub	s0,t0,s0
   1a1f4:	feb40413          	addi	s0,s0,-21
   1a1f8:	00e2b2b3          	sltu	t0,t0,a4
   1a1fc:	0012cf13          	xori	t5,t0,1
   1a200:	00245413          	srli	s0,s0,0x2
   1a204:	00400713          	li	a4,4
   1a208:	0a028463          	beqz	t0,1a2b0 <__mdiff+0x1d0>
   1a20c:	00ef8fb3          	add	t6,t6,a4
   1a210:	00e58833          	add	a6,a1,a4
   1a214:	00010eb7          	lui	t4,0x10
   1a218:	00080e13          	mv	t3,a6
   1a21c:	000f8613          	mv	a2,t6
   1a220:	fffe8e93          	addi	t4,t4,-1 # ffff <exit-0xb5>
   1a224:	0c6ff463          	bgeu	t6,t1,1a2ec <__mdiff+0x20c>
   1a228:	00062783          	lw	a5,0(a2)
   1a22c:	004e0e13          	addi	t3,t3,4
   1a230:	00460613          	addi	a2,a2,4
   1a234:	01d7f733          	and	a4,a5,t4
   1a238:	00d70733          	add	a4,a4,a3
   1a23c:	0107d593          	srli	a1,a5,0x10
   1a240:	41075713          	srai	a4,a4,0x10
   1a244:	00b70733          	add	a4,a4,a1
   1a248:	00d787b3          	add	a5,a5,a3
   1a24c:	01d7f7b3          	and	a5,a5,t4
   1a250:	01071693          	slli	a3,a4,0x10
   1a254:	00d7e7b3          	or	a5,a5,a3
   1a258:	fefe2e23          	sw	a5,-4(t3)
   1a25c:	41075693          	srai	a3,a4,0x10
   1a260:	fc6664e3          	bltu	a2,t1,1a228 <__mdiff+0x148>
   1a264:	fff30313          	addi	t1,t1,-1
   1a268:	41f30333          	sub	t1,t1,t6
   1a26c:	ffc37313          	andi	t1,t1,-4
   1a270:	00680733          	add	a4,a6,t1
   1a274:	00079a63          	bnez	a5,1a288 <__mdiff+0x1a8>
   1a278:	ffc72783          	lw	a5,-4(a4)
   1a27c:	fff88893          	addi	a7,a7,-1
   1a280:	ffc70713          	addi	a4,a4,-4
   1a284:	fe078ae3          	beqz	a5,1a278 <__mdiff+0x198>
   1a288:	00c12083          	lw	ra,12(sp)
   1a28c:	00812403          	lw	s0,8(sp)
   1a290:	01152823          	sw	a7,16(a0)
   1a294:	00412483          	lw	s1,4(sp)
   1a298:	00012903          	lw	s2,0(sp)
   1a29c:	01010113          	addi	sp,sp,16
   1a2a0:	00008067          	ret
   1a2a4:	00100913          	li	s2,1
   1a2a8:	e806dee3          	bgez	a3,1a144 <__mdiff+0x64>
   1a2ac:	ea9ff06f          	j	1a154 <__mdiff+0x74>
   1a2b0:	00140713          	addi	a4,s0,1
   1a2b4:	00271713          	slli	a4,a4,0x2
   1a2b8:	f55ff06f          	j	1a20c <__mdiff+0x12c>
   1a2bc:	00000593          	li	a1,0
   1a2c0:	b50ff0ef          	jal	19610 <_Balloc>
   1a2c4:	06050063          	beqz	a0,1a324 <__mdiff+0x244>
   1a2c8:	00c12083          	lw	ra,12(sp)
   1a2cc:	00812403          	lw	s0,8(sp)
   1a2d0:	00100793          	li	a5,1
   1a2d4:	00f52823          	sw	a5,16(a0)
   1a2d8:	00052a23          	sw	zero,20(a0)
   1a2dc:	00412483          	lw	s1,4(sp)
   1a2e0:	00012903          	lw	s2,0(sp)
   1a2e4:	01010113          	addi	sp,sp,16
   1a2e8:	00008067          	ret
   1a2ec:	00000713          	li	a4,0
   1a2f0:	000f1663          	bnez	t5,1a2fc <__mdiff+0x21c>
   1a2f4:	00e58733          	add	a4,a1,a4
   1a2f8:	f7dff06f          	j	1a274 <__mdiff+0x194>
   1a2fc:	00241713          	slli	a4,s0,0x2
   1a300:	00e58733          	add	a4,a1,a4
   1a304:	f71ff06f          	j	1a274 <__mdiff+0x194>
   1a308:	00006697          	auipc	a3,0x6
   1a30c:	67868693          	addi	a3,a3,1656 # 20980 <_exit+0x148>
   1a310:	00000613          	li	a2,0
   1a314:	24500593          	li	a1,581
   1a318:	00006517          	auipc	a0,0x6
   1a31c:	67c50513          	addi	a0,a0,1660 # 20994 <_exit+0x15c>
   1a320:	074010ef          	jal	1b394 <__assert_func>
   1a324:	00006697          	auipc	a3,0x6
   1a328:	65c68693          	addi	a3,a3,1628 # 20980 <_exit+0x148>
   1a32c:	00000613          	li	a2,0
   1a330:	23700593          	li	a1,567
   1a334:	00006517          	auipc	a0,0x6
   1a338:	66050513          	addi	a0,a0,1632 # 20994 <_exit+0x15c>
   1a33c:	058010ef          	jal	1b394 <__assert_func>

0001a340 <__ulp>:
   1a340:	7ff007b7          	lui	a5,0x7ff00
   1a344:	00b7f5b3          	and	a1,a5,a1
   1a348:	fcc007b7          	lui	a5,0xfcc00
   1a34c:	00f585b3          	add	a1,a1,a5
   1a350:	00000793          	li	a5,0
   1a354:	00b05663          	blez	a1,1a360 <__ulp+0x20>
   1a358:	00078513          	mv	a0,a5
   1a35c:	00008067          	ret
   1a360:	40b005b3          	neg	a1,a1
   1a364:	4145d593          	srai	a1,a1,0x14
   1a368:	01300793          	li	a5,19
   1a36c:	00b7cc63          	blt	a5,a1,1a384 <__ulp+0x44>
   1a370:	000807b7          	lui	a5,0x80
   1a374:	40b7d5b3          	sra	a1,a5,a1
   1a378:	00000793          	li	a5,0
   1a37c:	00078513          	mv	a0,a5
   1a380:	00008067          	ret
   1a384:	fec58593          	addi	a1,a1,-20
   1a388:	01e00713          	li	a4,30
   1a38c:	00100793          	li	a5,1
   1a390:	00b74663          	blt	a4,a1,1a39c <__ulp+0x5c>
   1a394:	800007b7          	lui	a5,0x80000
   1a398:	00b7d7b3          	srl	a5,a5,a1
   1a39c:	00000593          	li	a1,0
   1a3a0:	00078513          	mv	a0,a5
   1a3a4:	00008067          	ret

0001a3a8 <__b2d>:
   1a3a8:	fe010113          	addi	sp,sp,-32
   1a3ac:	00912a23          	sw	s1,20(sp)
   1a3b0:	01052483          	lw	s1,16(a0)
   1a3b4:	00812c23          	sw	s0,24(sp)
   1a3b8:	01450413          	addi	s0,a0,20
   1a3bc:	00249493          	slli	s1,s1,0x2
   1a3c0:	009404b3          	add	s1,s0,s1
   1a3c4:	01212823          	sw	s2,16(sp)
   1a3c8:	ffc4a903          	lw	s2,-4(s1)
   1a3cc:	01312623          	sw	s3,12(sp)
   1a3d0:	01412423          	sw	s4,8(sp)
   1a3d4:	00090513          	mv	a0,s2
   1a3d8:	00058993          	mv	s3,a1
   1a3dc:	00112e23          	sw	ra,28(sp)
   1a3e0:	d8cff0ef          	jal	1996c <__hi0bits>
   1a3e4:	02000713          	li	a4,32
   1a3e8:	40a707b3          	sub	a5,a4,a0
   1a3ec:	00f9a023          	sw	a5,0(s3)
   1a3f0:	00a00793          	li	a5,10
   1a3f4:	ffc48a13          	addi	s4,s1,-4
   1a3f8:	08a7dc63          	bge	a5,a0,1a490 <__b2d+0xe8>
   1a3fc:	ff550613          	addi	a2,a0,-11
   1a400:	05447063          	bgeu	s0,s4,1a440 <__b2d+0x98>
   1a404:	ff84a783          	lw	a5,-8(s1)
   1a408:	04060c63          	beqz	a2,1a460 <__b2d+0xb8>
   1a40c:	40c706b3          	sub	a3,a4,a2
   1a410:	00d7d733          	srl	a4,a5,a3
   1a414:	00c91933          	sll	s2,s2,a2
   1a418:	00e96933          	or	s2,s2,a4
   1a41c:	ff848593          	addi	a1,s1,-8
   1a420:	3ff00737          	lui	a4,0x3ff00
   1a424:	00e96733          	or	a4,s2,a4
   1a428:	00c797b3          	sll	a5,a5,a2
   1a42c:	02b47e63          	bgeu	s0,a1,1a468 <__b2d+0xc0>
   1a430:	ff44a603          	lw	a2,-12(s1)
   1a434:	00d656b3          	srl	a3,a2,a3
   1a438:	00d7e7b3          	or	a5,a5,a3
   1a43c:	02c0006f          	j	1a468 <__b2d+0xc0>
   1a440:	00b00793          	li	a5,11
   1a444:	00f50c63          	beq	a0,a5,1a45c <__b2d+0xb4>
   1a448:	00c91933          	sll	s2,s2,a2
   1a44c:	3ff00737          	lui	a4,0x3ff00
   1a450:	00e96733          	or	a4,s2,a4
   1a454:	00000793          	li	a5,0
   1a458:	0100006f          	j	1a468 <__b2d+0xc0>
   1a45c:	00000793          	li	a5,0
   1a460:	3ff00737          	lui	a4,0x3ff00
   1a464:	00e96733          	or	a4,s2,a4
   1a468:	01c12083          	lw	ra,28(sp)
   1a46c:	01812403          	lw	s0,24(sp)
   1a470:	01412483          	lw	s1,20(sp)
   1a474:	01012903          	lw	s2,16(sp)
   1a478:	00c12983          	lw	s3,12(sp)
   1a47c:	00812a03          	lw	s4,8(sp)
   1a480:	00078513          	mv	a0,a5
   1a484:	00070593          	mv	a1,a4
   1a488:	02010113          	addi	sp,sp,32
   1a48c:	00008067          	ret
   1a490:	00b00693          	li	a3,11
   1a494:	40a686b3          	sub	a3,a3,a0
   1a498:	3ff007b7          	lui	a5,0x3ff00
   1a49c:	00d95733          	srl	a4,s2,a3
   1a4a0:	00f76733          	or	a4,a4,a5
   1a4a4:	00000793          	li	a5,0
   1a4a8:	01447663          	bgeu	s0,s4,1a4b4 <__b2d+0x10c>
   1a4ac:	ff84a783          	lw	a5,-8(s1)
   1a4b0:	00d7d7b3          	srl	a5,a5,a3
   1a4b4:	01550513          	addi	a0,a0,21
   1a4b8:	00a91933          	sll	s2,s2,a0
   1a4bc:	00f967b3          	or	a5,s2,a5
   1a4c0:	fa9ff06f          	j	1a468 <__b2d+0xc0>

0001a4c4 <__d2b>:
   1a4c4:	fd010113          	addi	sp,sp,-48
   1a4c8:	01512a23          	sw	s5,20(sp)
   1a4cc:	00058a93          	mv	s5,a1
   1a4d0:	00100593          	li	a1,1
   1a4d4:	02912223          	sw	s1,36(sp)
   1a4d8:	01312e23          	sw	s3,28(sp)
   1a4dc:	01412c23          	sw	s4,24(sp)
   1a4e0:	02112623          	sw	ra,44(sp)
   1a4e4:	02812423          	sw	s0,40(sp)
   1a4e8:	03212023          	sw	s2,32(sp)
   1a4ec:	00060493          	mv	s1,a2
   1a4f0:	00068a13          	mv	s4,a3
   1a4f4:	00070993          	mv	s3,a4
   1a4f8:	918ff0ef          	jal	19610 <_Balloc>
   1a4fc:	10050263          	beqz	a0,1a600 <__d2b+0x13c>
   1a500:	00100737          	lui	a4,0x100
   1a504:	0144d913          	srli	s2,s1,0x14
   1a508:	fff70793          	addi	a5,a4,-1 # fffff <__BSS_END__+0xdd1ff>
   1a50c:	7ff97913          	andi	s2,s2,2047
   1a510:	00050413          	mv	s0,a0
   1a514:	0097f7b3          	and	a5,a5,s1
   1a518:	00090463          	beqz	s2,1a520 <__d2b+0x5c>
   1a51c:	00e7e7b3          	or	a5,a5,a4
   1a520:	00f12623          	sw	a5,12(sp)
   1a524:	060a9263          	bnez	s5,1a588 <__d2b+0xc4>
   1a528:	00c10513          	addi	a0,sp,12
   1a52c:	cb8ff0ef          	jal	199e4 <__lo0bits>
   1a530:	00c12703          	lw	a4,12(sp)
   1a534:	00100493          	li	s1,1
   1a538:	00942823          	sw	s1,16(s0)
   1a53c:	00e42a23          	sw	a4,20(s0)
   1a540:	02050793          	addi	a5,a0,32
   1a544:	08090863          	beqz	s2,1a5d4 <__d2b+0x110>
   1a548:	bcd90913          	addi	s2,s2,-1075
   1a54c:	00f90933          	add	s2,s2,a5
   1a550:	03500493          	li	s1,53
   1a554:	012a2023          	sw	s2,0(s4)
   1a558:	40f48533          	sub	a0,s1,a5
   1a55c:	00a9a023          	sw	a0,0(s3)
   1a560:	02c12083          	lw	ra,44(sp)
   1a564:	00040513          	mv	a0,s0
   1a568:	02812403          	lw	s0,40(sp)
   1a56c:	02412483          	lw	s1,36(sp)
   1a570:	02012903          	lw	s2,32(sp)
   1a574:	01c12983          	lw	s3,28(sp)
   1a578:	01812a03          	lw	s4,24(sp)
   1a57c:	01412a83          	lw	s5,20(sp)
   1a580:	03010113          	addi	sp,sp,48
   1a584:	00008067          	ret
   1a588:	00810513          	addi	a0,sp,8
   1a58c:	01512423          	sw	s5,8(sp)
   1a590:	c54ff0ef          	jal	199e4 <__lo0bits>
   1a594:	00c12703          	lw	a4,12(sp)
   1a598:	00050793          	mv	a5,a0
   1a59c:	04050e63          	beqz	a0,1a5f8 <__d2b+0x134>
   1a5a0:	00812603          	lw	a2,8(sp)
   1a5a4:	02000693          	li	a3,32
   1a5a8:	40a686b3          	sub	a3,a3,a0
   1a5ac:	00d716b3          	sll	a3,a4,a3
   1a5b0:	00a75733          	srl	a4,a4,a0
   1a5b4:	00c6e6b3          	or	a3,a3,a2
   1a5b8:	00e12623          	sw	a4,12(sp)
   1a5bc:	00e034b3          	snez	s1,a4
   1a5c0:	00148493          	addi	s1,s1,1
   1a5c4:	00d42a23          	sw	a3,20(s0)
   1a5c8:	00e42c23          	sw	a4,24(s0)
   1a5cc:	00942823          	sw	s1,16(s0)
   1a5d0:	f6091ce3          	bnez	s2,1a548 <__d2b+0x84>
   1a5d4:	00249713          	slli	a4,s1,0x2
   1a5d8:	00e40733          	add	a4,s0,a4
   1a5dc:	01072503          	lw	a0,16(a4)
   1a5e0:	bce78793          	addi	a5,a5,-1074 # 3feffbce <__BSS_END__+0x3fedcdce>
   1a5e4:	00fa2023          	sw	a5,0(s4)
   1a5e8:	b84ff0ef          	jal	1996c <__hi0bits>
   1a5ec:	00549493          	slli	s1,s1,0x5
   1a5f0:	40a48533          	sub	a0,s1,a0
   1a5f4:	f69ff06f          	j	1a55c <__d2b+0x98>
   1a5f8:	00812683          	lw	a3,8(sp)
   1a5fc:	fc1ff06f          	j	1a5bc <__d2b+0xf8>
   1a600:	00006697          	auipc	a3,0x6
   1a604:	38068693          	addi	a3,a3,896 # 20980 <_exit+0x148>
   1a608:	00000613          	li	a2,0
   1a60c:	30f00593          	li	a1,783
   1a610:	00006517          	auipc	a0,0x6
   1a614:	38450513          	addi	a0,a0,900 # 20994 <_exit+0x15c>
   1a618:	57d000ef          	jal	1b394 <__assert_func>

0001a61c <__ratio>:
   1a61c:	fd010113          	addi	sp,sp,-48
   1a620:	03212023          	sw	s2,32(sp)
   1a624:	00058913          	mv	s2,a1
   1a628:	00810593          	addi	a1,sp,8
   1a62c:	02112623          	sw	ra,44(sp)
   1a630:	02812423          	sw	s0,40(sp)
   1a634:	02912223          	sw	s1,36(sp)
   1a638:	01312e23          	sw	s3,28(sp)
   1a63c:	00050993          	mv	s3,a0
   1a640:	d69ff0ef          	jal	1a3a8 <__b2d>
   1a644:	00050493          	mv	s1,a0
   1a648:	00058413          	mv	s0,a1
   1a64c:	00090513          	mv	a0,s2
   1a650:	00c10593          	addi	a1,sp,12
   1a654:	d55ff0ef          	jal	1a3a8 <__b2d>
   1a658:	01092703          	lw	a4,16(s2)
   1a65c:	0109a783          	lw	a5,16(s3)
   1a660:	00c12683          	lw	a3,12(sp)
   1a664:	40e787b3          	sub	a5,a5,a4
   1a668:	00812703          	lw	a4,8(sp)
   1a66c:	00579793          	slli	a5,a5,0x5
   1a670:	40d70733          	sub	a4,a4,a3
   1a674:	00e787b3          	add	a5,a5,a4
   1a678:	00050713          	mv	a4,a0
   1a67c:	02f05e63          	blez	a5,1a6b8 <__ratio+0x9c>
   1a680:	01479793          	slli	a5,a5,0x14
   1a684:	00878433          	add	s0,a5,s0
   1a688:	00058693          	mv	a3,a1
   1a68c:	00048513          	mv	a0,s1
   1a690:	00040593          	mv	a1,s0
   1a694:	00070613          	mv	a2,a4
   1a698:	41d010ef          	jal	1c2b4 <__divdf3>
   1a69c:	02c12083          	lw	ra,44(sp)
   1a6a0:	02812403          	lw	s0,40(sp)
   1a6a4:	02412483          	lw	s1,36(sp)
   1a6a8:	02012903          	lw	s2,32(sp)
   1a6ac:	01c12983          	lw	s3,28(sp)
   1a6b0:	03010113          	addi	sp,sp,48
   1a6b4:	00008067          	ret
   1a6b8:	01479793          	slli	a5,a5,0x14
   1a6bc:	40f585b3          	sub	a1,a1,a5
   1a6c0:	fc9ff06f          	j	1a688 <__ratio+0x6c>

0001a6c4 <_mprec_log10>:
   1a6c4:	ff010113          	addi	sp,sp,-16
   1a6c8:	01212023          	sw	s2,0(sp)
   1a6cc:	00112623          	sw	ra,12(sp)
   1a6d0:	01700793          	li	a5,23
   1a6d4:	00050913          	mv	s2,a0
   1a6d8:	06a7d263          	bge	a5,a0,1a73c <_mprec_log10+0x78>
   1a6dc:	00008717          	auipc	a4,0x8
   1a6e0:	2cc70713          	addi	a4,a4,716 # 229a8 <__SDATA_BEGIN__+0x30>
   1a6e4:	00072783          	lw	a5,0(a4)
   1a6e8:	00472583          	lw	a1,4(a4)
   1a6ec:	00008717          	auipc	a4,0x8
   1a6f0:	2c470713          	addi	a4,a4,708 # 229b0 <__SDATA_BEGIN__+0x38>
   1a6f4:	00812423          	sw	s0,8(sp)
   1a6f8:	00912223          	sw	s1,4(sp)
   1a6fc:	00072403          	lw	s0,0(a4)
   1a700:	00472483          	lw	s1,4(a4)
   1a704:	00078513          	mv	a0,a5
   1a708:	00040613          	mv	a2,s0
   1a70c:	00048693          	mv	a3,s1
   1a710:	4c8020ef          	jal	1cbd8 <__muldf3>
   1a714:	fff90913          	addi	s2,s2,-1
   1a718:	00050793          	mv	a5,a0
   1a71c:	fe0914e3          	bnez	s2,1a704 <_mprec_log10+0x40>
   1a720:	00812403          	lw	s0,8(sp)
   1a724:	00c12083          	lw	ra,12(sp)
   1a728:	00412483          	lw	s1,4(sp)
   1a72c:	00012903          	lw	s2,0(sp)
   1a730:	00078513          	mv	a0,a5
   1a734:	01010113          	addi	sp,sp,16
   1a738:	00008067          	ret
   1a73c:	00351913          	slli	s2,a0,0x3
   1a740:	00006717          	auipc	a4,0x6
   1a744:	7b070713          	addi	a4,a4,1968 # 20ef0 <__mprec_tens>
   1a748:	01270733          	add	a4,a4,s2
   1a74c:	00072783          	lw	a5,0(a4)
   1a750:	00c12083          	lw	ra,12(sp)
   1a754:	00472583          	lw	a1,4(a4)
   1a758:	00012903          	lw	s2,0(sp)
   1a75c:	00078513          	mv	a0,a5
   1a760:	01010113          	addi	sp,sp,16
   1a764:	00008067          	ret

0001a768 <__copybits>:
   1a768:	01062683          	lw	a3,16(a2)
   1a76c:	fff58593          	addi	a1,a1,-1
   1a770:	4055d593          	srai	a1,a1,0x5
   1a774:	00158593          	addi	a1,a1,1
   1a778:	01460793          	addi	a5,a2,20
   1a77c:	00269693          	slli	a3,a3,0x2
   1a780:	00259593          	slli	a1,a1,0x2
   1a784:	00d786b3          	add	a3,a5,a3
   1a788:	00b505b3          	add	a1,a0,a1
   1a78c:	02d7f863          	bgeu	a5,a3,1a7bc <__copybits+0x54>
   1a790:	00050713          	mv	a4,a0
   1a794:	0007a803          	lw	a6,0(a5)
   1a798:	00478793          	addi	a5,a5,4
   1a79c:	00470713          	addi	a4,a4,4
   1a7a0:	ff072e23          	sw	a6,-4(a4)
   1a7a4:	fed7e8e3          	bltu	a5,a3,1a794 <__copybits+0x2c>
   1a7a8:	40c687b3          	sub	a5,a3,a2
   1a7ac:	feb78793          	addi	a5,a5,-21
   1a7b0:	ffc7f793          	andi	a5,a5,-4
   1a7b4:	00478793          	addi	a5,a5,4
   1a7b8:	00f50533          	add	a0,a0,a5
   1a7bc:	00b57863          	bgeu	a0,a1,1a7cc <__copybits+0x64>
   1a7c0:	00450513          	addi	a0,a0,4
   1a7c4:	fe052e23          	sw	zero,-4(a0)
   1a7c8:	feb56ce3          	bltu	a0,a1,1a7c0 <__copybits+0x58>
   1a7cc:	00008067          	ret

0001a7d0 <__any_on>:
   1a7d0:	01052703          	lw	a4,16(a0)
   1a7d4:	4055d613          	srai	a2,a1,0x5
   1a7d8:	01450693          	addi	a3,a0,20
   1a7dc:	02c75263          	bge	a4,a2,1a800 <__any_on+0x30>
   1a7e0:	00271713          	slli	a4,a4,0x2
   1a7e4:	00e687b3          	add	a5,a3,a4
   1a7e8:	04f6f263          	bgeu	a3,a5,1a82c <__any_on+0x5c>
   1a7ec:	ffc7a703          	lw	a4,-4(a5)
   1a7f0:	ffc78793          	addi	a5,a5,-4
   1a7f4:	fe070ae3          	beqz	a4,1a7e8 <__any_on+0x18>
   1a7f8:	00100513          	li	a0,1
   1a7fc:	00008067          	ret
   1a800:	00261793          	slli	a5,a2,0x2
   1a804:	00f687b3          	add	a5,a3,a5
   1a808:	fee650e3          	bge	a2,a4,1a7e8 <__any_on+0x18>
   1a80c:	01f5f593          	andi	a1,a1,31
   1a810:	fc058ce3          	beqz	a1,1a7e8 <__any_on+0x18>
   1a814:	0007a603          	lw	a2,0(a5)
   1a818:	00100513          	li	a0,1
   1a81c:	00b65733          	srl	a4,a2,a1
   1a820:	00b71733          	sll	a4,a4,a1
   1a824:	fce602e3          	beq	a2,a4,1a7e8 <__any_on+0x18>
   1a828:	00008067          	ret
   1a82c:	00000513          	li	a0,0
   1a830:	00008067          	ret

0001a834 <_realloc_r>:
   1a834:	fd010113          	addi	sp,sp,-48
   1a838:	02912223          	sw	s1,36(sp)
   1a83c:	02112623          	sw	ra,44(sp)
   1a840:	00060493          	mv	s1,a2
   1a844:	1e058863          	beqz	a1,1aa34 <_realloc_r+0x200>
   1a848:	02812423          	sw	s0,40(sp)
   1a84c:	03212023          	sw	s2,32(sp)
   1a850:	00058413          	mv	s0,a1
   1a854:	01312e23          	sw	s3,28(sp)
   1a858:	01512a23          	sw	s5,20(sp)
   1a85c:	01412c23          	sw	s4,24(sp)
   1a860:	00050913          	mv	s2,a0
   1a864:	b58f70ef          	jal	11bbc <__malloc_lock>
   1a868:	ffc42703          	lw	a4,-4(s0)
   1a86c:	00b48793          	addi	a5,s1,11
   1a870:	01600693          	li	a3,22
   1a874:	ff840a93          	addi	s5,s0,-8
   1a878:	ffc77993          	andi	s3,a4,-4
   1a87c:	10f6f263          	bgeu	a3,a5,1a980 <_realloc_r+0x14c>
   1a880:	ff87fa13          	andi	s4,a5,-8
   1a884:	1007c263          	bltz	a5,1a988 <_realloc_r+0x154>
   1a888:	109a6063          	bltu	s4,s1,1a988 <_realloc_r+0x154>
   1a88c:	1349d263          	bge	s3,s4,1a9b0 <_realloc_r+0x17c>
   1a890:	01812423          	sw	s8,8(sp)
   1a894:	00008c17          	auipc	s8,0x8
   1a898:	b6cc0c13          	addi	s8,s8,-1172 # 22400 <__malloc_av_>
   1a89c:	008c2603          	lw	a2,8(s8)
   1a8a0:	013a86b3          	add	a3,s5,s3
   1a8a4:	0046a783          	lw	a5,4(a3)
   1a8a8:	1cd60863          	beq	a2,a3,1aa78 <_realloc_r+0x244>
   1a8ac:	ffe7f613          	andi	a2,a5,-2
   1a8b0:	00c68633          	add	a2,a3,a2
   1a8b4:	00462603          	lw	a2,4(a2)
   1a8b8:	00167613          	andi	a2,a2,1
   1a8bc:	14061a63          	bnez	a2,1aa10 <_realloc_r+0x1dc>
   1a8c0:	ffc7f793          	andi	a5,a5,-4
   1a8c4:	00f98633          	add	a2,s3,a5
   1a8c8:	0d465863          	bge	a2,s4,1a998 <_realloc_r+0x164>
   1a8cc:	00177713          	andi	a4,a4,1
   1a8d0:	02071c63          	bnez	a4,1a908 <_realloc_r+0xd4>
   1a8d4:	01712623          	sw	s7,12(sp)
   1a8d8:	ff842b83          	lw	s7,-8(s0)
   1a8dc:	01612823          	sw	s6,16(sp)
   1a8e0:	417a8bb3          	sub	s7,s5,s7
   1a8e4:	004ba703          	lw	a4,4(s7)
   1a8e8:	ffc77713          	andi	a4,a4,-4
   1a8ec:	00e787b3          	add	a5,a5,a4
   1a8f0:	01378b33          	add	s6,a5,s3
   1a8f4:	334b5c63          	bge	s6,s4,1ac2c <_realloc_r+0x3f8>
   1a8f8:	00e98b33          	add	s6,s3,a4
   1a8fc:	294b5863          	bge	s6,s4,1ab8c <_realloc_r+0x358>
   1a900:	01012b03          	lw	s6,16(sp)
   1a904:	00c12b83          	lw	s7,12(sp)
   1a908:	00048593          	mv	a1,s1
   1a90c:	00090513          	mv	a0,s2
   1a910:	ae5f60ef          	jal	113f4 <_malloc_r>
   1a914:	00050493          	mv	s1,a0
   1a918:	40050863          	beqz	a0,1ad28 <_realloc_r+0x4f4>
   1a91c:	ffc42783          	lw	a5,-4(s0)
   1a920:	ff850713          	addi	a4,a0,-8
   1a924:	ffe7f793          	andi	a5,a5,-2
   1a928:	00fa87b3          	add	a5,s5,a5
   1a92c:	24e78663          	beq	a5,a4,1ab78 <_realloc_r+0x344>
   1a930:	ffc98613          	addi	a2,s3,-4
   1a934:	02400793          	li	a5,36
   1a938:	2ec7e463          	bltu	a5,a2,1ac20 <_realloc_r+0x3ec>
   1a93c:	01300713          	li	a4,19
   1a940:	20c76a63          	bltu	a4,a2,1ab54 <_realloc_r+0x320>
   1a944:	00050793          	mv	a5,a0
   1a948:	00040713          	mv	a4,s0
   1a94c:	00072683          	lw	a3,0(a4)
   1a950:	00d7a023          	sw	a3,0(a5)
   1a954:	00472683          	lw	a3,4(a4)
   1a958:	00d7a223          	sw	a3,4(a5)
   1a95c:	00872703          	lw	a4,8(a4)
   1a960:	00e7a423          	sw	a4,8(a5)
   1a964:	00040593          	mv	a1,s0
   1a968:	00090513          	mv	a0,s2
   1a96c:	f84f60ef          	jal	110f0 <_free_r>
   1a970:	00090513          	mv	a0,s2
   1a974:	a4cf70ef          	jal	11bc0 <__malloc_unlock>
   1a978:	00812c03          	lw	s8,8(sp)
   1a97c:	06c0006f          	j	1a9e8 <_realloc_r+0x1b4>
   1a980:	01000a13          	li	s4,16
   1a984:	f09a74e3          	bgeu	s4,s1,1a88c <_realloc_r+0x58>
   1a988:	00c00793          	li	a5,12
   1a98c:	00f92023          	sw	a5,0(s2)
   1a990:	00000493          	li	s1,0
   1a994:	0540006f          	j	1a9e8 <_realloc_r+0x1b4>
   1a998:	00c6a783          	lw	a5,12(a3)
   1a99c:	0086a703          	lw	a4,8(a3)
   1a9a0:	00812c03          	lw	s8,8(sp)
   1a9a4:	00060993          	mv	s3,a2
   1a9a8:	00f72623          	sw	a5,12(a4)
   1a9ac:	00e7a423          	sw	a4,8(a5)
   1a9b0:	004aa783          	lw	a5,4(s5)
   1a9b4:	414986b3          	sub	a3,s3,s4
   1a9b8:	00f00613          	li	a2,15
   1a9bc:	0017f793          	andi	a5,a5,1
   1a9c0:	013a8733          	add	a4,s5,s3
   1a9c4:	08d66263          	bltu	a2,a3,1aa48 <_realloc_r+0x214>
   1a9c8:	0137e7b3          	or	a5,a5,s3
   1a9cc:	00faa223          	sw	a5,4(s5)
   1a9d0:	00472783          	lw	a5,4(a4)
   1a9d4:	0017e793          	ori	a5,a5,1
   1a9d8:	00f72223          	sw	a5,4(a4)
   1a9dc:	00090513          	mv	a0,s2
   1a9e0:	9e0f70ef          	jal	11bc0 <__malloc_unlock>
   1a9e4:	00040493          	mv	s1,s0
   1a9e8:	02812403          	lw	s0,40(sp)
   1a9ec:	02c12083          	lw	ra,44(sp)
   1a9f0:	02012903          	lw	s2,32(sp)
   1a9f4:	01c12983          	lw	s3,28(sp)
   1a9f8:	01812a03          	lw	s4,24(sp)
   1a9fc:	01412a83          	lw	s5,20(sp)
   1aa00:	00048513          	mv	a0,s1
   1aa04:	02412483          	lw	s1,36(sp)
   1aa08:	03010113          	addi	sp,sp,48
   1aa0c:	00008067          	ret
   1aa10:	00177713          	andi	a4,a4,1
   1aa14:	ee071ae3          	bnez	a4,1a908 <_realloc_r+0xd4>
   1aa18:	01712623          	sw	s7,12(sp)
   1aa1c:	ff842b83          	lw	s7,-8(s0)
   1aa20:	01612823          	sw	s6,16(sp)
   1aa24:	417a8bb3          	sub	s7,s5,s7
   1aa28:	004ba703          	lw	a4,4(s7)
   1aa2c:	ffc77713          	andi	a4,a4,-4
   1aa30:	ec9ff06f          	j	1a8f8 <_realloc_r+0xc4>
   1aa34:	02c12083          	lw	ra,44(sp)
   1aa38:	02412483          	lw	s1,36(sp)
   1aa3c:	00060593          	mv	a1,a2
   1aa40:	03010113          	addi	sp,sp,48
   1aa44:	9b1f606f          	j	113f4 <_malloc_r>
   1aa48:	0147e7b3          	or	a5,a5,s4
   1aa4c:	00faa223          	sw	a5,4(s5)
   1aa50:	014a85b3          	add	a1,s5,s4
   1aa54:	0016e693          	ori	a3,a3,1
   1aa58:	00d5a223          	sw	a3,4(a1)
   1aa5c:	00472783          	lw	a5,4(a4)
   1aa60:	00858593          	addi	a1,a1,8
   1aa64:	00090513          	mv	a0,s2
   1aa68:	0017e793          	ori	a5,a5,1
   1aa6c:	00f72223          	sw	a5,4(a4)
   1aa70:	e80f60ef          	jal	110f0 <_free_r>
   1aa74:	f69ff06f          	j	1a9dc <_realloc_r+0x1a8>
   1aa78:	ffc7f793          	andi	a5,a5,-4
   1aa7c:	013786b3          	add	a3,a5,s3
   1aa80:	010a0613          	addi	a2,s4,16
   1aa84:	26c6d063          	bge	a3,a2,1ace4 <_realloc_r+0x4b0>
   1aa88:	00177713          	andi	a4,a4,1
   1aa8c:	e6071ee3          	bnez	a4,1a908 <_realloc_r+0xd4>
   1aa90:	01712623          	sw	s7,12(sp)
   1aa94:	ff842b83          	lw	s7,-8(s0)
   1aa98:	01612823          	sw	s6,16(sp)
   1aa9c:	417a8bb3          	sub	s7,s5,s7
   1aaa0:	004ba703          	lw	a4,4(s7)
   1aaa4:	ffc77713          	andi	a4,a4,-4
   1aaa8:	00e787b3          	add	a5,a5,a4
   1aaac:	01378b33          	add	s6,a5,s3
   1aab0:	e4cb44e3          	blt	s6,a2,1a8f8 <_realloc_r+0xc4>
   1aab4:	00cba783          	lw	a5,12(s7)
   1aab8:	008ba703          	lw	a4,8(s7)
   1aabc:	ffc98613          	addi	a2,s3,-4
   1aac0:	02400693          	li	a3,36
   1aac4:	00f72623          	sw	a5,12(a4)
   1aac8:	00e7a423          	sw	a4,8(a5)
   1aacc:	008b8493          	addi	s1,s7,8
   1aad0:	28c6e463          	bltu	a3,a2,1ad58 <_realloc_r+0x524>
   1aad4:	01300713          	li	a4,19
   1aad8:	00048793          	mv	a5,s1
   1aadc:	02c77263          	bgeu	a4,a2,1ab00 <_realloc_r+0x2cc>
   1aae0:	00042703          	lw	a4,0(s0)
   1aae4:	01b00793          	li	a5,27
   1aae8:	00eba423          	sw	a4,8(s7)
   1aaec:	00442703          	lw	a4,4(s0)
   1aaf0:	00eba623          	sw	a4,12(s7)
   1aaf4:	26c7ea63          	bltu	a5,a2,1ad68 <_realloc_r+0x534>
   1aaf8:	00840413          	addi	s0,s0,8
   1aafc:	010b8793          	addi	a5,s7,16
   1ab00:	00042703          	lw	a4,0(s0)
   1ab04:	00e7a023          	sw	a4,0(a5)
   1ab08:	00442703          	lw	a4,4(s0)
   1ab0c:	00e7a223          	sw	a4,4(a5)
   1ab10:	00842703          	lw	a4,8(s0)
   1ab14:	00e7a423          	sw	a4,8(a5)
   1ab18:	014b8733          	add	a4,s7,s4
   1ab1c:	414b07b3          	sub	a5,s6,s4
   1ab20:	00ec2423          	sw	a4,8(s8)
   1ab24:	0017e793          	ori	a5,a5,1
   1ab28:	00f72223          	sw	a5,4(a4)
   1ab2c:	004ba783          	lw	a5,4(s7)
   1ab30:	00090513          	mv	a0,s2
   1ab34:	0017f793          	andi	a5,a5,1
   1ab38:	0147e7b3          	or	a5,a5,s4
   1ab3c:	00fba223          	sw	a5,4(s7)
   1ab40:	880f70ef          	jal	11bc0 <__malloc_unlock>
   1ab44:	01012b03          	lw	s6,16(sp)
   1ab48:	00c12b83          	lw	s7,12(sp)
   1ab4c:	00812c03          	lw	s8,8(sp)
   1ab50:	e99ff06f          	j	1a9e8 <_realloc_r+0x1b4>
   1ab54:	00042683          	lw	a3,0(s0)
   1ab58:	01b00713          	li	a4,27
   1ab5c:	00d52023          	sw	a3,0(a0)
   1ab60:	00442683          	lw	a3,4(s0)
   1ab64:	00d52223          	sw	a3,4(a0)
   1ab68:	14c76e63          	bltu	a4,a2,1acc4 <_realloc_r+0x490>
   1ab6c:	00840713          	addi	a4,s0,8
   1ab70:	00850793          	addi	a5,a0,8
   1ab74:	dd9ff06f          	j	1a94c <_realloc_r+0x118>
   1ab78:	ffc52783          	lw	a5,-4(a0)
   1ab7c:	00812c03          	lw	s8,8(sp)
   1ab80:	ffc7f793          	andi	a5,a5,-4
   1ab84:	00f989b3          	add	s3,s3,a5
   1ab88:	e29ff06f          	j	1a9b0 <_realloc_r+0x17c>
   1ab8c:	00cba783          	lw	a5,12(s7)
   1ab90:	008ba703          	lw	a4,8(s7)
   1ab94:	ffc98613          	addi	a2,s3,-4
   1ab98:	02400693          	li	a3,36
   1ab9c:	00f72623          	sw	a5,12(a4)
   1aba0:	00e7a423          	sw	a4,8(a5)
   1aba4:	008b8493          	addi	s1,s7,8
   1aba8:	10c6e663          	bltu	a3,a2,1acb4 <_realloc_r+0x480>
   1abac:	01300713          	li	a4,19
   1abb0:	00048793          	mv	a5,s1
   1abb4:	02c77c63          	bgeu	a4,a2,1abec <_realloc_r+0x3b8>
   1abb8:	00042703          	lw	a4,0(s0)
   1abbc:	01b00793          	li	a5,27
   1abc0:	00eba423          	sw	a4,8(s7)
   1abc4:	00442703          	lw	a4,4(s0)
   1abc8:	00eba623          	sw	a4,12(s7)
   1abcc:	14c7f863          	bgeu	a5,a2,1ad1c <_realloc_r+0x4e8>
   1abd0:	00842783          	lw	a5,8(s0)
   1abd4:	00fba823          	sw	a5,16(s7)
   1abd8:	00c42783          	lw	a5,12(s0)
   1abdc:	00fbaa23          	sw	a5,20(s7)
   1abe0:	0ad60c63          	beq	a2,a3,1ac98 <_realloc_r+0x464>
   1abe4:	01040413          	addi	s0,s0,16
   1abe8:	018b8793          	addi	a5,s7,24
   1abec:	00042703          	lw	a4,0(s0)
   1abf0:	00e7a023          	sw	a4,0(a5)
   1abf4:	00442703          	lw	a4,4(s0)
   1abf8:	00e7a223          	sw	a4,4(a5)
   1abfc:	00842703          	lw	a4,8(s0)
   1ac00:	00e7a423          	sw	a4,8(a5)
   1ac04:	000b0993          	mv	s3,s6
   1ac08:	000b8a93          	mv	s5,s7
   1ac0c:	01012b03          	lw	s6,16(sp)
   1ac10:	00c12b83          	lw	s7,12(sp)
   1ac14:	00812c03          	lw	s8,8(sp)
   1ac18:	00048413          	mv	s0,s1
   1ac1c:	d95ff06f          	j	1a9b0 <_realloc_r+0x17c>
   1ac20:	00040593          	mv	a1,s0
   1ac24:	e49fb0ef          	jal	16a6c <memmove>
   1ac28:	d3dff06f          	j	1a964 <_realloc_r+0x130>
   1ac2c:	00c6a783          	lw	a5,12(a3)
   1ac30:	0086a703          	lw	a4,8(a3)
   1ac34:	ffc98613          	addi	a2,s3,-4
   1ac38:	02400693          	li	a3,36
   1ac3c:	00f72623          	sw	a5,12(a4)
   1ac40:	00e7a423          	sw	a4,8(a5)
   1ac44:	008ba703          	lw	a4,8(s7)
   1ac48:	00cba783          	lw	a5,12(s7)
   1ac4c:	008b8493          	addi	s1,s7,8
   1ac50:	00f72623          	sw	a5,12(a4)
   1ac54:	00e7a423          	sw	a4,8(a5)
   1ac58:	04c6ee63          	bltu	a3,a2,1acb4 <_realloc_r+0x480>
   1ac5c:	01300713          	li	a4,19
   1ac60:	00048793          	mv	a5,s1
   1ac64:	f8c774e3          	bgeu	a4,a2,1abec <_realloc_r+0x3b8>
   1ac68:	00042703          	lw	a4,0(s0)
   1ac6c:	01b00793          	li	a5,27
   1ac70:	00eba423          	sw	a4,8(s7)
   1ac74:	00442703          	lw	a4,4(s0)
   1ac78:	00eba623          	sw	a4,12(s7)
   1ac7c:	0ac7f063          	bgeu	a5,a2,1ad1c <_realloc_r+0x4e8>
   1ac80:	00842703          	lw	a4,8(s0)
   1ac84:	02400793          	li	a5,36
   1ac88:	00eba823          	sw	a4,16(s7)
   1ac8c:	00c42703          	lw	a4,12(s0)
   1ac90:	00ebaa23          	sw	a4,20(s7)
   1ac94:	f4f618e3          	bne	a2,a5,1abe4 <_realloc_r+0x3b0>
   1ac98:	01042703          	lw	a4,16(s0)
   1ac9c:	020b8793          	addi	a5,s7,32
   1aca0:	01840413          	addi	s0,s0,24
   1aca4:	00ebac23          	sw	a4,24(s7)
   1aca8:	ffc42703          	lw	a4,-4(s0)
   1acac:	00ebae23          	sw	a4,28(s7)
   1acb0:	f3dff06f          	j	1abec <_realloc_r+0x3b8>
   1acb4:	00040593          	mv	a1,s0
   1acb8:	00048513          	mv	a0,s1
   1acbc:	db1fb0ef          	jal	16a6c <memmove>
   1acc0:	f45ff06f          	j	1ac04 <_realloc_r+0x3d0>
   1acc4:	00842703          	lw	a4,8(s0)
   1acc8:	00e52423          	sw	a4,8(a0)
   1accc:	00c42703          	lw	a4,12(s0)
   1acd0:	00e52623          	sw	a4,12(a0)
   1acd4:	06f60463          	beq	a2,a5,1ad3c <_realloc_r+0x508>
   1acd8:	01040713          	addi	a4,s0,16
   1acdc:	01050793          	addi	a5,a0,16
   1ace0:	c6dff06f          	j	1a94c <_realloc_r+0x118>
   1ace4:	014a8ab3          	add	s5,s5,s4
   1ace8:	414687b3          	sub	a5,a3,s4
   1acec:	015c2423          	sw	s5,8(s8)
   1acf0:	0017e793          	ori	a5,a5,1
   1acf4:	00faa223          	sw	a5,4(s5)
   1acf8:	ffc42783          	lw	a5,-4(s0)
   1acfc:	00090513          	mv	a0,s2
   1ad00:	00040493          	mv	s1,s0
   1ad04:	0017f793          	andi	a5,a5,1
   1ad08:	0147e7b3          	or	a5,a5,s4
   1ad0c:	fef42e23          	sw	a5,-4(s0)
   1ad10:	eb1f60ef          	jal	11bc0 <__malloc_unlock>
   1ad14:	00812c03          	lw	s8,8(sp)
   1ad18:	cd1ff06f          	j	1a9e8 <_realloc_r+0x1b4>
   1ad1c:	00840413          	addi	s0,s0,8
   1ad20:	010b8793          	addi	a5,s7,16
   1ad24:	ec9ff06f          	j	1abec <_realloc_r+0x3b8>
   1ad28:	00090513          	mv	a0,s2
   1ad2c:	e95f60ef          	jal	11bc0 <__malloc_unlock>
   1ad30:	00000493          	li	s1,0
   1ad34:	00812c03          	lw	s8,8(sp)
   1ad38:	cb1ff06f          	j	1a9e8 <_realloc_r+0x1b4>
   1ad3c:	01042683          	lw	a3,16(s0)
   1ad40:	01840713          	addi	a4,s0,24
   1ad44:	01850793          	addi	a5,a0,24
   1ad48:	00d52823          	sw	a3,16(a0)
   1ad4c:	01442683          	lw	a3,20(s0)
   1ad50:	00d52a23          	sw	a3,20(a0)
   1ad54:	bf9ff06f          	j	1a94c <_realloc_r+0x118>
   1ad58:	00040593          	mv	a1,s0
   1ad5c:	00048513          	mv	a0,s1
   1ad60:	d0dfb0ef          	jal	16a6c <memmove>
   1ad64:	db5ff06f          	j	1ab18 <_realloc_r+0x2e4>
   1ad68:	00842783          	lw	a5,8(s0)
   1ad6c:	00fba823          	sw	a5,16(s7)
   1ad70:	00c42783          	lw	a5,12(s0)
   1ad74:	00fbaa23          	sw	a5,20(s7)
   1ad78:	00d60863          	beq	a2,a3,1ad88 <_realloc_r+0x554>
   1ad7c:	01040413          	addi	s0,s0,16
   1ad80:	018b8793          	addi	a5,s7,24
   1ad84:	d7dff06f          	j	1ab00 <_realloc_r+0x2cc>
   1ad88:	01042703          	lw	a4,16(s0)
   1ad8c:	020b8793          	addi	a5,s7,32
   1ad90:	01840413          	addi	s0,s0,24
   1ad94:	00ebac23          	sw	a4,24(s7)
   1ad98:	ffc42703          	lw	a4,-4(s0)
   1ad9c:	00ebae23          	sw	a4,28(s7)
   1ada0:	d61ff06f          	j	1ab00 <_realloc_r+0x2cc>

0001ada4 <_wctomb_r>:
   1ada4:	e681a783          	lw	a5,-408(gp) # 228e8 <__global_locale+0xe0>
   1ada8:	00078067          	jr	a5

0001adac <__ascii_wctomb>:
   1adac:	02058463          	beqz	a1,1add4 <__ascii_wctomb+0x28>
   1adb0:	0ff00793          	li	a5,255
   1adb4:	00c7e863          	bltu	a5,a2,1adc4 <__ascii_wctomb+0x18>
   1adb8:	00c58023          	sb	a2,0(a1)
   1adbc:	00100513          	li	a0,1
   1adc0:	00008067          	ret
   1adc4:	08a00793          	li	a5,138
   1adc8:	00f52023          	sw	a5,0(a0)
   1adcc:	fff00513          	li	a0,-1
   1add0:	00008067          	ret
   1add4:	00000513          	li	a0,0
   1add8:	00008067          	ret

0001addc <_wcrtomb_r>:
   1addc:	fe010113          	addi	sp,sp,-32
   1ade0:	00812c23          	sw	s0,24(sp)
   1ade4:	00912a23          	sw	s1,20(sp)
   1ade8:	00112e23          	sw	ra,28(sp)
   1adec:	00050413          	mv	s0,a0
   1adf0:	00068493          	mv	s1,a3
   1adf4:	e681a783          	lw	a5,-408(gp) # 228e8 <__global_locale+0xe0>
   1adf8:	02058263          	beqz	a1,1ae1c <_wcrtomb_r+0x40>
   1adfc:	000780e7          	jalr	a5
   1ae00:	fff00793          	li	a5,-1
   1ae04:	02f50663          	beq	a0,a5,1ae30 <_wcrtomb_r+0x54>
   1ae08:	01c12083          	lw	ra,28(sp)
   1ae0c:	01812403          	lw	s0,24(sp)
   1ae10:	01412483          	lw	s1,20(sp)
   1ae14:	02010113          	addi	sp,sp,32
   1ae18:	00008067          	ret
   1ae1c:	00000613          	li	a2,0
   1ae20:	00410593          	addi	a1,sp,4
   1ae24:	000780e7          	jalr	a5
   1ae28:	fff00793          	li	a5,-1
   1ae2c:	fcf51ee3          	bne	a0,a5,1ae08 <_wcrtomb_r+0x2c>
   1ae30:	0004a023          	sw	zero,0(s1)
   1ae34:	08a00793          	li	a5,138
   1ae38:	01c12083          	lw	ra,28(sp)
   1ae3c:	00f42023          	sw	a5,0(s0)
   1ae40:	01812403          	lw	s0,24(sp)
   1ae44:	01412483          	lw	s1,20(sp)
   1ae48:	02010113          	addi	sp,sp,32
   1ae4c:	00008067          	ret

0001ae50 <wcrtomb>:
   1ae50:	fe010113          	addi	sp,sp,-32
   1ae54:	00812c23          	sw	s0,24(sp)
   1ae58:	00912a23          	sw	s1,20(sp)
   1ae5c:	00112e23          	sw	ra,28(sp)
   1ae60:	00060413          	mv	s0,a2
   1ae64:	f5c1a483          	lw	s1,-164(gp) # 229dc <_impure_ptr>
   1ae68:	e681a783          	lw	a5,-408(gp) # 228e8 <__global_locale+0xe0>
   1ae6c:	02050a63          	beqz	a0,1aea0 <wcrtomb+0x50>
   1ae70:	00058613          	mv	a2,a1
   1ae74:	00040693          	mv	a3,s0
   1ae78:	00050593          	mv	a1,a0
   1ae7c:	00048513          	mv	a0,s1
   1ae80:	000780e7          	jalr	a5
   1ae84:	fff00793          	li	a5,-1
   1ae88:	02f50a63          	beq	a0,a5,1aebc <wcrtomb+0x6c>
   1ae8c:	01c12083          	lw	ra,28(sp)
   1ae90:	01812403          	lw	s0,24(sp)
   1ae94:	01412483          	lw	s1,20(sp)
   1ae98:	02010113          	addi	sp,sp,32
   1ae9c:	00008067          	ret
   1aea0:	00060693          	mv	a3,a2
   1aea4:	00410593          	addi	a1,sp,4
   1aea8:	00000613          	li	a2,0
   1aeac:	00048513          	mv	a0,s1
   1aeb0:	000780e7          	jalr	a5
   1aeb4:	fff00793          	li	a5,-1
   1aeb8:	fcf51ae3          	bne	a0,a5,1ae8c <wcrtomb+0x3c>
   1aebc:	00042023          	sw	zero,0(s0)
   1aec0:	01c12083          	lw	ra,28(sp)
   1aec4:	01812403          	lw	s0,24(sp)
   1aec8:	08a00793          	li	a5,138
   1aecc:	00f4a023          	sw	a5,0(s1)
   1aed0:	01412483          	lw	s1,20(sp)
   1aed4:	02010113          	addi	sp,sp,32
   1aed8:	00008067          	ret

0001aedc <__smakebuf_r>:
   1aedc:	00c59783          	lh	a5,12(a1)
   1aee0:	f8010113          	addi	sp,sp,-128
   1aee4:	06812c23          	sw	s0,120(sp)
   1aee8:	06112e23          	sw	ra,124(sp)
   1aeec:	0027f713          	andi	a4,a5,2
   1aef0:	00058413          	mv	s0,a1
   1aef4:	02070463          	beqz	a4,1af1c <__smakebuf_r+0x40>
   1aef8:	04358793          	addi	a5,a1,67
   1aefc:	00f5a023          	sw	a5,0(a1)
   1af00:	00f5a823          	sw	a5,16(a1)
   1af04:	00100793          	li	a5,1
   1af08:	00f5aa23          	sw	a5,20(a1)
   1af0c:	07c12083          	lw	ra,124(sp)
   1af10:	07812403          	lw	s0,120(sp)
   1af14:	08010113          	addi	sp,sp,128
   1af18:	00008067          	ret
   1af1c:	00e59583          	lh	a1,14(a1)
   1af20:	06912a23          	sw	s1,116(sp)
   1af24:	07212823          	sw	s2,112(sp)
   1af28:	07312623          	sw	s3,108(sp)
   1af2c:	07412423          	sw	s4,104(sp)
   1af30:	00050493          	mv	s1,a0
   1af34:	0805c663          	bltz	a1,1afc0 <__smakebuf_r+0xe4>
   1af38:	00810613          	addi	a2,sp,8
   1af3c:	3b0000ef          	jal	1b2ec <_fstat_r>
   1af40:	06054e63          	bltz	a0,1afbc <__smakebuf_r+0xe0>
   1af44:	00c12783          	lw	a5,12(sp)
   1af48:	0000f937          	lui	s2,0xf
   1af4c:	000019b7          	lui	s3,0x1
   1af50:	00f97933          	and	s2,s2,a5
   1af54:	ffffe7b7          	lui	a5,0xffffe
   1af58:	00f90933          	add	s2,s2,a5
   1af5c:	00193913          	seqz	s2,s2
   1af60:	40000a13          	li	s4,1024
   1af64:	80098993          	addi	s3,s3,-2048 # 800 <exit-0xf8b4>
   1af68:	000a0593          	mv	a1,s4
   1af6c:	00048513          	mv	a0,s1
   1af70:	c84f60ef          	jal	113f4 <_malloc_r>
   1af74:	00c41783          	lh	a5,12(s0)
   1af78:	06050863          	beqz	a0,1afe8 <__smakebuf_r+0x10c>
   1af7c:	0807e793          	ori	a5,a5,128
   1af80:	00a42023          	sw	a0,0(s0)
   1af84:	00a42823          	sw	a0,16(s0)
   1af88:	00f41623          	sh	a5,12(s0)
   1af8c:	01442a23          	sw	s4,20(s0)
   1af90:	0a091063          	bnez	s2,1b030 <__smakebuf_r+0x154>
   1af94:	0137e7b3          	or	a5,a5,s3
   1af98:	07c12083          	lw	ra,124(sp)
   1af9c:	00f41623          	sh	a5,12(s0)
   1afa0:	07812403          	lw	s0,120(sp)
   1afa4:	07412483          	lw	s1,116(sp)
   1afa8:	07012903          	lw	s2,112(sp)
   1afac:	06c12983          	lw	s3,108(sp)
   1afb0:	06812a03          	lw	s4,104(sp)
   1afb4:	08010113          	addi	sp,sp,128
   1afb8:	00008067          	ret
   1afbc:	00c41783          	lh	a5,12(s0)
   1afc0:	0807f793          	andi	a5,a5,128
   1afc4:	00000913          	li	s2,0
   1afc8:	04078e63          	beqz	a5,1b024 <__smakebuf_r+0x148>
   1afcc:	04000a13          	li	s4,64
   1afd0:	000a0593          	mv	a1,s4
   1afd4:	00048513          	mv	a0,s1
   1afd8:	c1cf60ef          	jal	113f4 <_malloc_r>
   1afdc:	00c41783          	lh	a5,12(s0)
   1afe0:	00000993          	li	s3,0
   1afe4:	f8051ce3          	bnez	a0,1af7c <__smakebuf_r+0xa0>
   1afe8:	2007f713          	andi	a4,a5,512
   1afec:	04071e63          	bnez	a4,1b048 <__smakebuf_r+0x16c>
   1aff0:	ffc7f793          	andi	a5,a5,-4
   1aff4:	0027e793          	ori	a5,a5,2
   1aff8:	04340713          	addi	a4,s0,67
   1affc:	00f41623          	sh	a5,12(s0)
   1b000:	00100793          	li	a5,1
   1b004:	07412483          	lw	s1,116(sp)
   1b008:	07012903          	lw	s2,112(sp)
   1b00c:	06c12983          	lw	s3,108(sp)
   1b010:	06812a03          	lw	s4,104(sp)
   1b014:	00e42023          	sw	a4,0(s0)
   1b018:	00e42823          	sw	a4,16(s0)
   1b01c:	00f42a23          	sw	a5,20(s0)
   1b020:	eedff06f          	j	1af0c <__smakebuf_r+0x30>
   1b024:	40000a13          	li	s4,1024
   1b028:	00000993          	li	s3,0
   1b02c:	f3dff06f          	j	1af68 <__smakebuf_r+0x8c>
   1b030:	00e41583          	lh	a1,14(s0)
   1b034:	00048513          	mv	a0,s1
   1b038:	30c000ef          	jal	1b344 <_isatty_r>
   1b03c:	02051063          	bnez	a0,1b05c <__smakebuf_r+0x180>
   1b040:	00c41783          	lh	a5,12(s0)
   1b044:	f51ff06f          	j	1af94 <__smakebuf_r+0xb8>
   1b048:	07412483          	lw	s1,116(sp)
   1b04c:	07012903          	lw	s2,112(sp)
   1b050:	06c12983          	lw	s3,108(sp)
   1b054:	06812a03          	lw	s4,104(sp)
   1b058:	eb5ff06f          	j	1af0c <__smakebuf_r+0x30>
   1b05c:	00c45783          	lhu	a5,12(s0)
   1b060:	ffc7f793          	andi	a5,a5,-4
   1b064:	0017e793          	ori	a5,a5,1
   1b068:	01079793          	slli	a5,a5,0x10
   1b06c:	4107d793          	srai	a5,a5,0x10
   1b070:	f25ff06f          	j	1af94 <__smakebuf_r+0xb8>

0001b074 <__swhatbuf_r>:
   1b074:	f9010113          	addi	sp,sp,-112
   1b078:	06812423          	sw	s0,104(sp)
   1b07c:	00058413          	mv	s0,a1
   1b080:	00e59583          	lh	a1,14(a1)
   1b084:	06912223          	sw	s1,100(sp)
   1b088:	07212023          	sw	s2,96(sp)
   1b08c:	06112623          	sw	ra,108(sp)
   1b090:	00060493          	mv	s1,a2
   1b094:	00068913          	mv	s2,a3
   1b098:	0405ca63          	bltz	a1,1b0ec <__swhatbuf_r+0x78>
   1b09c:	00810613          	addi	a2,sp,8
   1b0a0:	24c000ef          	jal	1b2ec <_fstat_r>
   1b0a4:	04054463          	bltz	a0,1b0ec <__swhatbuf_r+0x78>
   1b0a8:	00c12703          	lw	a4,12(sp)
   1b0ac:	0000f7b7          	lui	a5,0xf
   1b0b0:	06c12083          	lw	ra,108(sp)
   1b0b4:	00e7f7b3          	and	a5,a5,a4
   1b0b8:	ffffe737          	lui	a4,0xffffe
   1b0bc:	00e787b3          	add	a5,a5,a4
   1b0c0:	06812403          	lw	s0,104(sp)
   1b0c4:	0017b793          	seqz	a5,a5
   1b0c8:	00f92023          	sw	a5,0(s2) # f000 <exit-0x10b4>
   1b0cc:	40000713          	li	a4,1024
   1b0d0:	00e4a023          	sw	a4,0(s1)
   1b0d4:	00001537          	lui	a0,0x1
   1b0d8:	06412483          	lw	s1,100(sp)
   1b0dc:	06012903          	lw	s2,96(sp)
   1b0e0:	80050513          	addi	a0,a0,-2048 # 800 <exit-0xf8b4>
   1b0e4:	07010113          	addi	sp,sp,112
   1b0e8:	00008067          	ret
   1b0ec:	00c45783          	lhu	a5,12(s0)
   1b0f0:	0807f793          	andi	a5,a5,128
   1b0f4:	02078863          	beqz	a5,1b124 <__swhatbuf_r+0xb0>
   1b0f8:	06c12083          	lw	ra,108(sp)
   1b0fc:	06812403          	lw	s0,104(sp)
   1b100:	00000793          	li	a5,0
   1b104:	00f92023          	sw	a5,0(s2)
   1b108:	04000713          	li	a4,64
   1b10c:	00e4a023          	sw	a4,0(s1)
   1b110:	06012903          	lw	s2,96(sp)
   1b114:	06412483          	lw	s1,100(sp)
   1b118:	00000513          	li	a0,0
   1b11c:	07010113          	addi	sp,sp,112
   1b120:	00008067          	ret
   1b124:	06c12083          	lw	ra,108(sp)
   1b128:	06812403          	lw	s0,104(sp)
   1b12c:	00f92023          	sw	a5,0(s2)
   1b130:	40000713          	li	a4,1024
   1b134:	00e4a023          	sw	a4,0(s1)
   1b138:	06012903          	lw	s2,96(sp)
   1b13c:	06412483          	lw	s1,100(sp)
   1b140:	00000513          	li	a0,0
   1b144:	07010113          	addi	sp,sp,112
   1b148:	00008067          	ret

0001b14c <__swbuf_r>:
   1b14c:	fe010113          	addi	sp,sp,-32
   1b150:	00812c23          	sw	s0,24(sp)
   1b154:	00912a23          	sw	s1,20(sp)
   1b158:	01212823          	sw	s2,16(sp)
   1b15c:	00112e23          	sw	ra,28(sp)
   1b160:	00050913          	mv	s2,a0
   1b164:	00058493          	mv	s1,a1
   1b168:	00060413          	mv	s0,a2
   1b16c:	00050663          	beqz	a0,1b178 <__swbuf_r+0x2c>
   1b170:	03452783          	lw	a5,52(a0)
   1b174:	16078063          	beqz	a5,1b2d4 <__swbuf_r+0x188>
   1b178:	01842783          	lw	a5,24(s0)
   1b17c:	00c41703          	lh	a4,12(s0)
   1b180:	00f42423          	sw	a5,8(s0)
   1b184:	00877793          	andi	a5,a4,8
   1b188:	08078063          	beqz	a5,1b208 <__swbuf_r+0xbc>
   1b18c:	01042783          	lw	a5,16(s0)
   1b190:	06078c63          	beqz	a5,1b208 <__swbuf_r+0xbc>
   1b194:	01312623          	sw	s3,12(sp)
   1b198:	01271693          	slli	a3,a4,0x12
   1b19c:	0ff4f993          	zext.b	s3,s1
   1b1a0:	0ff4f493          	zext.b	s1,s1
   1b1a4:	0806d863          	bgez	a3,1b234 <__swbuf_r+0xe8>
   1b1a8:	00042703          	lw	a4,0(s0)
   1b1ac:	01442683          	lw	a3,20(s0)
   1b1b0:	40f707b3          	sub	a5,a4,a5
   1b1b4:	0ad7d863          	bge	a5,a3,1b264 <__swbuf_r+0x118>
   1b1b8:	00842683          	lw	a3,8(s0)
   1b1bc:	00170613          	addi	a2,a4,1 # ffffe001 <__BSS_END__+0xfffdb201>
   1b1c0:	00c42023          	sw	a2,0(s0)
   1b1c4:	fff68693          	addi	a3,a3,-1
   1b1c8:	00d42423          	sw	a3,8(s0)
   1b1cc:	01370023          	sb	s3,0(a4)
   1b1d0:	01442703          	lw	a4,20(s0)
   1b1d4:	00178793          	addi	a5,a5,1 # f001 <exit-0x10b3>
   1b1d8:	0cf70263          	beq	a4,a5,1b29c <__swbuf_r+0x150>
   1b1dc:	00c45783          	lhu	a5,12(s0)
   1b1e0:	0017f793          	andi	a5,a5,1
   1b1e4:	0c079a63          	bnez	a5,1b2b8 <__swbuf_r+0x16c>
   1b1e8:	00c12983          	lw	s3,12(sp)
   1b1ec:	01c12083          	lw	ra,28(sp)
   1b1f0:	01812403          	lw	s0,24(sp)
   1b1f4:	01012903          	lw	s2,16(sp)
   1b1f8:	00048513          	mv	a0,s1
   1b1fc:	01412483          	lw	s1,20(sp)
   1b200:	02010113          	addi	sp,sp,32
   1b204:	00008067          	ret
   1b208:	00040593          	mv	a1,s0
   1b20c:	00090513          	mv	a0,s2
   1b210:	a4cfb0ef          	jal	1645c <__swsetup_r>
   1b214:	08051e63          	bnez	a0,1b2b0 <__swbuf_r+0x164>
   1b218:	00c41703          	lh	a4,12(s0)
   1b21c:	01312623          	sw	s3,12(sp)
   1b220:	01042783          	lw	a5,16(s0)
   1b224:	01271693          	slli	a3,a4,0x12
   1b228:	0ff4f993          	zext.b	s3,s1
   1b22c:	0ff4f493          	zext.b	s1,s1
   1b230:	f606cce3          	bltz	a3,1b1a8 <__swbuf_r+0x5c>
   1b234:	06442683          	lw	a3,100(s0)
   1b238:	ffffe637          	lui	a2,0xffffe
   1b23c:	000025b7          	lui	a1,0x2
   1b240:	00b76733          	or	a4,a4,a1
   1b244:	fff60613          	addi	a2,a2,-1 # ffffdfff <__BSS_END__+0xfffdb1ff>
   1b248:	00c6f6b3          	and	a3,a3,a2
   1b24c:	00e41623          	sh	a4,12(s0)
   1b250:	00042703          	lw	a4,0(s0)
   1b254:	06d42223          	sw	a3,100(s0)
   1b258:	01442683          	lw	a3,20(s0)
   1b25c:	40f707b3          	sub	a5,a4,a5
   1b260:	f4d7cce3          	blt	a5,a3,1b1b8 <__swbuf_r+0x6c>
   1b264:	00040593          	mv	a1,s0
   1b268:	00090513          	mv	a0,s2
   1b26c:	bfdfa0ef          	jal	15e68 <_fflush_r>
   1b270:	02051e63          	bnez	a0,1b2ac <__swbuf_r+0x160>
   1b274:	00042703          	lw	a4,0(s0)
   1b278:	00842683          	lw	a3,8(s0)
   1b27c:	00100793          	li	a5,1
   1b280:	00170613          	addi	a2,a4,1
   1b284:	fff68693          	addi	a3,a3,-1
   1b288:	00c42023          	sw	a2,0(s0)
   1b28c:	00d42423          	sw	a3,8(s0)
   1b290:	01370023          	sb	s3,0(a4)
   1b294:	01442703          	lw	a4,20(s0)
   1b298:	f4f712e3          	bne	a4,a5,1b1dc <__swbuf_r+0x90>
   1b29c:	00040593          	mv	a1,s0
   1b2a0:	00090513          	mv	a0,s2
   1b2a4:	bc5fa0ef          	jal	15e68 <_fflush_r>
   1b2a8:	f40500e3          	beqz	a0,1b1e8 <__swbuf_r+0x9c>
   1b2ac:	00c12983          	lw	s3,12(sp)
   1b2b0:	fff00493          	li	s1,-1
   1b2b4:	f39ff06f          	j	1b1ec <__swbuf_r+0xa0>
   1b2b8:	00a00793          	li	a5,10
   1b2bc:	f2f496e3          	bne	s1,a5,1b1e8 <__swbuf_r+0x9c>
   1b2c0:	00040593          	mv	a1,s0
   1b2c4:	00090513          	mv	a0,s2
   1b2c8:	ba1fa0ef          	jal	15e68 <_fflush_r>
   1b2cc:	f0050ee3          	beqz	a0,1b1e8 <__swbuf_r+0x9c>
   1b2d0:	fddff06f          	j	1b2ac <__swbuf_r+0x160>
   1b2d4:	b90f50ef          	jal	10664 <__sinit>
   1b2d8:	ea1ff06f          	j	1b178 <__swbuf_r+0x2c>

0001b2dc <__swbuf>:
   1b2dc:	00058613          	mv	a2,a1
   1b2e0:	00050593          	mv	a1,a0
   1b2e4:	f5c1a503          	lw	a0,-164(gp) # 229dc <_impure_ptr>
   1b2e8:	e65ff06f          	j	1b14c <__swbuf_r>

0001b2ec <_fstat_r>:
   1b2ec:	ff010113          	addi	sp,sp,-16
   1b2f0:	00058713          	mv	a4,a1
   1b2f4:	00812423          	sw	s0,8(sp)
   1b2f8:	00060593          	mv	a1,a2
   1b2fc:	00050413          	mv	s0,a0
   1b300:	00070513          	mv	a0,a4
   1b304:	f601a623          	sw	zero,-148(gp) # 229ec <errno>
   1b308:	00112623          	sw	ra,12(sp)
   1b30c:	48c050ef          	jal	20798 <_fstat>
   1b310:	fff00793          	li	a5,-1
   1b314:	00f50a63          	beq	a0,a5,1b328 <_fstat_r+0x3c>
   1b318:	00c12083          	lw	ra,12(sp)
   1b31c:	00812403          	lw	s0,8(sp)
   1b320:	01010113          	addi	sp,sp,16
   1b324:	00008067          	ret
   1b328:	f6c1a783          	lw	a5,-148(gp) # 229ec <errno>
   1b32c:	fe0786e3          	beqz	a5,1b318 <_fstat_r+0x2c>
   1b330:	00c12083          	lw	ra,12(sp)
   1b334:	00f42023          	sw	a5,0(s0)
   1b338:	00812403          	lw	s0,8(sp)
   1b33c:	01010113          	addi	sp,sp,16
   1b340:	00008067          	ret

0001b344 <_isatty_r>:
   1b344:	ff010113          	addi	sp,sp,-16
   1b348:	00812423          	sw	s0,8(sp)
   1b34c:	00050413          	mv	s0,a0
   1b350:	00058513          	mv	a0,a1
   1b354:	f601a623          	sw	zero,-148(gp) # 229ec <errno>
   1b358:	00112623          	sw	ra,12(sp)
   1b35c:	45c050ef          	jal	207b8 <_isatty>
   1b360:	fff00793          	li	a5,-1
   1b364:	00f50a63          	beq	a0,a5,1b378 <_isatty_r+0x34>
   1b368:	00c12083          	lw	ra,12(sp)
   1b36c:	00812403          	lw	s0,8(sp)
   1b370:	01010113          	addi	sp,sp,16
   1b374:	00008067          	ret
   1b378:	f6c1a783          	lw	a5,-148(gp) # 229ec <errno>
   1b37c:	fe0786e3          	beqz	a5,1b368 <_isatty_r+0x24>
   1b380:	00c12083          	lw	ra,12(sp)
   1b384:	00f42023          	sw	a5,0(s0)
   1b388:	00812403          	lw	s0,8(sp)
   1b38c:	01010113          	addi	sp,sp,16
   1b390:	00008067          	ret

0001b394 <__assert_func>:
   1b394:	ff010113          	addi	sp,sp,-16
   1b398:	00068793          	mv	a5,a3
   1b39c:	f5c1a703          	lw	a4,-164(gp) # 229dc <_impure_ptr>
   1b3a0:	00060813          	mv	a6,a2
   1b3a4:	00112623          	sw	ra,12(sp)
   1b3a8:	00c72883          	lw	a7,12(a4)
   1b3ac:	00078613          	mv	a2,a5
   1b3b0:	00050693          	mv	a3,a0
   1b3b4:	00058713          	mv	a4,a1
   1b3b8:	00005797          	auipc	a5,0x5
   1b3bc:	67478793          	addi	a5,a5,1652 # 20a2c <_exit+0x1f4>
   1b3c0:	00080c63          	beqz	a6,1b3d8 <__assert_func+0x44>
   1b3c4:	00005597          	auipc	a1,0x5
   1b3c8:	67858593          	addi	a1,a1,1656 # 20a3c <_exit+0x204>
   1b3cc:	00088513          	mv	a0,a7
   1b3d0:	14c000ef          	jal	1b51c <fiprintf>
   1b3d4:	198000ef          	jal	1b56c <abort>
   1b3d8:	00005797          	auipc	a5,0x5
   1b3dc:	66078793          	addi	a5,a5,1632 # 20a38 <_exit+0x200>
   1b3e0:	00078813          	mv	a6,a5
   1b3e4:	fe1ff06f          	j	1b3c4 <__assert_func+0x30>

0001b3e8 <__assert>:
   1b3e8:	ff010113          	addi	sp,sp,-16
   1b3ec:	00060693          	mv	a3,a2
   1b3f0:	00000613          	li	a2,0
   1b3f4:	00112623          	sw	ra,12(sp)
   1b3f8:	f9dff0ef          	jal	1b394 <__assert_func>

0001b3fc <_calloc_r>:
   1b3fc:	02c5b7b3          	mulhu	a5,a1,a2
   1b400:	ff010113          	addi	sp,sp,-16
   1b404:	00112623          	sw	ra,12(sp)
   1b408:	00812423          	sw	s0,8(sp)
   1b40c:	02c585b3          	mul	a1,a1,a2
   1b410:	0a079063          	bnez	a5,1b4b0 <_calloc_r+0xb4>
   1b414:	fe1f50ef          	jal	113f4 <_malloc_r>
   1b418:	00050413          	mv	s0,a0
   1b41c:	0a050063          	beqz	a0,1b4bc <_calloc_r+0xc0>
   1b420:	ffc52603          	lw	a2,-4(a0)
   1b424:	02400713          	li	a4,36
   1b428:	ffc67613          	andi	a2,a2,-4
   1b42c:	ffc60613          	addi	a2,a2,-4
   1b430:	04c76863          	bltu	a4,a2,1b480 <_calloc_r+0x84>
   1b434:	01300693          	li	a3,19
   1b438:	00050793          	mv	a5,a0
   1b43c:	02c6f263          	bgeu	a3,a2,1b460 <_calloc_r+0x64>
   1b440:	00052023          	sw	zero,0(a0)
   1b444:	00052223          	sw	zero,4(a0)
   1b448:	01b00793          	li	a5,27
   1b44c:	04c7f863          	bgeu	a5,a2,1b49c <_calloc_r+0xa0>
   1b450:	00052423          	sw	zero,8(a0)
   1b454:	00052623          	sw	zero,12(a0)
   1b458:	01050793          	addi	a5,a0,16
   1b45c:	06e60c63          	beq	a2,a4,1b4d4 <_calloc_r+0xd8>
   1b460:	0007a023          	sw	zero,0(a5)
   1b464:	0007a223          	sw	zero,4(a5)
   1b468:	0007a423          	sw	zero,8(a5)
   1b46c:	00c12083          	lw	ra,12(sp)
   1b470:	00040513          	mv	a0,s0
   1b474:	00812403          	lw	s0,8(sp)
   1b478:	01010113          	addi	sp,sp,16
   1b47c:	00008067          	ret
   1b480:	00000593          	li	a1,0
   1b484:	881f50ef          	jal	10d04 <memset>
   1b488:	00c12083          	lw	ra,12(sp)
   1b48c:	00040513          	mv	a0,s0
   1b490:	00812403          	lw	s0,8(sp)
   1b494:	01010113          	addi	sp,sp,16
   1b498:	00008067          	ret
   1b49c:	00850793          	addi	a5,a0,8
   1b4a0:	0007a023          	sw	zero,0(a5)
   1b4a4:	0007a223          	sw	zero,4(a5)
   1b4a8:	0007a423          	sw	zero,8(a5)
   1b4ac:	fc1ff06f          	j	1b46c <_calloc_r+0x70>
   1b4b0:	0b4000ef          	jal	1b564 <__errno>
   1b4b4:	00c00793          	li	a5,12
   1b4b8:	00f52023          	sw	a5,0(a0)
   1b4bc:	00000413          	li	s0,0
   1b4c0:	00c12083          	lw	ra,12(sp)
   1b4c4:	00040513          	mv	a0,s0
   1b4c8:	00812403          	lw	s0,8(sp)
   1b4cc:	01010113          	addi	sp,sp,16
   1b4d0:	00008067          	ret
   1b4d4:	00052823          	sw	zero,16(a0)
   1b4d8:	01850793          	addi	a5,a0,24
   1b4dc:	00052a23          	sw	zero,20(a0)
   1b4e0:	f81ff06f          	j	1b460 <_calloc_r+0x64>

0001b4e4 <_fiprintf_r>:
   1b4e4:	fc010113          	addi	sp,sp,-64
   1b4e8:	02c10313          	addi	t1,sp,44
   1b4ec:	02d12623          	sw	a3,44(sp)
   1b4f0:	00030693          	mv	a3,t1
   1b4f4:	00112e23          	sw	ra,28(sp)
   1b4f8:	02e12823          	sw	a4,48(sp)
   1b4fc:	02f12a23          	sw	a5,52(sp)
   1b500:	03012c23          	sw	a6,56(sp)
   1b504:	03112e23          	sw	a7,60(sp)
   1b508:	00612623          	sw	t1,12(sp)
   1b50c:	9bcf90ef          	jal	146c8 <_vfiprintf_r>
   1b510:	01c12083          	lw	ra,28(sp)
   1b514:	04010113          	addi	sp,sp,64
   1b518:	00008067          	ret

0001b51c <fiprintf>:
   1b51c:	fc010113          	addi	sp,sp,-64
   1b520:	02810313          	addi	t1,sp,40
   1b524:	02c12423          	sw	a2,40(sp)
   1b528:	02d12623          	sw	a3,44(sp)
   1b52c:	00058613          	mv	a2,a1
   1b530:	00030693          	mv	a3,t1
   1b534:	00050593          	mv	a1,a0
   1b538:	f5c1a503          	lw	a0,-164(gp) # 229dc <_impure_ptr>
   1b53c:	00112e23          	sw	ra,28(sp)
   1b540:	02e12823          	sw	a4,48(sp)
   1b544:	02f12a23          	sw	a5,52(sp)
   1b548:	03012c23          	sw	a6,56(sp)
   1b54c:	03112e23          	sw	a7,60(sp)
   1b550:	00612623          	sw	t1,12(sp)
   1b554:	974f90ef          	jal	146c8 <_vfiprintf_r>
   1b558:	01c12083          	lw	ra,28(sp)
   1b55c:	04010113          	addi	sp,sp,64
   1b560:	00008067          	ret

0001b564 <__errno>:
   1b564:	f5c1a503          	lw	a0,-164(gp) # 229dc <_impure_ptr>
   1b568:	00008067          	ret

0001b56c <abort>:
   1b56c:	ff010113          	addi	sp,sp,-16
   1b570:	00600513          	li	a0,6
   1b574:	00112623          	sw	ra,12(sp)
   1b578:	2a8000ef          	jal	1b820 <raise>
   1b57c:	00100513          	li	a0,1
   1b580:	2b8050ef          	jal	20838 <_exit>

0001b584 <_init_signal_r>:
   1b584:	11852783          	lw	a5,280(a0)
   1b588:	00078663          	beqz	a5,1b594 <_init_signal_r+0x10>
   1b58c:	00000513          	li	a0,0
   1b590:	00008067          	ret
   1b594:	ff010113          	addi	sp,sp,-16
   1b598:	08000593          	li	a1,128
   1b59c:	00812423          	sw	s0,8(sp)
   1b5a0:	00112623          	sw	ra,12(sp)
   1b5a4:	00050413          	mv	s0,a0
   1b5a8:	e4df50ef          	jal	113f4 <_malloc_r>
   1b5ac:	10a42c23          	sw	a0,280(s0)
   1b5b0:	02050463          	beqz	a0,1b5d8 <_init_signal_r+0x54>
   1b5b4:	08050793          	addi	a5,a0,128
   1b5b8:	00052023          	sw	zero,0(a0)
   1b5bc:	00450513          	addi	a0,a0,4
   1b5c0:	fef51ce3          	bne	a0,a5,1b5b8 <_init_signal_r+0x34>
   1b5c4:	00000513          	li	a0,0
   1b5c8:	00c12083          	lw	ra,12(sp)
   1b5cc:	00812403          	lw	s0,8(sp)
   1b5d0:	01010113          	addi	sp,sp,16
   1b5d4:	00008067          	ret
   1b5d8:	fff00513          	li	a0,-1
   1b5dc:	fedff06f          	j	1b5c8 <_init_signal_r+0x44>

0001b5e0 <_signal_r>:
   1b5e0:	fe010113          	addi	sp,sp,-32
   1b5e4:	00912a23          	sw	s1,20(sp)
   1b5e8:	00112e23          	sw	ra,28(sp)
   1b5ec:	01f00793          	li	a5,31
   1b5f0:	00050493          	mv	s1,a0
   1b5f4:	02b7ec63          	bltu	a5,a1,1b62c <_signal_r+0x4c>
   1b5f8:	11852783          	lw	a5,280(a0)
   1b5fc:	00812c23          	sw	s0,24(sp)
   1b600:	00058413          	mv	s0,a1
   1b604:	02078c63          	beqz	a5,1b63c <_signal_r+0x5c>
   1b608:	00241413          	slli	s0,s0,0x2
   1b60c:	008787b3          	add	a5,a5,s0
   1b610:	01812403          	lw	s0,24(sp)
   1b614:	0007a503          	lw	a0,0(a5)
   1b618:	00c7a023          	sw	a2,0(a5)
   1b61c:	01c12083          	lw	ra,28(sp)
   1b620:	01412483          	lw	s1,20(sp)
   1b624:	02010113          	addi	sp,sp,32
   1b628:	00008067          	ret
   1b62c:	01600793          	li	a5,22
   1b630:	00f52023          	sw	a5,0(a0)
   1b634:	fff00513          	li	a0,-1
   1b638:	fe5ff06f          	j	1b61c <_signal_r+0x3c>
   1b63c:	08000593          	li	a1,128
   1b640:	00c12623          	sw	a2,12(sp)
   1b644:	db1f50ef          	jal	113f4 <_malloc_r>
   1b648:	10a4ac23          	sw	a0,280(s1)
   1b64c:	00c12603          	lw	a2,12(sp)
   1b650:	00050793          	mv	a5,a0
   1b654:	00050713          	mv	a4,a0
   1b658:	08050693          	addi	a3,a0,128
   1b65c:	00050a63          	beqz	a0,1b670 <_signal_r+0x90>
   1b660:	00072023          	sw	zero,0(a4)
   1b664:	00470713          	addi	a4,a4,4
   1b668:	fed71ce3          	bne	a4,a3,1b660 <_signal_r+0x80>
   1b66c:	f9dff06f          	j	1b608 <_signal_r+0x28>
   1b670:	01812403          	lw	s0,24(sp)
   1b674:	fff00513          	li	a0,-1
   1b678:	fa5ff06f          	j	1b61c <_signal_r+0x3c>

0001b67c <_raise_r>:
   1b67c:	ff010113          	addi	sp,sp,-16
   1b680:	00912223          	sw	s1,4(sp)
   1b684:	00112623          	sw	ra,12(sp)
   1b688:	01f00793          	li	a5,31
   1b68c:	00050493          	mv	s1,a0
   1b690:	0ab7e063          	bltu	a5,a1,1b730 <_raise_r+0xb4>
   1b694:	11852783          	lw	a5,280(a0)
   1b698:	00812423          	sw	s0,8(sp)
   1b69c:	00058413          	mv	s0,a1
   1b6a0:	04078463          	beqz	a5,1b6e8 <_raise_r+0x6c>
   1b6a4:	00259713          	slli	a4,a1,0x2
   1b6a8:	00e787b3          	add	a5,a5,a4
   1b6ac:	0007a703          	lw	a4,0(a5)
   1b6b0:	02070c63          	beqz	a4,1b6e8 <_raise_r+0x6c>
   1b6b4:	00100693          	li	a3,1
   1b6b8:	00d70c63          	beq	a4,a3,1b6d0 <_raise_r+0x54>
   1b6bc:	fff00693          	li	a3,-1
   1b6c0:	04d70863          	beq	a4,a3,1b710 <_raise_r+0x94>
   1b6c4:	0007a023          	sw	zero,0(a5)
   1b6c8:	00058513          	mv	a0,a1
   1b6cc:	000700e7          	jalr	a4
   1b6d0:	00812403          	lw	s0,8(sp)
   1b6d4:	00000513          	li	a0,0
   1b6d8:	00c12083          	lw	ra,12(sp)
   1b6dc:	00412483          	lw	s1,4(sp)
   1b6e0:	01010113          	addi	sp,sp,16
   1b6e4:	00008067          	ret
   1b6e8:	00048513          	mv	a0,s1
   1b6ec:	430000ef          	jal	1bb1c <_getpid_r>
   1b6f0:	00040613          	mv	a2,s0
   1b6f4:	00812403          	lw	s0,8(sp)
   1b6f8:	00c12083          	lw	ra,12(sp)
   1b6fc:	00050593          	mv	a1,a0
   1b700:	00048513          	mv	a0,s1
   1b704:	00412483          	lw	s1,4(sp)
   1b708:	01010113          	addi	sp,sp,16
   1b70c:	3b80006f          	j	1bac4 <_kill_r>
   1b710:	00812403          	lw	s0,8(sp)
   1b714:	00c12083          	lw	ra,12(sp)
   1b718:	01600793          	li	a5,22
   1b71c:	00f52023          	sw	a5,0(a0)
   1b720:	00412483          	lw	s1,4(sp)
   1b724:	00100513          	li	a0,1
   1b728:	01010113          	addi	sp,sp,16
   1b72c:	00008067          	ret
   1b730:	01600793          	li	a5,22
   1b734:	00f52023          	sw	a5,0(a0)
   1b738:	fff00513          	li	a0,-1
   1b73c:	f9dff06f          	j	1b6d8 <_raise_r+0x5c>

0001b740 <__sigtramp_r>:
   1b740:	01f00793          	li	a5,31
   1b744:	0cb7ea63          	bltu	a5,a1,1b818 <__sigtramp_r+0xd8>
   1b748:	11852783          	lw	a5,280(a0)
   1b74c:	ff010113          	addi	sp,sp,-16
   1b750:	00812423          	sw	s0,8(sp)
   1b754:	00912223          	sw	s1,4(sp)
   1b758:	00112623          	sw	ra,12(sp)
   1b75c:	00058413          	mv	s0,a1
   1b760:	00050493          	mv	s1,a0
   1b764:	08078063          	beqz	a5,1b7e4 <__sigtramp_r+0xa4>
   1b768:	00241713          	slli	a4,s0,0x2
   1b76c:	00e787b3          	add	a5,a5,a4
   1b770:	0007a703          	lw	a4,0(a5)
   1b774:	02070c63          	beqz	a4,1b7ac <__sigtramp_r+0x6c>
   1b778:	fff00693          	li	a3,-1
   1b77c:	06d70063          	beq	a4,a3,1b7dc <__sigtramp_r+0x9c>
   1b780:	00100693          	li	a3,1
   1b784:	04d70063          	beq	a4,a3,1b7c4 <__sigtramp_r+0x84>
   1b788:	00040513          	mv	a0,s0
   1b78c:	0007a023          	sw	zero,0(a5)
   1b790:	000700e7          	jalr	a4
   1b794:	00000513          	li	a0,0
   1b798:	00c12083          	lw	ra,12(sp)
   1b79c:	00812403          	lw	s0,8(sp)
   1b7a0:	00412483          	lw	s1,4(sp)
   1b7a4:	01010113          	addi	sp,sp,16
   1b7a8:	00008067          	ret
   1b7ac:	00c12083          	lw	ra,12(sp)
   1b7b0:	00812403          	lw	s0,8(sp)
   1b7b4:	00412483          	lw	s1,4(sp)
   1b7b8:	00100513          	li	a0,1
   1b7bc:	01010113          	addi	sp,sp,16
   1b7c0:	00008067          	ret
   1b7c4:	00c12083          	lw	ra,12(sp)
   1b7c8:	00812403          	lw	s0,8(sp)
   1b7cc:	00412483          	lw	s1,4(sp)
   1b7d0:	00300513          	li	a0,3
   1b7d4:	01010113          	addi	sp,sp,16
   1b7d8:	00008067          	ret
   1b7dc:	00200513          	li	a0,2
   1b7e0:	fb9ff06f          	j	1b798 <__sigtramp_r+0x58>
   1b7e4:	08000593          	li	a1,128
   1b7e8:	c0df50ef          	jal	113f4 <_malloc_r>
   1b7ec:	10a4ac23          	sw	a0,280(s1)
   1b7f0:	00050793          	mv	a5,a0
   1b7f4:	00050e63          	beqz	a0,1b810 <__sigtramp_r+0xd0>
   1b7f8:	00050713          	mv	a4,a0
   1b7fc:	08050693          	addi	a3,a0,128
   1b800:	00072023          	sw	zero,0(a4)
   1b804:	00470713          	addi	a4,a4,4
   1b808:	fed71ce3          	bne	a4,a3,1b800 <__sigtramp_r+0xc0>
   1b80c:	f5dff06f          	j	1b768 <__sigtramp_r+0x28>
   1b810:	fff00513          	li	a0,-1
   1b814:	f85ff06f          	j	1b798 <__sigtramp_r+0x58>
   1b818:	fff00513          	li	a0,-1
   1b81c:	00008067          	ret

0001b820 <raise>:
   1b820:	ff010113          	addi	sp,sp,-16
   1b824:	00912223          	sw	s1,4(sp)
   1b828:	00112623          	sw	ra,12(sp)
   1b82c:	01f00793          	li	a5,31
   1b830:	f5c1a483          	lw	s1,-164(gp) # 229dc <_impure_ptr>
   1b834:	08a7ee63          	bltu	a5,a0,1b8d0 <raise+0xb0>
   1b838:	1184a783          	lw	a5,280(s1)
   1b83c:	00812423          	sw	s0,8(sp)
   1b840:	00050413          	mv	s0,a0
   1b844:	04078263          	beqz	a5,1b888 <raise+0x68>
   1b848:	00251713          	slli	a4,a0,0x2
   1b84c:	00e787b3          	add	a5,a5,a4
   1b850:	0007a703          	lw	a4,0(a5)
   1b854:	02070a63          	beqz	a4,1b888 <raise+0x68>
   1b858:	00100693          	li	a3,1
   1b85c:	00d70a63          	beq	a4,a3,1b870 <raise+0x50>
   1b860:	fff00693          	li	a3,-1
   1b864:	04d70663          	beq	a4,a3,1b8b0 <raise+0x90>
   1b868:	0007a023          	sw	zero,0(a5)
   1b86c:	000700e7          	jalr	a4
   1b870:	00812403          	lw	s0,8(sp)
   1b874:	00000513          	li	a0,0
   1b878:	00c12083          	lw	ra,12(sp)
   1b87c:	00412483          	lw	s1,4(sp)
   1b880:	01010113          	addi	sp,sp,16
   1b884:	00008067          	ret
   1b888:	00048513          	mv	a0,s1
   1b88c:	290000ef          	jal	1bb1c <_getpid_r>
   1b890:	00040613          	mv	a2,s0
   1b894:	00812403          	lw	s0,8(sp)
   1b898:	00c12083          	lw	ra,12(sp)
   1b89c:	00050593          	mv	a1,a0
   1b8a0:	00048513          	mv	a0,s1
   1b8a4:	00412483          	lw	s1,4(sp)
   1b8a8:	01010113          	addi	sp,sp,16
   1b8ac:	2180006f          	j	1bac4 <_kill_r>
   1b8b0:	00812403          	lw	s0,8(sp)
   1b8b4:	00c12083          	lw	ra,12(sp)
   1b8b8:	01600793          	li	a5,22
   1b8bc:	00f4a023          	sw	a5,0(s1)
   1b8c0:	00100513          	li	a0,1
   1b8c4:	00412483          	lw	s1,4(sp)
   1b8c8:	01010113          	addi	sp,sp,16
   1b8cc:	00008067          	ret
   1b8d0:	01600793          	li	a5,22
   1b8d4:	00f4a023          	sw	a5,0(s1)
   1b8d8:	fff00513          	li	a0,-1
   1b8dc:	f9dff06f          	j	1b878 <raise+0x58>

0001b8e0 <signal>:
   1b8e0:	ff010113          	addi	sp,sp,-16
   1b8e4:	01212023          	sw	s2,0(sp)
   1b8e8:	00112623          	sw	ra,12(sp)
   1b8ec:	01f00793          	li	a5,31
   1b8f0:	f5c1a903          	lw	s2,-164(gp) # 229dc <_impure_ptr>
   1b8f4:	04a7e263          	bltu	a5,a0,1b938 <signal+0x58>
   1b8f8:	00812423          	sw	s0,8(sp)
   1b8fc:	00050413          	mv	s0,a0
   1b900:	11892503          	lw	a0,280(s2)
   1b904:	00912223          	sw	s1,4(sp)
   1b908:	00058493          	mv	s1,a1
   1b90c:	02050e63          	beqz	a0,1b948 <signal+0x68>
   1b910:	00241413          	slli	s0,s0,0x2
   1b914:	008507b3          	add	a5,a0,s0
   1b918:	0007a503          	lw	a0,0(a5)
   1b91c:	00812403          	lw	s0,8(sp)
   1b920:	0097a023          	sw	s1,0(a5)
   1b924:	00412483          	lw	s1,4(sp)
   1b928:	00c12083          	lw	ra,12(sp)
   1b92c:	00012903          	lw	s2,0(sp)
   1b930:	01010113          	addi	sp,sp,16
   1b934:	00008067          	ret
   1b938:	01600793          	li	a5,22
   1b93c:	00f92023          	sw	a5,0(s2)
   1b940:	fff00513          	li	a0,-1
   1b944:	fe5ff06f          	j	1b928 <signal+0x48>
   1b948:	08000593          	li	a1,128
   1b94c:	00090513          	mv	a0,s2
   1b950:	aa5f50ef          	jal	113f4 <_malloc_r>
   1b954:	10a92c23          	sw	a0,280(s2)
   1b958:	00050793          	mv	a5,a0
   1b95c:	08050713          	addi	a4,a0,128
   1b960:	00050a63          	beqz	a0,1b974 <signal+0x94>
   1b964:	0007a023          	sw	zero,0(a5)
   1b968:	00478793          	addi	a5,a5,4
   1b96c:	fef71ce3          	bne	a4,a5,1b964 <signal+0x84>
   1b970:	fa1ff06f          	j	1b910 <signal+0x30>
   1b974:	00812403          	lw	s0,8(sp)
   1b978:	00412483          	lw	s1,4(sp)
   1b97c:	fff00513          	li	a0,-1
   1b980:	fa9ff06f          	j	1b928 <signal+0x48>

0001b984 <_init_signal>:
   1b984:	ff010113          	addi	sp,sp,-16
   1b988:	00812423          	sw	s0,8(sp)
   1b98c:	f5c1a403          	lw	s0,-164(gp) # 229dc <_impure_ptr>
   1b990:	11842783          	lw	a5,280(s0)
   1b994:	00112623          	sw	ra,12(sp)
   1b998:	00078c63          	beqz	a5,1b9b0 <_init_signal+0x2c>
   1b99c:	00000513          	li	a0,0
   1b9a0:	00c12083          	lw	ra,12(sp)
   1b9a4:	00812403          	lw	s0,8(sp)
   1b9a8:	01010113          	addi	sp,sp,16
   1b9ac:	00008067          	ret
   1b9b0:	08000593          	li	a1,128
   1b9b4:	00040513          	mv	a0,s0
   1b9b8:	a3df50ef          	jal	113f4 <_malloc_r>
   1b9bc:	10a42c23          	sw	a0,280(s0)
   1b9c0:	00050c63          	beqz	a0,1b9d8 <_init_signal+0x54>
   1b9c4:	08050793          	addi	a5,a0,128
   1b9c8:	00052023          	sw	zero,0(a0)
   1b9cc:	00450513          	addi	a0,a0,4
   1b9d0:	fef51ce3          	bne	a0,a5,1b9c8 <_init_signal+0x44>
   1b9d4:	fc9ff06f          	j	1b99c <_init_signal+0x18>
   1b9d8:	fff00513          	li	a0,-1
   1b9dc:	fc5ff06f          	j	1b9a0 <_init_signal+0x1c>

0001b9e0 <__sigtramp>:
   1b9e0:	ff010113          	addi	sp,sp,-16
   1b9e4:	00912223          	sw	s1,4(sp)
   1b9e8:	00112623          	sw	ra,12(sp)
   1b9ec:	01f00793          	li	a5,31
   1b9f0:	f5c1a483          	lw	s1,-164(gp) # 229dc <_impure_ptr>
   1b9f4:	0ca7e463          	bltu	a5,a0,1babc <__sigtramp+0xdc>
   1b9f8:	1184a783          	lw	a5,280(s1)
   1b9fc:	00812423          	sw	s0,8(sp)
   1ba00:	00050413          	mv	s0,a0
   1ba04:	08078263          	beqz	a5,1ba88 <__sigtramp+0xa8>
   1ba08:	00241713          	slli	a4,s0,0x2
   1ba0c:	00e787b3          	add	a5,a5,a4
   1ba10:	0007a703          	lw	a4,0(a5)
   1ba14:	02070c63          	beqz	a4,1ba4c <__sigtramp+0x6c>
   1ba18:	fff00693          	li	a3,-1
   1ba1c:	06d70063          	beq	a4,a3,1ba7c <__sigtramp+0x9c>
   1ba20:	00100693          	li	a3,1
   1ba24:	04d70063          	beq	a4,a3,1ba64 <__sigtramp+0x84>
   1ba28:	00040513          	mv	a0,s0
   1ba2c:	0007a023          	sw	zero,0(a5)
   1ba30:	000700e7          	jalr	a4
   1ba34:	00812403          	lw	s0,8(sp)
   1ba38:	00000513          	li	a0,0
   1ba3c:	00c12083          	lw	ra,12(sp)
   1ba40:	00412483          	lw	s1,4(sp)
   1ba44:	01010113          	addi	sp,sp,16
   1ba48:	00008067          	ret
   1ba4c:	00812403          	lw	s0,8(sp)
   1ba50:	00c12083          	lw	ra,12(sp)
   1ba54:	00412483          	lw	s1,4(sp)
   1ba58:	00100513          	li	a0,1
   1ba5c:	01010113          	addi	sp,sp,16
   1ba60:	00008067          	ret
   1ba64:	00812403          	lw	s0,8(sp)
   1ba68:	00c12083          	lw	ra,12(sp)
   1ba6c:	00412483          	lw	s1,4(sp)
   1ba70:	00300513          	li	a0,3
   1ba74:	01010113          	addi	sp,sp,16
   1ba78:	00008067          	ret
   1ba7c:	00812403          	lw	s0,8(sp)
   1ba80:	00200513          	li	a0,2
   1ba84:	fb9ff06f          	j	1ba3c <__sigtramp+0x5c>
   1ba88:	08000593          	li	a1,128
   1ba8c:	00048513          	mv	a0,s1
   1ba90:	965f50ef          	jal	113f4 <_malloc_r>
   1ba94:	10a4ac23          	sw	a0,280(s1)
   1ba98:	00050793          	mv	a5,a0
   1ba9c:	00050e63          	beqz	a0,1bab8 <__sigtramp+0xd8>
   1baa0:	00050713          	mv	a4,a0
   1baa4:	08050693          	addi	a3,a0,128
   1baa8:	00072023          	sw	zero,0(a4)
   1baac:	00470713          	addi	a4,a4,4
   1bab0:	fee69ce3          	bne	a3,a4,1baa8 <__sigtramp+0xc8>
   1bab4:	f55ff06f          	j	1ba08 <__sigtramp+0x28>
   1bab8:	00812403          	lw	s0,8(sp)
   1babc:	fff00513          	li	a0,-1
   1bac0:	f7dff06f          	j	1ba3c <__sigtramp+0x5c>

0001bac4 <_kill_r>:
   1bac4:	ff010113          	addi	sp,sp,-16
   1bac8:	00058713          	mv	a4,a1
   1bacc:	00812423          	sw	s0,8(sp)
   1bad0:	00060593          	mv	a1,a2
   1bad4:	00050413          	mv	s0,a0
   1bad8:	00070513          	mv	a0,a4
   1badc:	f601a623          	sw	zero,-148(gp) # 229ec <errno>
   1bae0:	00112623          	sw	ra,12(sp)
   1bae4:	4e5040ef          	jal	207c8 <_kill>
   1bae8:	fff00793          	li	a5,-1
   1baec:	00f50a63          	beq	a0,a5,1bb00 <_kill_r+0x3c>
   1baf0:	00c12083          	lw	ra,12(sp)
   1baf4:	00812403          	lw	s0,8(sp)
   1baf8:	01010113          	addi	sp,sp,16
   1bafc:	00008067          	ret
   1bb00:	f6c1a783          	lw	a5,-148(gp) # 229ec <errno>
   1bb04:	fe0786e3          	beqz	a5,1baf0 <_kill_r+0x2c>
   1bb08:	00c12083          	lw	ra,12(sp)
   1bb0c:	00f42023          	sw	a5,0(s0)
   1bb10:	00812403          	lw	s0,8(sp)
   1bb14:	01010113          	addi	sp,sp,16
   1bb18:	00008067          	ret

0001bb1c <_getpid_r>:
   1bb1c:	48d0406f          	j	207a8 <_getpid>

0001bb20 <__adddf3>:
   1bb20:	00100837          	lui	a6,0x100
   1bb24:	fff80813          	addi	a6,a6,-1 # fffff <__BSS_END__+0xdd1ff>
   1bb28:	fe010113          	addi	sp,sp,-32
   1bb2c:	00b878b3          	and	a7,a6,a1
   1bb30:	0145d713          	srli	a4,a1,0x14
   1bb34:	01d55793          	srli	a5,a0,0x1d
   1bb38:	00d87833          	and	a6,a6,a3
   1bb3c:	00912a23          	sw	s1,20(sp)
   1bb40:	7ff77493          	andi	s1,a4,2047
   1bb44:	00389713          	slli	a4,a7,0x3
   1bb48:	0146d893          	srli	a7,a3,0x14
   1bb4c:	00381813          	slli	a6,a6,0x3
   1bb50:	01212823          	sw	s2,16(sp)
   1bb54:	00e7e7b3          	or	a5,a5,a4
   1bb58:	7ff8f893          	andi	a7,a7,2047
   1bb5c:	01d65713          	srli	a4,a2,0x1d
   1bb60:	00112e23          	sw	ra,28(sp)
   1bb64:	00812c23          	sw	s0,24(sp)
   1bb68:	01312623          	sw	s3,12(sp)
   1bb6c:	01f5d913          	srli	s2,a1,0x1f
   1bb70:	01f6d693          	srli	a3,a3,0x1f
   1bb74:	01076733          	or	a4,a4,a6
   1bb78:	00351513          	slli	a0,a0,0x3
   1bb7c:	00361613          	slli	a2,a2,0x3
   1bb80:	41148833          	sub	a6,s1,a7
   1bb84:	2ad91a63          	bne	s2,a3,1be38 <__adddf3+0x318>
   1bb88:	11005c63          	blez	a6,1bca0 <__adddf3+0x180>
   1bb8c:	04089063          	bnez	a7,1bbcc <__adddf3+0xac>
   1bb90:	00c766b3          	or	a3,a4,a2
   1bb94:	66068063          	beqz	a3,1c1f4 <__adddf3+0x6d4>
   1bb98:	fff80593          	addi	a1,a6,-1
   1bb9c:	02059063          	bnez	a1,1bbbc <__adddf3+0x9c>
   1bba0:	00c50633          	add	a2,a0,a2
   1bba4:	00a636b3          	sltu	a3,a2,a0
   1bba8:	00e78733          	add	a4,a5,a4
   1bbac:	00060513          	mv	a0,a2
   1bbb0:	00d707b3          	add	a5,a4,a3
   1bbb4:	00100493          	li	s1,1
   1bbb8:	06c0006f          	j	1bc24 <__adddf3+0x104>
   1bbbc:	7ff00693          	li	a3,2047
   1bbc0:	02d81063          	bne	a6,a3,1bbe0 <__adddf3+0xc0>
   1bbc4:	7ff00493          	li	s1,2047
   1bbc8:	1f80006f          	j	1bdc0 <__adddf3+0x2a0>
   1bbcc:	7ff00693          	li	a3,2047
   1bbd0:	1ed48863          	beq	s1,a3,1bdc0 <__adddf3+0x2a0>
   1bbd4:	008006b7          	lui	a3,0x800
   1bbd8:	00d76733          	or	a4,a4,a3
   1bbdc:	00080593          	mv	a1,a6
   1bbe0:	03800693          	li	a3,56
   1bbe4:	0ab6c863          	blt	a3,a1,1bc94 <__adddf3+0x174>
   1bbe8:	01f00693          	li	a3,31
   1bbec:	06b6ca63          	blt	a3,a1,1bc60 <__adddf3+0x140>
   1bbf0:	02000813          	li	a6,32
   1bbf4:	40b80833          	sub	a6,a6,a1
   1bbf8:	010716b3          	sll	a3,a4,a6
   1bbfc:	00b658b3          	srl	a7,a2,a1
   1bc00:	01061833          	sll	a6,a2,a6
   1bc04:	0116e6b3          	or	a3,a3,a7
   1bc08:	01003833          	snez	a6,a6
   1bc0c:	0106e6b3          	or	a3,a3,a6
   1bc10:	00b755b3          	srl	a1,a4,a1
   1bc14:	00a68533          	add	a0,a3,a0
   1bc18:	00f585b3          	add	a1,a1,a5
   1bc1c:	00d536b3          	sltu	a3,a0,a3
   1bc20:	00d587b3          	add	a5,a1,a3
   1bc24:	00879713          	slli	a4,a5,0x8
   1bc28:	18075c63          	bgez	a4,1bdc0 <__adddf3+0x2a0>
   1bc2c:	00148493          	addi	s1,s1,1
   1bc30:	7ff00713          	li	a4,2047
   1bc34:	5ae48a63          	beq	s1,a4,1c1e8 <__adddf3+0x6c8>
   1bc38:	ff800737          	lui	a4,0xff800
   1bc3c:	fff70713          	addi	a4,a4,-1 # ff7fffff <__BSS_END__+0xff7dd1ff>
   1bc40:	00e7f733          	and	a4,a5,a4
   1bc44:	00155793          	srli	a5,a0,0x1
   1bc48:	00157513          	andi	a0,a0,1
   1bc4c:	00a7e7b3          	or	a5,a5,a0
   1bc50:	01f71513          	slli	a0,a4,0x1f
   1bc54:	00f56533          	or	a0,a0,a5
   1bc58:	00175793          	srli	a5,a4,0x1
   1bc5c:	1640006f          	j	1bdc0 <__adddf3+0x2a0>
   1bc60:	fe058693          	addi	a3,a1,-32
   1bc64:	02000893          	li	a7,32
   1bc68:	00d756b3          	srl	a3,a4,a3
   1bc6c:	00000813          	li	a6,0
   1bc70:	01158863          	beq	a1,a7,1bc80 <__adddf3+0x160>
   1bc74:	04000813          	li	a6,64
   1bc78:	40b80833          	sub	a6,a6,a1
   1bc7c:	01071833          	sll	a6,a4,a6
   1bc80:	00c86833          	or	a6,a6,a2
   1bc84:	01003833          	snez	a6,a6
   1bc88:	0106e6b3          	or	a3,a3,a6
   1bc8c:	00000593          	li	a1,0
   1bc90:	f85ff06f          	j	1bc14 <__adddf3+0xf4>
   1bc94:	00c766b3          	or	a3,a4,a2
   1bc98:	00d036b3          	snez	a3,a3
   1bc9c:	ff1ff06f          	j	1bc8c <__adddf3+0x16c>
   1bca0:	0c080a63          	beqz	a6,1bd74 <__adddf3+0x254>
   1bca4:	409886b3          	sub	a3,a7,s1
   1bca8:	02049463          	bnez	s1,1bcd0 <__adddf3+0x1b0>
   1bcac:	00a7e5b3          	or	a1,a5,a0
   1bcb0:	50058e63          	beqz	a1,1c1cc <__adddf3+0x6ac>
   1bcb4:	fff68593          	addi	a1,a3,-1 # 7fffff <__BSS_END__+0x7dd1ff>
   1bcb8:	ee0584e3          	beqz	a1,1bba0 <__adddf3+0x80>
   1bcbc:	7ff00813          	li	a6,2047
   1bcc0:	03069263          	bne	a3,a6,1bce4 <__adddf3+0x1c4>
   1bcc4:	00070793          	mv	a5,a4
   1bcc8:	00060513          	mv	a0,a2
   1bccc:	ef9ff06f          	j	1bbc4 <__adddf3+0xa4>
   1bcd0:	7ff00593          	li	a1,2047
   1bcd4:	feb888e3          	beq	a7,a1,1bcc4 <__adddf3+0x1a4>
   1bcd8:	008005b7          	lui	a1,0x800
   1bcdc:	00b7e7b3          	or	a5,a5,a1
   1bce0:	00068593          	mv	a1,a3
   1bce4:	03800693          	li	a3,56
   1bce8:	08b6c063          	blt	a3,a1,1bd68 <__adddf3+0x248>
   1bcec:	01f00693          	li	a3,31
   1bcf0:	04b6c263          	blt	a3,a1,1bd34 <__adddf3+0x214>
   1bcf4:	02000813          	li	a6,32
   1bcf8:	40b80833          	sub	a6,a6,a1
   1bcfc:	010796b3          	sll	a3,a5,a6
   1bd00:	00b55333          	srl	t1,a0,a1
   1bd04:	01051833          	sll	a6,a0,a6
   1bd08:	0066e6b3          	or	a3,a3,t1
   1bd0c:	01003833          	snez	a6,a6
   1bd10:	0106e6b3          	or	a3,a3,a6
   1bd14:	00b7d5b3          	srl	a1,a5,a1
   1bd18:	00c68633          	add	a2,a3,a2
   1bd1c:	00e585b3          	add	a1,a1,a4
   1bd20:	00d636b3          	sltu	a3,a2,a3
   1bd24:	00060513          	mv	a0,a2
   1bd28:	00d587b3          	add	a5,a1,a3
   1bd2c:	00088493          	mv	s1,a7
   1bd30:	ef5ff06f          	j	1bc24 <__adddf3+0x104>
   1bd34:	fe058693          	addi	a3,a1,-32 # 7fffe0 <__BSS_END__+0x7dd1e0>
   1bd38:	02000313          	li	t1,32
   1bd3c:	00d7d6b3          	srl	a3,a5,a3
   1bd40:	00000813          	li	a6,0
   1bd44:	00658863          	beq	a1,t1,1bd54 <__adddf3+0x234>
   1bd48:	04000813          	li	a6,64
   1bd4c:	40b80833          	sub	a6,a6,a1
   1bd50:	01079833          	sll	a6,a5,a6
   1bd54:	00a86833          	or	a6,a6,a0
   1bd58:	01003833          	snez	a6,a6
   1bd5c:	0106e6b3          	or	a3,a3,a6
   1bd60:	00000593          	li	a1,0
   1bd64:	fb5ff06f          	j	1bd18 <__adddf3+0x1f8>
   1bd68:	00a7e6b3          	or	a3,a5,a0
   1bd6c:	00d036b3          	snez	a3,a3
   1bd70:	ff1ff06f          	j	1bd60 <__adddf3+0x240>
   1bd74:	00148693          	addi	a3,s1,1
   1bd78:	7fe6f593          	andi	a1,a3,2046
   1bd7c:	08059663          	bnez	a1,1be08 <__adddf3+0x2e8>
   1bd80:	00a7e6b3          	or	a3,a5,a0
   1bd84:	06049263          	bnez	s1,1bde8 <__adddf3+0x2c8>
   1bd88:	44068863          	beqz	a3,1c1d8 <__adddf3+0x6b8>
   1bd8c:	00c766b3          	or	a3,a4,a2
   1bd90:	02068863          	beqz	a3,1bdc0 <__adddf3+0x2a0>
   1bd94:	00c50633          	add	a2,a0,a2
   1bd98:	00a636b3          	sltu	a3,a2,a0
   1bd9c:	00e78733          	add	a4,a5,a4
   1bda0:	00d707b3          	add	a5,a4,a3
   1bda4:	00879713          	slli	a4,a5,0x8
   1bda8:	00060513          	mv	a0,a2
   1bdac:	00075a63          	bgez	a4,1bdc0 <__adddf3+0x2a0>
   1bdb0:	ff800737          	lui	a4,0xff800
   1bdb4:	fff70713          	addi	a4,a4,-1 # ff7fffff <__BSS_END__+0xff7dd1ff>
   1bdb8:	00e7f7b3          	and	a5,a5,a4
   1bdbc:	00100493          	li	s1,1
   1bdc0:	00757713          	andi	a4,a0,7
   1bdc4:	44070863          	beqz	a4,1c214 <__adddf3+0x6f4>
   1bdc8:	00f57713          	andi	a4,a0,15
   1bdcc:	00400693          	li	a3,4
   1bdd0:	44d70263          	beq	a4,a3,1c214 <__adddf3+0x6f4>
   1bdd4:	00450713          	addi	a4,a0,4
   1bdd8:	00a736b3          	sltu	a3,a4,a0
   1bddc:	00d787b3          	add	a5,a5,a3
   1bde0:	00070513          	mv	a0,a4
   1bde4:	4300006f          	j	1c214 <__adddf3+0x6f4>
   1bde8:	ec068ee3          	beqz	a3,1bcc4 <__adddf3+0x1a4>
   1bdec:	00c76633          	or	a2,a4,a2
   1bdf0:	dc060ae3          	beqz	a2,1bbc4 <__adddf3+0xa4>
   1bdf4:	00000913          	li	s2,0
   1bdf8:	004007b7          	lui	a5,0x400
   1bdfc:	00000513          	li	a0,0
   1be00:	7ff00493          	li	s1,2047
   1be04:	4100006f          	j	1c214 <__adddf3+0x6f4>
   1be08:	7ff00593          	li	a1,2047
   1be0c:	3cb68c63          	beq	a3,a1,1c1e4 <__adddf3+0x6c4>
   1be10:	00c50633          	add	a2,a0,a2
   1be14:	00a63533          	sltu	a0,a2,a0
   1be18:	00e78733          	add	a4,a5,a4
   1be1c:	00a70733          	add	a4,a4,a0
   1be20:	01f71513          	slli	a0,a4,0x1f
   1be24:	00165613          	srli	a2,a2,0x1
   1be28:	00c56533          	or	a0,a0,a2
   1be2c:	00175793          	srli	a5,a4,0x1
   1be30:	00068493          	mv	s1,a3
   1be34:	f8dff06f          	j	1bdc0 <__adddf3+0x2a0>
   1be38:	0f005c63          	blez	a6,1bf30 <__adddf3+0x410>
   1be3c:	08089e63          	bnez	a7,1bed8 <__adddf3+0x3b8>
   1be40:	00c766b3          	or	a3,a4,a2
   1be44:	3a068863          	beqz	a3,1c1f4 <__adddf3+0x6d4>
   1be48:	fff80693          	addi	a3,a6,-1
   1be4c:	02069063          	bnez	a3,1be6c <__adddf3+0x34c>
   1be50:	40c50633          	sub	a2,a0,a2
   1be54:	00c536b3          	sltu	a3,a0,a2
   1be58:	40e78733          	sub	a4,a5,a4
   1be5c:	00060513          	mv	a0,a2
   1be60:	40d707b3          	sub	a5,a4,a3
   1be64:	00100493          	li	s1,1
   1be68:	0540006f          	j	1bebc <__adddf3+0x39c>
   1be6c:	7ff00593          	li	a1,2047
   1be70:	d4b80ae3          	beq	a6,a1,1bbc4 <__adddf3+0xa4>
   1be74:	03800593          	li	a1,56
   1be78:	0ad5c663          	blt	a1,a3,1bf24 <__adddf3+0x404>
   1be7c:	01f00593          	li	a1,31
   1be80:	06d5c863          	blt	a1,a3,1bef0 <__adddf3+0x3d0>
   1be84:	02000813          	li	a6,32
   1be88:	40d80833          	sub	a6,a6,a3
   1be8c:	00d658b3          	srl	a7,a2,a3
   1be90:	010715b3          	sll	a1,a4,a6
   1be94:	01061833          	sll	a6,a2,a6
   1be98:	0115e5b3          	or	a1,a1,a7
   1be9c:	01003833          	snez	a6,a6
   1bea0:	0105e633          	or	a2,a1,a6
   1bea4:	00d756b3          	srl	a3,a4,a3
   1bea8:	40c50633          	sub	a2,a0,a2
   1beac:	00c53733          	sltu	a4,a0,a2
   1beb0:	40d786b3          	sub	a3,a5,a3
   1beb4:	00060513          	mv	a0,a2
   1beb8:	40e687b3          	sub	a5,a3,a4
   1bebc:	00879713          	slli	a4,a5,0x8
   1bec0:	f00750e3          	bgez	a4,1bdc0 <__adddf3+0x2a0>
   1bec4:	00800437          	lui	s0,0x800
   1bec8:	fff40413          	addi	s0,s0,-1 # 7fffff <__BSS_END__+0x7dd1ff>
   1becc:	0087f433          	and	s0,a5,s0
   1bed0:	00050993          	mv	s3,a0
   1bed4:	2100006f          	j	1c0e4 <__adddf3+0x5c4>
   1bed8:	7ff00693          	li	a3,2047
   1bedc:	eed482e3          	beq	s1,a3,1bdc0 <__adddf3+0x2a0>
   1bee0:	008006b7          	lui	a3,0x800
   1bee4:	00d76733          	or	a4,a4,a3
   1bee8:	00080693          	mv	a3,a6
   1beec:	f89ff06f          	j	1be74 <__adddf3+0x354>
   1bef0:	fe068593          	addi	a1,a3,-32 # 7fffe0 <__BSS_END__+0x7dd1e0>
   1bef4:	02000893          	li	a7,32
   1bef8:	00b755b3          	srl	a1,a4,a1
   1befc:	00000813          	li	a6,0
   1bf00:	01168863          	beq	a3,a7,1bf10 <__adddf3+0x3f0>
   1bf04:	04000813          	li	a6,64
   1bf08:	40d80833          	sub	a6,a6,a3
   1bf0c:	01071833          	sll	a6,a4,a6
   1bf10:	00c86833          	or	a6,a6,a2
   1bf14:	01003833          	snez	a6,a6
   1bf18:	0105e633          	or	a2,a1,a6
   1bf1c:	00000693          	li	a3,0
   1bf20:	f89ff06f          	j	1bea8 <__adddf3+0x388>
   1bf24:	00c76633          	or	a2,a4,a2
   1bf28:	00c03633          	snez	a2,a2
   1bf2c:	ff1ff06f          	j	1bf1c <__adddf3+0x3fc>
   1bf30:	0e080863          	beqz	a6,1c020 <__adddf3+0x500>
   1bf34:	40988833          	sub	a6,a7,s1
   1bf38:	04049263          	bnez	s1,1bf7c <__adddf3+0x45c>
   1bf3c:	00a7e5b3          	or	a1,a5,a0
   1bf40:	2a058e63          	beqz	a1,1c1fc <__adddf3+0x6dc>
   1bf44:	fff80593          	addi	a1,a6,-1
   1bf48:	00059e63          	bnez	a1,1bf64 <__adddf3+0x444>
   1bf4c:	40a60533          	sub	a0,a2,a0
   1bf50:	40f70733          	sub	a4,a4,a5
   1bf54:	00a63633          	sltu	a2,a2,a0
   1bf58:	40c707b3          	sub	a5,a4,a2
   1bf5c:	00068913          	mv	s2,a3
   1bf60:	f05ff06f          	j	1be64 <__adddf3+0x344>
   1bf64:	7ff00313          	li	t1,2047
   1bf68:	02681463          	bne	a6,t1,1bf90 <__adddf3+0x470>
   1bf6c:	00070793          	mv	a5,a4
   1bf70:	00060513          	mv	a0,a2
   1bf74:	7ff00493          	li	s1,2047
   1bf78:	0d00006f          	j	1c048 <__adddf3+0x528>
   1bf7c:	7ff00593          	li	a1,2047
   1bf80:	feb886e3          	beq	a7,a1,1bf6c <__adddf3+0x44c>
   1bf84:	008005b7          	lui	a1,0x800
   1bf88:	00b7e7b3          	or	a5,a5,a1
   1bf8c:	00080593          	mv	a1,a6
   1bf90:	03800813          	li	a6,56
   1bf94:	08b84063          	blt	a6,a1,1c014 <__adddf3+0x4f4>
   1bf98:	01f00813          	li	a6,31
   1bf9c:	04b84263          	blt	a6,a1,1bfe0 <__adddf3+0x4c0>
   1bfa0:	02000313          	li	t1,32
   1bfa4:	40b30333          	sub	t1,t1,a1
   1bfa8:	00b55e33          	srl	t3,a0,a1
   1bfac:	00679833          	sll	a6,a5,t1
   1bfb0:	00651333          	sll	t1,a0,t1
   1bfb4:	01c86833          	or	a6,a6,t3
   1bfb8:	00603333          	snez	t1,t1
   1bfbc:	00686533          	or	a0,a6,t1
   1bfc0:	00b7d5b3          	srl	a1,a5,a1
   1bfc4:	40a60533          	sub	a0,a2,a0
   1bfc8:	40b705b3          	sub	a1,a4,a1
   1bfcc:	00a63633          	sltu	a2,a2,a0
   1bfd0:	40c587b3          	sub	a5,a1,a2
   1bfd4:	00088493          	mv	s1,a7
   1bfd8:	00068913          	mv	s2,a3
   1bfdc:	ee1ff06f          	j	1bebc <__adddf3+0x39c>
   1bfe0:	fe058813          	addi	a6,a1,-32 # 7fffe0 <__BSS_END__+0x7dd1e0>
   1bfe4:	02000e13          	li	t3,32
   1bfe8:	0107d833          	srl	a6,a5,a6
   1bfec:	00000313          	li	t1,0
   1bff0:	01c58863          	beq	a1,t3,1c000 <__adddf3+0x4e0>
   1bff4:	04000313          	li	t1,64
   1bff8:	40b30333          	sub	t1,t1,a1
   1bffc:	00679333          	sll	t1,a5,t1
   1c000:	00a36333          	or	t1,t1,a0
   1c004:	00603333          	snez	t1,t1
   1c008:	00686533          	or	a0,a6,t1
   1c00c:	00000593          	li	a1,0
   1c010:	fb5ff06f          	j	1bfc4 <__adddf3+0x4a4>
   1c014:	00a7e533          	or	a0,a5,a0
   1c018:	00a03533          	snez	a0,a0
   1c01c:	ff1ff06f          	j	1c00c <__adddf3+0x4ec>
   1c020:	00148593          	addi	a1,s1,1
   1c024:	7fe5f593          	andi	a1,a1,2046
   1c028:	08059663          	bnez	a1,1c0b4 <__adddf3+0x594>
   1c02c:	00a7e833          	or	a6,a5,a0
   1c030:	00c765b3          	or	a1,a4,a2
   1c034:	06049063          	bnez	s1,1c094 <__adddf3+0x574>
   1c038:	00081c63          	bnez	a6,1c050 <__adddf3+0x530>
   1c03c:	10058e63          	beqz	a1,1c158 <__adddf3+0x638>
   1c040:	00070793          	mv	a5,a4
   1c044:	00060513          	mv	a0,a2
   1c048:	00068913          	mv	s2,a3
   1c04c:	d75ff06f          	j	1bdc0 <__adddf3+0x2a0>
   1c050:	d60588e3          	beqz	a1,1bdc0 <__adddf3+0x2a0>
   1c054:	40c50833          	sub	a6,a0,a2
   1c058:	010538b3          	sltu	a7,a0,a6
   1c05c:	40e785b3          	sub	a1,a5,a4
   1c060:	411585b3          	sub	a1,a1,a7
   1c064:	00859893          	slli	a7,a1,0x8
   1c068:	0008dc63          	bgez	a7,1c080 <__adddf3+0x560>
   1c06c:	40a60533          	sub	a0,a2,a0
   1c070:	40f70733          	sub	a4,a4,a5
   1c074:	00a63633          	sltu	a2,a2,a0
   1c078:	40c707b3          	sub	a5,a4,a2
   1c07c:	fcdff06f          	j	1c048 <__adddf3+0x528>
   1c080:	00b86533          	or	a0,a6,a1
   1c084:	18050463          	beqz	a0,1c20c <__adddf3+0x6ec>
   1c088:	00058793          	mv	a5,a1
   1c08c:	00080513          	mv	a0,a6
   1c090:	d31ff06f          	j	1bdc0 <__adddf3+0x2a0>
   1c094:	00081c63          	bnez	a6,1c0ac <__adddf3+0x58c>
   1c098:	d4058ee3          	beqz	a1,1bdf4 <__adddf3+0x2d4>
   1c09c:	00070793          	mv	a5,a4
   1c0a0:	00060513          	mv	a0,a2
   1c0a4:	00068913          	mv	s2,a3
   1c0a8:	b1dff06f          	j	1bbc4 <__adddf3+0xa4>
   1c0ac:	b0058ce3          	beqz	a1,1bbc4 <__adddf3+0xa4>
   1c0b0:	d45ff06f          	j	1bdf4 <__adddf3+0x2d4>
   1c0b4:	40c505b3          	sub	a1,a0,a2
   1c0b8:	00b53833          	sltu	a6,a0,a1
   1c0bc:	40e78433          	sub	s0,a5,a4
   1c0c0:	41040433          	sub	s0,s0,a6
   1c0c4:	00841813          	slli	a6,s0,0x8
   1c0c8:	00058993          	mv	s3,a1
   1c0cc:	08085063          	bgez	a6,1c14c <__adddf3+0x62c>
   1c0d0:	40a609b3          	sub	s3,a2,a0
   1c0d4:	40f70433          	sub	s0,a4,a5
   1c0d8:	01363633          	sltu	a2,a2,s3
   1c0dc:	40c40433          	sub	s0,s0,a2
   1c0e0:	00068913          	mv	s2,a3
   1c0e4:	06040e63          	beqz	s0,1c160 <__adddf3+0x640>
   1c0e8:	00040513          	mv	a0,s0
   1c0ec:	650040ef          	jal	2073c <__clzsi2>
   1c0f0:	ff850693          	addi	a3,a0,-8
   1c0f4:	02000793          	li	a5,32
   1c0f8:	40d787b3          	sub	a5,a5,a3
   1c0fc:	00d41433          	sll	s0,s0,a3
   1c100:	00f9d7b3          	srl	a5,s3,a5
   1c104:	0087e7b3          	or	a5,a5,s0
   1c108:	00d99433          	sll	s0,s3,a3
   1c10c:	0a96c463          	blt	a3,s1,1c1b4 <__adddf3+0x694>
   1c110:	409686b3          	sub	a3,a3,s1
   1c114:	00168613          	addi	a2,a3,1
   1c118:	01f00713          	li	a4,31
   1c11c:	06c74263          	blt	a4,a2,1c180 <__adddf3+0x660>
   1c120:	02000713          	li	a4,32
   1c124:	40c70733          	sub	a4,a4,a2
   1c128:	00e79533          	sll	a0,a5,a4
   1c12c:	00c456b3          	srl	a3,s0,a2
   1c130:	00e41733          	sll	a4,s0,a4
   1c134:	00d56533          	or	a0,a0,a3
   1c138:	00e03733          	snez	a4,a4
   1c13c:	00e56533          	or	a0,a0,a4
   1c140:	00c7d7b3          	srl	a5,a5,a2
   1c144:	00000493          	li	s1,0
   1c148:	c79ff06f          	j	1bdc0 <__adddf3+0x2a0>
   1c14c:	0085e5b3          	or	a1,a1,s0
   1c150:	f8059ae3          	bnez	a1,1c0e4 <__adddf3+0x5c4>
   1c154:	00000493          	li	s1,0
   1c158:	00000913          	li	s2,0
   1c15c:	08c0006f          	j	1c1e8 <__adddf3+0x6c8>
   1c160:	00098513          	mv	a0,s3
   1c164:	5d8040ef          	jal	2073c <__clzsi2>
   1c168:	01850693          	addi	a3,a0,24
   1c16c:	01f00793          	li	a5,31
   1c170:	f8d7d2e3          	bge	a5,a3,1c0f4 <__adddf3+0x5d4>
   1c174:	ff850793          	addi	a5,a0,-8
   1c178:	00f997b3          	sll	a5,s3,a5
   1c17c:	f91ff06f          	j	1c10c <__adddf3+0x5ec>
   1c180:	fe168693          	addi	a3,a3,-31
   1c184:	00d7d533          	srl	a0,a5,a3
   1c188:	02000693          	li	a3,32
   1c18c:	00000713          	li	a4,0
   1c190:	00d60863          	beq	a2,a3,1c1a0 <__adddf3+0x680>
   1c194:	04000713          	li	a4,64
   1c198:	40c70733          	sub	a4,a4,a2
   1c19c:	00e79733          	sll	a4,a5,a4
   1c1a0:	00e46733          	or	a4,s0,a4
   1c1a4:	00e03733          	snez	a4,a4
   1c1a8:	00e56533          	or	a0,a0,a4
   1c1ac:	00000793          	li	a5,0
   1c1b0:	f95ff06f          	j	1c144 <__adddf3+0x624>
   1c1b4:	ff800737          	lui	a4,0xff800
   1c1b8:	fff70713          	addi	a4,a4,-1 # ff7fffff <__BSS_END__+0xff7dd1ff>
   1c1bc:	40d484b3          	sub	s1,s1,a3
   1c1c0:	00e7f7b3          	and	a5,a5,a4
   1c1c4:	00040513          	mv	a0,s0
   1c1c8:	bf9ff06f          	j	1bdc0 <__adddf3+0x2a0>
   1c1cc:	00070793          	mv	a5,a4
   1c1d0:	00060513          	mv	a0,a2
   1c1d4:	c5dff06f          	j	1be30 <__adddf3+0x310>
   1c1d8:	00070793          	mv	a5,a4
   1c1dc:	00060513          	mv	a0,a2
   1c1e0:	be1ff06f          	j	1bdc0 <__adddf3+0x2a0>
   1c1e4:	7ff00493          	li	s1,2047
   1c1e8:	00000793          	li	a5,0
   1c1ec:	00000513          	li	a0,0
   1c1f0:	0240006f          	j	1c214 <__adddf3+0x6f4>
   1c1f4:	00080493          	mv	s1,a6
   1c1f8:	bc9ff06f          	j	1bdc0 <__adddf3+0x2a0>
   1c1fc:	00070793          	mv	a5,a4
   1c200:	00060513          	mv	a0,a2
   1c204:	00080493          	mv	s1,a6
   1c208:	e41ff06f          	j	1c048 <__adddf3+0x528>
   1c20c:	00000793          	li	a5,0
   1c210:	00000913          	li	s2,0
   1c214:	00879713          	slli	a4,a5,0x8
   1c218:	00075e63          	bgez	a4,1c234 <__adddf3+0x714>
   1c21c:	00148493          	addi	s1,s1,1
   1c220:	7ff00713          	li	a4,2047
   1c224:	08e48263          	beq	s1,a4,1c2a8 <__adddf3+0x788>
   1c228:	ff800737          	lui	a4,0xff800
   1c22c:	fff70713          	addi	a4,a4,-1 # ff7fffff <__BSS_END__+0xff7dd1ff>
   1c230:	00e7f7b3          	and	a5,a5,a4
   1c234:	01d79693          	slli	a3,a5,0x1d
   1c238:	00355513          	srli	a0,a0,0x3
   1c23c:	7ff00713          	li	a4,2047
   1c240:	00a6e6b3          	or	a3,a3,a0
   1c244:	0037d793          	srli	a5,a5,0x3
   1c248:	00e49e63          	bne	s1,a4,1c264 <__adddf3+0x744>
   1c24c:	00f6e6b3          	or	a3,a3,a5
   1c250:	00000793          	li	a5,0
   1c254:	00068863          	beqz	a3,1c264 <__adddf3+0x744>
   1c258:	000807b7          	lui	a5,0x80
   1c25c:	00000693          	li	a3,0
   1c260:	00000913          	li	s2,0
   1c264:	01449713          	slli	a4,s1,0x14
   1c268:	7ff00637          	lui	a2,0x7ff00
   1c26c:	00c79793          	slli	a5,a5,0xc
   1c270:	00c77733          	and	a4,a4,a2
   1c274:	01c12083          	lw	ra,28(sp)
   1c278:	01812403          	lw	s0,24(sp)
   1c27c:	00c7d793          	srli	a5,a5,0xc
   1c280:	00f767b3          	or	a5,a4,a5
   1c284:	01f91713          	slli	a4,s2,0x1f
   1c288:	00e7e633          	or	a2,a5,a4
   1c28c:	01412483          	lw	s1,20(sp)
   1c290:	01012903          	lw	s2,16(sp)
   1c294:	00c12983          	lw	s3,12(sp)
   1c298:	00068513          	mv	a0,a3
   1c29c:	00060593          	mv	a1,a2
   1c2a0:	02010113          	addi	sp,sp,32
   1c2a4:	00008067          	ret
   1c2a8:	00000793          	li	a5,0
   1c2ac:	00000513          	li	a0,0
   1c2b0:	f85ff06f          	j	1c234 <__adddf3+0x714>

0001c2b4 <__divdf3>:
   1c2b4:	fd010113          	addi	sp,sp,-48
   1c2b8:	0145d813          	srli	a6,a1,0x14
   1c2bc:	02912223          	sw	s1,36(sp)
   1c2c0:	03212023          	sw	s2,32(sp)
   1c2c4:	01312e23          	sw	s3,28(sp)
   1c2c8:	01612823          	sw	s6,16(sp)
   1c2cc:	01712623          	sw	s7,12(sp)
   1c2d0:	00c59493          	slli	s1,a1,0xc
   1c2d4:	02112623          	sw	ra,44(sp)
   1c2d8:	02812423          	sw	s0,40(sp)
   1c2dc:	01412c23          	sw	s4,24(sp)
   1c2e0:	01512a23          	sw	s5,20(sp)
   1c2e4:	7ff87813          	andi	a6,a6,2047
   1c2e8:	00050b13          	mv	s6,a0
   1c2ec:	00060b93          	mv	s7,a2
   1c2f0:	00068913          	mv	s2,a3
   1c2f4:	00c4d493          	srli	s1,s1,0xc
   1c2f8:	01f5d993          	srli	s3,a1,0x1f
   1c2fc:	0a080263          	beqz	a6,1c3a0 <__divdf3+0xec>
   1c300:	7ff00793          	li	a5,2047
   1c304:	10f80263          	beq	a6,a5,1c408 <__divdf3+0x154>
   1c308:	01d55a13          	srli	s4,a0,0x1d
   1c30c:	00349493          	slli	s1,s1,0x3
   1c310:	009a6a33          	or	s4,s4,s1
   1c314:	008007b7          	lui	a5,0x800
   1c318:	00fa6a33          	or	s4,s4,a5
   1c31c:	00351413          	slli	s0,a0,0x3
   1c320:	c0180a93          	addi	s5,a6,-1023
   1c324:	00000b13          	li	s6,0
   1c328:	01495713          	srli	a4,s2,0x14
   1c32c:	00c91493          	slli	s1,s2,0xc
   1c330:	7ff77713          	andi	a4,a4,2047
   1c334:	00c4d493          	srli	s1,s1,0xc
   1c338:	01f95913          	srli	s2,s2,0x1f
   1c33c:	10070463          	beqz	a4,1c444 <__divdf3+0x190>
   1c340:	7ff00793          	li	a5,2047
   1c344:	16f70863          	beq	a4,a5,1c4b4 <__divdf3+0x200>
   1c348:	00349493          	slli	s1,s1,0x3
   1c34c:	01dbd793          	srli	a5,s7,0x1d
   1c350:	0097e7b3          	or	a5,a5,s1
   1c354:	008004b7          	lui	s1,0x800
   1c358:	0097e4b3          	or	s1,a5,s1
   1c35c:	003b9f13          	slli	t5,s7,0x3
   1c360:	c0170713          	addi	a4,a4,-1023
   1c364:	00000793          	li	a5,0
   1c368:	40ea8833          	sub	a6,s5,a4
   1c36c:	002b1713          	slli	a4,s6,0x2
   1c370:	00f76733          	or	a4,a4,a5
   1c374:	fff70713          	addi	a4,a4,-1
   1c378:	00e00693          	li	a3,14
   1c37c:	0129c633          	xor	a2,s3,s2
   1c380:	16e6e663          	bltu	a3,a4,1c4ec <__divdf3+0x238>
   1c384:	00005697          	auipc	a3,0x5
   1c388:	d3868693          	addi	a3,a3,-712 # 210bc <_ctype_+0x104>
   1c38c:	00271713          	slli	a4,a4,0x2
   1c390:	00d70733          	add	a4,a4,a3
   1c394:	00072703          	lw	a4,0(a4)
   1c398:	00d70733          	add	a4,a4,a3
   1c39c:	00070067          	jr	a4
   1c3a0:	00a4ea33          	or	s4,s1,a0
   1c3a4:	060a0e63          	beqz	s4,1c420 <__divdf3+0x16c>
   1c3a8:	02048e63          	beqz	s1,1c3e4 <__divdf3+0x130>
   1c3ac:	00048513          	mv	a0,s1
   1c3b0:	38c040ef          	jal	2073c <__clzsi2>
   1c3b4:	ff550793          	addi	a5,a0,-11
   1c3b8:	01d00a13          	li	s4,29
   1c3bc:	ff850713          	addi	a4,a0,-8
   1c3c0:	40fa0a33          	sub	s4,s4,a5
   1c3c4:	00e494b3          	sll	s1,s1,a4
   1c3c8:	014b5a33          	srl	s4,s6,s4
   1c3cc:	009a6a33          	or	s4,s4,s1
   1c3d0:	00eb14b3          	sll	s1,s6,a4
   1c3d4:	c0d00813          	li	a6,-1011
   1c3d8:	40a80ab3          	sub	s5,a6,a0
   1c3dc:	00048413          	mv	s0,s1
   1c3e0:	f45ff06f          	j	1c324 <__divdf3+0x70>
   1c3e4:	358040ef          	jal	2073c <__clzsi2>
   1c3e8:	00050a13          	mv	s4,a0
   1c3ec:	015a0793          	addi	a5,s4,21
   1c3f0:	01c00713          	li	a4,28
   1c3f4:	02050513          	addi	a0,a0,32
   1c3f8:	fcf750e3          	bge	a4,a5,1c3b8 <__divdf3+0x104>
   1c3fc:	ff8a0a13          	addi	s4,s4,-8
   1c400:	014b1a33          	sll	s4,s6,s4
   1c404:	fd1ff06f          	j	1c3d4 <__divdf3+0x120>
   1c408:	00a4ea33          	or	s4,s1,a0
   1c40c:	020a1263          	bnez	s4,1c430 <__divdf3+0x17c>
   1c410:	00000413          	li	s0,0
   1c414:	7ff00a93          	li	s5,2047
   1c418:	00200b13          	li	s6,2
   1c41c:	f0dff06f          	j	1c328 <__divdf3+0x74>
   1c420:	00000413          	li	s0,0
   1c424:	00000a93          	li	s5,0
   1c428:	00100b13          	li	s6,1
   1c42c:	efdff06f          	j	1c328 <__divdf3+0x74>
   1c430:	00050413          	mv	s0,a0
   1c434:	00048a13          	mv	s4,s1
   1c438:	7ff00a93          	li	s5,2047
   1c43c:	00300b13          	li	s6,3
   1c440:	ee9ff06f          	j	1c328 <__divdf3+0x74>
   1c444:	0174ef33          	or	t5,s1,s7
   1c448:	080f0263          	beqz	t5,1c4cc <__divdf3+0x218>
   1c44c:	04048063          	beqz	s1,1c48c <__divdf3+0x1d8>
   1c450:	00048513          	mv	a0,s1
   1c454:	2e8040ef          	jal	2073c <__clzsi2>
   1c458:	ff550713          	addi	a4,a0,-11
   1c45c:	01d00793          	li	a5,29
   1c460:	ff850693          	addi	a3,a0,-8
   1c464:	40e787b3          	sub	a5,a5,a4
   1c468:	00d494b3          	sll	s1,s1,a3
   1c46c:	00fbd7b3          	srl	a5,s7,a5
   1c470:	0097e7b3          	or	a5,a5,s1
   1c474:	00db94b3          	sll	s1,s7,a3
   1c478:	c0d00713          	li	a4,-1011
   1c47c:	00048f13          	mv	t5,s1
   1c480:	40a70733          	sub	a4,a4,a0
   1c484:	00078493          	mv	s1,a5
   1c488:	eddff06f          	j	1c364 <__divdf3+0xb0>
   1c48c:	000b8513          	mv	a0,s7
   1c490:	2ac040ef          	jal	2073c <__clzsi2>
   1c494:	00050793          	mv	a5,a0
   1c498:	01578713          	addi	a4,a5,21 # 800015 <__BSS_END__+0x7dd215>
   1c49c:	01c00693          	li	a3,28
   1c4a0:	02050513          	addi	a0,a0,32
   1c4a4:	fae6dce3          	bge	a3,a4,1c45c <__divdf3+0x1a8>
   1c4a8:	ff878793          	addi	a5,a5,-8
   1c4ac:	00fb97b3          	sll	a5,s7,a5
   1c4b0:	fc9ff06f          	j	1c478 <__divdf3+0x1c4>
   1c4b4:	0174ef33          	or	t5,s1,s7
   1c4b8:	020f1263          	bnez	t5,1c4dc <__divdf3+0x228>
   1c4bc:	00000493          	li	s1,0
   1c4c0:	7ff00713          	li	a4,2047
   1c4c4:	00200793          	li	a5,2
   1c4c8:	ea1ff06f          	j	1c368 <__divdf3+0xb4>
   1c4cc:	00000493          	li	s1,0
   1c4d0:	00000713          	li	a4,0
   1c4d4:	00100793          	li	a5,1
   1c4d8:	e91ff06f          	j	1c368 <__divdf3+0xb4>
   1c4dc:	000b8f13          	mv	t5,s7
   1c4e0:	7ff00713          	li	a4,2047
   1c4e4:	00300793          	li	a5,3
   1c4e8:	e81ff06f          	j	1c368 <__divdf3+0xb4>
   1c4ec:	0144e663          	bltu	s1,s4,1c4f8 <__divdf3+0x244>
   1c4f0:	349a1c63          	bne	s4,s1,1c848 <__divdf3+0x594>
   1c4f4:	35e46a63          	bltu	s0,t5,1c848 <__divdf3+0x594>
   1c4f8:	01fa1693          	slli	a3,s4,0x1f
   1c4fc:	00145793          	srli	a5,s0,0x1
   1c500:	01f41713          	slli	a4,s0,0x1f
   1c504:	001a5a13          	srli	s4,s4,0x1
   1c508:	00f6e433          	or	s0,a3,a5
   1c50c:	00849893          	slli	a7,s1,0x8
   1c510:	018f5593          	srli	a1,t5,0x18
   1c514:	0115e5b3          	or	a1,a1,a7
   1c518:	0108d893          	srli	a7,a7,0x10
   1c51c:	031a5eb3          	divu	t4,s4,a7
   1c520:	01059313          	slli	t1,a1,0x10
   1c524:	01035313          	srli	t1,t1,0x10
   1c528:	01045793          	srli	a5,s0,0x10
   1c52c:	008f1513          	slli	a0,t5,0x8
   1c530:	031a7a33          	remu	s4,s4,a7
   1c534:	000e8693          	mv	a3,t4
   1c538:	03d30e33          	mul	t3,t1,t4
   1c53c:	010a1a13          	slli	s4,s4,0x10
   1c540:	0147e7b3          	or	a5,a5,s4
   1c544:	01c7fe63          	bgeu	a5,t3,1c560 <__divdf3+0x2ac>
   1c548:	00f587b3          	add	a5,a1,a5
   1c54c:	fffe8693          	addi	a3,t4,-1
   1c550:	00b7e863          	bltu	a5,a1,1c560 <__divdf3+0x2ac>
   1c554:	01c7f663          	bgeu	a5,t3,1c560 <__divdf3+0x2ac>
   1c558:	ffee8693          	addi	a3,t4,-2
   1c55c:	00b787b3          	add	a5,a5,a1
   1c560:	41c787b3          	sub	a5,a5,t3
   1c564:	0317df33          	divu	t5,a5,a7
   1c568:	01041413          	slli	s0,s0,0x10
   1c56c:	01045413          	srli	s0,s0,0x10
   1c570:	0317f7b3          	remu	a5,a5,a7
   1c574:	000f0e13          	mv	t3,t5
   1c578:	03e30eb3          	mul	t4,t1,t5
   1c57c:	01079793          	slli	a5,a5,0x10
   1c580:	00f467b3          	or	a5,s0,a5
   1c584:	01d7fe63          	bgeu	a5,t4,1c5a0 <__divdf3+0x2ec>
   1c588:	00f587b3          	add	a5,a1,a5
   1c58c:	ffff0e13          	addi	t3,t5,-1
   1c590:	00b7e863          	bltu	a5,a1,1c5a0 <__divdf3+0x2ec>
   1c594:	01d7f663          	bgeu	a5,t4,1c5a0 <__divdf3+0x2ec>
   1c598:	ffef0e13          	addi	t3,t5,-2
   1c59c:	00b787b3          	add	a5,a5,a1
   1c5a0:	01069693          	slli	a3,a3,0x10
   1c5a4:	00010437          	lui	s0,0x10
   1c5a8:	01c6e2b3          	or	t0,a3,t3
   1c5ac:	fff40e13          	addi	t3,s0,-1 # ffff <exit-0xb5>
   1c5b0:	01c2f6b3          	and	a3,t0,t3
   1c5b4:	0102df93          	srli	t6,t0,0x10
   1c5b8:	01c57e33          	and	t3,a0,t3
   1c5bc:	41d787b3          	sub	a5,a5,t4
   1c5c0:	01055e93          	srli	t4,a0,0x10
   1c5c4:	02de03b3          	mul	t2,t3,a3
   1c5c8:	03cf84b3          	mul	s1,t6,t3
   1c5cc:	02de86b3          	mul	a3,t4,a3
   1c5d0:	00968f33          	add	t5,a3,s1
   1c5d4:	0103d693          	srli	a3,t2,0x10
   1c5d8:	01e686b3          	add	a3,a3,t5
   1c5dc:	03df8fb3          	mul	t6,t6,t4
   1c5e0:	0096f463          	bgeu	a3,s1,1c5e8 <__divdf3+0x334>
   1c5e4:	008f8fb3          	add	t6,t6,s0
   1c5e8:	0106df13          	srli	t5,a3,0x10
   1c5ec:	01ff0f33          	add	t5,t5,t6
   1c5f0:	00010fb7          	lui	t6,0x10
   1c5f4:	ffff8f93          	addi	t6,t6,-1 # ffff <exit-0xb5>
   1c5f8:	01f6f6b3          	and	a3,a3,t6
   1c5fc:	01069693          	slli	a3,a3,0x10
   1c600:	01f3f3b3          	and	t2,t2,t6
   1c604:	007686b3          	add	a3,a3,t2
   1c608:	01e7e863          	bltu	a5,t5,1c618 <__divdf3+0x364>
   1c60c:	00028493          	mv	s1,t0
   1c610:	05e79863          	bne	a5,t5,1c660 <__divdf3+0x3ac>
   1c614:	04d77663          	bgeu	a4,a3,1c660 <__divdf3+0x3ac>
   1c618:	00a70fb3          	add	t6,a4,a0
   1c61c:	00efb3b3          	sltu	t2,t6,a4
   1c620:	00b38433          	add	s0,t2,a1
   1c624:	008787b3          	add	a5,a5,s0
   1c628:	fff28493          	addi	s1,t0,-1
   1c62c:	000f8713          	mv	a4,t6
   1c630:	00f5e663          	bltu	a1,a5,1c63c <__divdf3+0x388>
   1c634:	02f59663          	bne	a1,a5,1c660 <__divdf3+0x3ac>
   1c638:	02039463          	bnez	t2,1c660 <__divdf3+0x3ac>
   1c63c:	01e7e663          	bltu	a5,t5,1c648 <__divdf3+0x394>
   1c640:	02ff1063          	bne	t5,a5,1c660 <__divdf3+0x3ac>
   1c644:	00dffe63          	bgeu	t6,a3,1c660 <__divdf3+0x3ac>
   1c648:	01f50fb3          	add	t6,a0,t6
   1c64c:	000f8713          	mv	a4,t6
   1c650:	00afbfb3          	sltu	t6,t6,a0
   1c654:	00bf8fb3          	add	t6,t6,a1
   1c658:	ffe28493          	addi	s1,t0,-2
   1c65c:	01f787b3          	add	a5,a5,t6
   1c660:	40d706b3          	sub	a3,a4,a3
   1c664:	41e787b3          	sub	a5,a5,t5
   1c668:	00d73733          	sltu	a4,a4,a3
   1c66c:	40e787b3          	sub	a5,a5,a4
   1c670:	fff00f13          	li	t5,-1
   1c674:	12f58663          	beq	a1,a5,1c7a0 <__divdf3+0x4ec>
   1c678:	0317dfb3          	divu	t6,a5,a7
   1c67c:	0106d713          	srli	a4,a3,0x10
   1c680:	0317f7b3          	remu	a5,a5,a7
   1c684:	03f30f33          	mul	t5,t1,t6
   1c688:	01079793          	slli	a5,a5,0x10
   1c68c:	00f767b3          	or	a5,a4,a5
   1c690:	000f8713          	mv	a4,t6
   1c694:	01e7fe63          	bgeu	a5,t5,1c6b0 <__divdf3+0x3fc>
   1c698:	00f587b3          	add	a5,a1,a5
   1c69c:	ffff8713          	addi	a4,t6,-1
   1c6a0:	00b7e863          	bltu	a5,a1,1c6b0 <__divdf3+0x3fc>
   1c6a4:	01e7f663          	bgeu	a5,t5,1c6b0 <__divdf3+0x3fc>
   1c6a8:	ffef8713          	addi	a4,t6,-2
   1c6ac:	00b787b3          	add	a5,a5,a1
   1c6b0:	41e787b3          	sub	a5,a5,t5
   1c6b4:	0317df33          	divu	t5,a5,a7
   1c6b8:	01069693          	slli	a3,a3,0x10
   1c6bc:	0106d693          	srli	a3,a3,0x10
   1c6c0:	0317f7b3          	remu	a5,a5,a7
   1c6c4:	000f0893          	mv	a7,t5
   1c6c8:	03e30333          	mul	t1,t1,t5
   1c6cc:	01079793          	slli	a5,a5,0x10
   1c6d0:	00f6e7b3          	or	a5,a3,a5
   1c6d4:	0067fe63          	bgeu	a5,t1,1c6f0 <__divdf3+0x43c>
   1c6d8:	00f587b3          	add	a5,a1,a5
   1c6dc:	ffff0893          	addi	a7,t5,-1
   1c6e0:	00b7e863          	bltu	a5,a1,1c6f0 <__divdf3+0x43c>
   1c6e4:	0067f663          	bgeu	a5,t1,1c6f0 <__divdf3+0x43c>
   1c6e8:	ffef0893          	addi	a7,t5,-2
   1c6ec:	00b787b3          	add	a5,a5,a1
   1c6f0:	01071693          	slli	a3,a4,0x10
   1c6f4:	0116e6b3          	or	a3,a3,a7
   1c6f8:	01069713          	slli	a4,a3,0x10
   1c6fc:	01075713          	srli	a4,a4,0x10
   1c700:	406787b3          	sub	a5,a5,t1
   1c704:	0106d313          	srli	t1,a3,0x10
   1c708:	03c70f33          	mul	t5,a4,t3
   1c70c:	03c30e33          	mul	t3,t1,t3
   1c710:	026e8333          	mul	t1,t4,t1
   1c714:	02ee8eb3          	mul	t4,t4,a4
   1c718:	010f5713          	srli	a4,t5,0x10
   1c71c:	01ce8eb3          	add	t4,t4,t3
   1c720:	01d70733          	add	a4,a4,t4
   1c724:	01c77663          	bgeu	a4,t3,1c730 <__divdf3+0x47c>
   1c728:	000108b7          	lui	a7,0x10
   1c72c:	01130333          	add	t1,t1,a7
   1c730:	01075893          	srli	a7,a4,0x10
   1c734:	006888b3          	add	a7,a7,t1
   1c738:	00010337          	lui	t1,0x10
   1c73c:	fff30313          	addi	t1,t1,-1 # ffff <exit-0xb5>
   1c740:	00677733          	and	a4,a4,t1
   1c744:	01071713          	slli	a4,a4,0x10
   1c748:	006f7f33          	and	t5,t5,t1
   1c74c:	01e70733          	add	a4,a4,t5
   1c750:	0117e863          	bltu	a5,a7,1c760 <__divdf3+0x4ac>
   1c754:	23179c63          	bne	a5,a7,1c98c <__divdf3+0x6d8>
   1c758:	00068f13          	mv	t5,a3
   1c75c:	04070263          	beqz	a4,1c7a0 <__divdf3+0x4ec>
   1c760:	00f587b3          	add	a5,a1,a5
   1c764:	fff68f13          	addi	t5,a3,-1
   1c768:	00078313          	mv	t1,a5
   1c76c:	02b7e463          	bltu	a5,a1,1c794 <__divdf3+0x4e0>
   1c770:	0117e663          	bltu	a5,a7,1c77c <__divdf3+0x4c8>
   1c774:	21179a63          	bne	a5,a7,1c988 <__divdf3+0x6d4>
   1c778:	02e57063          	bgeu	a0,a4,1c798 <__divdf3+0x4e4>
   1c77c:	ffe68f13          	addi	t5,a3,-2
   1c780:	00151693          	slli	a3,a0,0x1
   1c784:	00a6b333          	sltu	t1,a3,a0
   1c788:	00b30333          	add	t1,t1,a1
   1c78c:	00678333          	add	t1,a5,t1
   1c790:	00068513          	mv	a0,a3
   1c794:	01131463          	bne	t1,a7,1c79c <__divdf3+0x4e8>
   1c798:	00a70463          	beq	a4,a0,1c7a0 <__divdf3+0x4ec>
   1c79c:	001f6f13          	ori	t5,t5,1
   1c7a0:	3ff80713          	addi	a4,a6,1023
   1c7a4:	10e05263          	blez	a4,1c8a8 <__divdf3+0x5f4>
   1c7a8:	007f7793          	andi	a5,t5,7
   1c7ac:	02078063          	beqz	a5,1c7cc <__divdf3+0x518>
   1c7b0:	00ff7793          	andi	a5,t5,15
   1c7b4:	00400693          	li	a3,4
   1c7b8:	00d78a63          	beq	a5,a3,1c7cc <__divdf3+0x518>
   1c7bc:	004f0793          	addi	a5,t5,4
   1c7c0:	01e7b6b3          	sltu	a3,a5,t5
   1c7c4:	00d484b3          	add	s1,s1,a3
   1c7c8:	00078f13          	mv	t5,a5
   1c7cc:	00749793          	slli	a5,s1,0x7
   1c7d0:	0007da63          	bgez	a5,1c7e4 <__divdf3+0x530>
   1c7d4:	ff0007b7          	lui	a5,0xff000
   1c7d8:	fff78793          	addi	a5,a5,-1 # feffffff <__BSS_END__+0xfefdd1ff>
   1c7dc:	00f4f4b3          	and	s1,s1,a5
   1c7e0:	40080713          	addi	a4,a6,1024
   1c7e4:	7fe00793          	li	a5,2046
   1c7e8:	08e7ca63          	blt	a5,a4,1c87c <__divdf3+0x5c8>
   1c7ec:	003f5f13          	srli	t5,t5,0x3
   1c7f0:	01d49793          	slli	a5,s1,0x1d
   1c7f4:	01e7ef33          	or	t5,a5,t5
   1c7f8:	0034d513          	srli	a0,s1,0x3
   1c7fc:	00c51513          	slli	a0,a0,0xc
   1c800:	02c12083          	lw	ra,44(sp)
   1c804:	02812403          	lw	s0,40(sp)
   1c808:	00c55513          	srli	a0,a0,0xc
   1c80c:	01471713          	slli	a4,a4,0x14
   1c810:	00a76733          	or	a4,a4,a0
   1c814:	01f61613          	slli	a2,a2,0x1f
   1c818:	00c767b3          	or	a5,a4,a2
   1c81c:	02412483          	lw	s1,36(sp)
   1c820:	02012903          	lw	s2,32(sp)
   1c824:	01c12983          	lw	s3,28(sp)
   1c828:	01812a03          	lw	s4,24(sp)
   1c82c:	01412a83          	lw	s5,20(sp)
   1c830:	01012b03          	lw	s6,16(sp)
   1c834:	00c12b83          	lw	s7,12(sp)
   1c838:	000f0513          	mv	a0,t5
   1c83c:	00078593          	mv	a1,a5
   1c840:	03010113          	addi	sp,sp,48
   1c844:	00008067          	ret
   1c848:	fff80813          	addi	a6,a6,-1
   1c84c:	00000713          	li	a4,0
   1c850:	cbdff06f          	j	1c50c <__divdf3+0x258>
   1c854:	00098613          	mv	a2,s3
   1c858:	000a0493          	mv	s1,s4
   1c85c:	00040f13          	mv	t5,s0
   1c860:	000b0793          	mv	a5,s6
   1c864:	00300713          	li	a4,3
   1c868:	0ee78863          	beq	a5,a4,1c958 <__divdf3+0x6a4>
   1c86c:	00100713          	li	a4,1
   1c870:	0ee78e63          	beq	a5,a4,1c96c <__divdf3+0x6b8>
   1c874:	00200713          	li	a4,2
   1c878:	f2e794e3          	bne	a5,a4,1c7a0 <__divdf3+0x4ec>
   1c87c:	00000513          	li	a0,0
   1c880:	00000f13          	li	t5,0
   1c884:	7ff00713          	li	a4,2047
   1c888:	f75ff06f          	j	1c7fc <__divdf3+0x548>
   1c88c:	00090613          	mv	a2,s2
   1c890:	fd5ff06f          	j	1c864 <__divdf3+0x5b0>
   1c894:	000804b7          	lui	s1,0x80
   1c898:	00000f13          	li	t5,0
   1c89c:	00000613          	li	a2,0
   1c8a0:	00300793          	li	a5,3
   1c8a4:	fc1ff06f          	j	1c864 <__divdf3+0x5b0>
   1c8a8:	00100513          	li	a0,1
   1c8ac:	40e50533          	sub	a0,a0,a4
   1c8b0:	03800793          	li	a5,56
   1c8b4:	0aa7cc63          	blt	a5,a0,1c96c <__divdf3+0x6b8>
   1c8b8:	01f00793          	li	a5,31
   1c8bc:	06a7c463          	blt	a5,a0,1c924 <__divdf3+0x670>
   1c8c0:	41e80813          	addi	a6,a6,1054
   1c8c4:	010497b3          	sll	a5,s1,a6
   1c8c8:	00af5733          	srl	a4,t5,a0
   1c8cc:	010f1833          	sll	a6,t5,a6
   1c8d0:	00e7e7b3          	or	a5,a5,a4
   1c8d4:	01003833          	snez	a6,a6
   1c8d8:	0107e7b3          	or	a5,a5,a6
   1c8dc:	00a4d533          	srl	a0,s1,a0
   1c8e0:	0077f713          	andi	a4,a5,7
   1c8e4:	02070063          	beqz	a4,1c904 <__divdf3+0x650>
   1c8e8:	00f7f713          	andi	a4,a5,15
   1c8ec:	00400693          	li	a3,4
   1c8f0:	00d70a63          	beq	a4,a3,1c904 <__divdf3+0x650>
   1c8f4:	00478713          	addi	a4,a5,4
   1c8f8:	00f736b3          	sltu	a3,a4,a5
   1c8fc:	00d50533          	add	a0,a0,a3
   1c900:	00070793          	mv	a5,a4
   1c904:	00851713          	slli	a4,a0,0x8
   1c908:	06074863          	bltz	a4,1c978 <__divdf3+0x6c4>
   1c90c:	01d51f13          	slli	t5,a0,0x1d
   1c910:	0037d793          	srli	a5,a5,0x3
   1c914:	00ff6f33          	or	t5,t5,a5
   1c918:	00355513          	srli	a0,a0,0x3
   1c91c:	00000713          	li	a4,0
   1c920:	eddff06f          	j	1c7fc <__divdf3+0x548>
   1c924:	fe100793          	li	a5,-31
   1c928:	40e787b3          	sub	a5,a5,a4
   1c92c:	02000693          	li	a3,32
   1c930:	00f4d7b3          	srl	a5,s1,a5
   1c934:	00000713          	li	a4,0
   1c938:	00d50663          	beq	a0,a3,1c944 <__divdf3+0x690>
   1c93c:	43e80713          	addi	a4,a6,1086
   1c940:	00e49733          	sll	a4,s1,a4
   1c944:	01e76733          	or	a4,a4,t5
   1c948:	00e03733          	snez	a4,a4
   1c94c:	00e7e7b3          	or	a5,a5,a4
   1c950:	00000513          	li	a0,0
   1c954:	f8dff06f          	j	1c8e0 <__divdf3+0x62c>
   1c958:	00080537          	lui	a0,0x80
   1c95c:	00000f13          	li	t5,0
   1c960:	7ff00713          	li	a4,2047
   1c964:	00000613          	li	a2,0
   1c968:	e95ff06f          	j	1c7fc <__divdf3+0x548>
   1c96c:	00000513          	li	a0,0
   1c970:	00000f13          	li	t5,0
   1c974:	fa9ff06f          	j	1c91c <__divdf3+0x668>
   1c978:	00000513          	li	a0,0
   1c97c:	00000f13          	li	t5,0
   1c980:	00100713          	li	a4,1
   1c984:	e79ff06f          	j	1c7fc <__divdf3+0x548>
   1c988:	000f0693          	mv	a3,t5
   1c98c:	00068f13          	mv	t5,a3
   1c990:	e0dff06f          	j	1c79c <__divdf3+0x4e8>

0001c994 <__eqdf2>:
   1c994:	0145d713          	srli	a4,a1,0x14
   1c998:	001007b7          	lui	a5,0x100
   1c99c:	fff78793          	addi	a5,a5,-1 # fffff <__BSS_END__+0xdd1ff>
   1c9a0:	0146d813          	srli	a6,a3,0x14
   1c9a4:	00050313          	mv	t1,a0
   1c9a8:	00050e93          	mv	t4,a0
   1c9ac:	7ff77713          	andi	a4,a4,2047
   1c9b0:	7ff00513          	li	a0,2047
   1c9b4:	00b7f8b3          	and	a7,a5,a1
   1c9b8:	00060f13          	mv	t5,a2
   1c9bc:	00d7f7b3          	and	a5,a5,a3
   1c9c0:	01f5d593          	srli	a1,a1,0x1f
   1c9c4:	7ff87813          	andi	a6,a6,2047
   1c9c8:	01f6d693          	srli	a3,a3,0x1f
   1c9cc:	00a71c63          	bne	a4,a0,1c9e4 <__eqdf2+0x50>
   1c9d0:	0068ee33          	or	t3,a7,t1
   1c9d4:	00100513          	li	a0,1
   1c9d8:	000e1463          	bnez	t3,1c9e0 <__eqdf2+0x4c>
   1c9dc:	00e80663          	beq	a6,a4,1c9e8 <__eqdf2+0x54>
   1c9e0:	00008067          	ret
   1c9e4:	00a81863          	bne	a6,a0,1c9f4 <__eqdf2+0x60>
   1c9e8:	00c7e633          	or	a2,a5,a2
   1c9ec:	00100513          	li	a0,1
   1c9f0:	fe0618e3          	bnez	a2,1c9e0 <__eqdf2+0x4c>
   1c9f4:	00100513          	li	a0,1
   1c9f8:	ff0714e3          	bne	a4,a6,1c9e0 <__eqdf2+0x4c>
   1c9fc:	fef892e3          	bne	a7,a5,1c9e0 <__eqdf2+0x4c>
   1ca00:	ffee90e3          	bne	t4,t5,1c9e0 <__eqdf2+0x4c>
   1ca04:	00d58a63          	beq	a1,a3,1ca18 <__eqdf2+0x84>
   1ca08:	fc071ce3          	bnez	a4,1c9e0 <__eqdf2+0x4c>
   1ca0c:	0068e8b3          	or	a7,a7,t1
   1ca10:	01103533          	snez	a0,a7
   1ca14:	00008067          	ret
   1ca18:	00000513          	li	a0,0
   1ca1c:	00008067          	ret

0001ca20 <__gedf2>:
   1ca20:	0146d793          	srli	a5,a3,0x14
   1ca24:	0145d893          	srli	a7,a1,0x14
   1ca28:	00100737          	lui	a4,0x100
   1ca2c:	fff70713          	addi	a4,a4,-1 # fffff <__BSS_END__+0xdd1ff>
   1ca30:	00050813          	mv	a6,a0
   1ca34:	00050e13          	mv	t3,a0
   1ca38:	7ff8f893          	andi	a7,a7,2047
   1ca3c:	7ff7f513          	andi	a0,a5,2047
   1ca40:	7ff00793          	li	a5,2047
   1ca44:	00b77333          	and	t1,a4,a1
   1ca48:	00060e93          	mv	t4,a2
   1ca4c:	00d77733          	and	a4,a4,a3
   1ca50:	01f5d593          	srli	a1,a1,0x1f
   1ca54:	01f6d693          	srli	a3,a3,0x1f
   1ca58:	00f89663          	bne	a7,a5,1ca64 <__gedf2+0x44>
   1ca5c:	010367b3          	or	a5,t1,a6
   1ca60:	06079c63          	bnez	a5,1cad8 <__gedf2+0xb8>
   1ca64:	7ff00793          	li	a5,2047
   1ca68:	00f51663          	bne	a0,a5,1ca74 <__gedf2+0x54>
   1ca6c:	00c767b3          	or	a5,a4,a2
   1ca70:	06079463          	bnez	a5,1cad8 <__gedf2+0xb8>
   1ca74:	00000793          	li	a5,0
   1ca78:	00089663          	bnez	a7,1ca84 <__gedf2+0x64>
   1ca7c:	01036833          	or	a6,t1,a6
   1ca80:	00183793          	seqz	a5,a6
   1ca84:	04051e63          	bnez	a0,1cae0 <__gedf2+0xc0>
   1ca88:	00c76633          	or	a2,a4,a2
   1ca8c:	00078c63          	beqz	a5,1caa4 <__gedf2+0x84>
   1ca90:	02060063          	beqz	a2,1cab0 <__gedf2+0x90>
   1ca94:	00100513          	li	a0,1
   1ca98:	00069c63          	bnez	a3,1cab0 <__gedf2+0x90>
   1ca9c:	fff00513          	li	a0,-1
   1caa0:	00008067          	ret
   1caa4:	04061063          	bnez	a2,1cae4 <__gedf2+0xc4>
   1caa8:	fff00513          	li	a0,-1
   1caac:	04058463          	beqz	a1,1caf4 <__gedf2+0xd4>
   1cab0:	00008067          	ret
   1cab4:	fea8c0e3          	blt	a7,a0,1ca94 <__gedf2+0x74>
   1cab8:	fe6768e3          	bltu	a4,t1,1caa8 <__gedf2+0x88>
   1cabc:	00e31863          	bne	t1,a4,1cacc <__gedf2+0xac>
   1cac0:	ffcee4e3          	bltu	t4,t3,1caa8 <__gedf2+0x88>
   1cac4:	00000513          	li	a0,0
   1cac8:	ffde74e3          	bgeu	t3,t4,1cab0 <__gedf2+0x90>
   1cacc:	00100513          	li	a0,1
   1cad0:	fe0590e3          	bnez	a1,1cab0 <__gedf2+0x90>
   1cad4:	fc9ff06f          	j	1ca9c <__gedf2+0x7c>
   1cad8:	ffe00513          	li	a0,-2
   1cadc:	00008067          	ret
   1cae0:	fa079ae3          	bnez	a5,1ca94 <__gedf2+0x74>
   1cae4:	fcb692e3          	bne	a3,a1,1caa8 <__gedf2+0x88>
   1cae8:	fd1556e3          	bge	a0,a7,1cab4 <__gedf2+0x94>
   1caec:	fff00513          	li	a0,-1
   1caf0:	fc0690e3          	bnez	a3,1cab0 <__gedf2+0x90>
   1caf4:	00100513          	li	a0,1
   1caf8:	00008067          	ret

0001cafc <__ledf2>:
   1cafc:	0146d793          	srli	a5,a3,0x14
   1cb00:	0145d893          	srli	a7,a1,0x14
   1cb04:	00100737          	lui	a4,0x100
   1cb08:	fff70713          	addi	a4,a4,-1 # fffff <__BSS_END__+0xdd1ff>
   1cb0c:	00050813          	mv	a6,a0
   1cb10:	00050e13          	mv	t3,a0
   1cb14:	7ff8f893          	andi	a7,a7,2047
   1cb18:	7ff7f513          	andi	a0,a5,2047
   1cb1c:	7ff00793          	li	a5,2047
   1cb20:	00b77333          	and	t1,a4,a1
   1cb24:	00060e93          	mv	t4,a2
   1cb28:	00d77733          	and	a4,a4,a3
   1cb2c:	01f5d593          	srli	a1,a1,0x1f
   1cb30:	01f6d693          	srli	a3,a3,0x1f
   1cb34:	00f89663          	bne	a7,a5,1cb40 <__ledf2+0x44>
   1cb38:	010367b3          	or	a5,t1,a6
   1cb3c:	06079c63          	bnez	a5,1cbb4 <__ledf2+0xb8>
   1cb40:	7ff00793          	li	a5,2047
   1cb44:	00f51663          	bne	a0,a5,1cb50 <__ledf2+0x54>
   1cb48:	00c767b3          	or	a5,a4,a2
   1cb4c:	06079463          	bnez	a5,1cbb4 <__ledf2+0xb8>
   1cb50:	00000793          	li	a5,0
   1cb54:	00089663          	bnez	a7,1cb60 <__ledf2+0x64>
   1cb58:	01036833          	or	a6,t1,a6
   1cb5c:	00183793          	seqz	a5,a6
   1cb60:	04051e63          	bnez	a0,1cbbc <__ledf2+0xc0>
   1cb64:	00c76633          	or	a2,a4,a2
   1cb68:	00078c63          	beqz	a5,1cb80 <__ledf2+0x84>
   1cb6c:	02060063          	beqz	a2,1cb8c <__ledf2+0x90>
   1cb70:	00100513          	li	a0,1
   1cb74:	00069c63          	bnez	a3,1cb8c <__ledf2+0x90>
   1cb78:	fff00513          	li	a0,-1
   1cb7c:	00008067          	ret
   1cb80:	04061063          	bnez	a2,1cbc0 <__ledf2+0xc4>
   1cb84:	fff00513          	li	a0,-1
   1cb88:	04058463          	beqz	a1,1cbd0 <__ledf2+0xd4>
   1cb8c:	00008067          	ret
   1cb90:	fea8c0e3          	blt	a7,a0,1cb70 <__ledf2+0x74>
   1cb94:	fe6768e3          	bltu	a4,t1,1cb84 <__ledf2+0x88>
   1cb98:	00e31863          	bne	t1,a4,1cba8 <__ledf2+0xac>
   1cb9c:	ffcee4e3          	bltu	t4,t3,1cb84 <__ledf2+0x88>
   1cba0:	00000513          	li	a0,0
   1cba4:	ffde74e3          	bgeu	t3,t4,1cb8c <__ledf2+0x90>
   1cba8:	00100513          	li	a0,1
   1cbac:	fe0590e3          	bnez	a1,1cb8c <__ledf2+0x90>
   1cbb0:	fc9ff06f          	j	1cb78 <__ledf2+0x7c>
   1cbb4:	00200513          	li	a0,2
   1cbb8:	00008067          	ret
   1cbbc:	fa079ae3          	bnez	a5,1cb70 <__ledf2+0x74>
   1cbc0:	fcb692e3          	bne	a3,a1,1cb84 <__ledf2+0x88>
   1cbc4:	fd1556e3          	bge	a0,a7,1cb90 <__ledf2+0x94>
   1cbc8:	fff00513          	li	a0,-1
   1cbcc:	fc0690e3          	bnez	a3,1cb8c <__ledf2+0x90>
   1cbd0:	00100513          	li	a0,1
   1cbd4:	00008067          	ret

0001cbd8 <__muldf3>:
   1cbd8:	fd010113          	addi	sp,sp,-48
   1cbdc:	01512a23          	sw	s5,20(sp)
   1cbe0:	0145da93          	srli	s5,a1,0x14
   1cbe4:	02812423          	sw	s0,40(sp)
   1cbe8:	02912223          	sw	s1,36(sp)
   1cbec:	01312e23          	sw	s3,28(sp)
   1cbf0:	01412c23          	sw	s4,24(sp)
   1cbf4:	01612823          	sw	s6,16(sp)
   1cbf8:	00c59493          	slli	s1,a1,0xc
   1cbfc:	02112623          	sw	ra,44(sp)
   1cc00:	03212023          	sw	s2,32(sp)
   1cc04:	01712623          	sw	s7,12(sp)
   1cc08:	7ffafa93          	andi	s5,s5,2047
   1cc0c:	00050413          	mv	s0,a0
   1cc10:	00060b13          	mv	s6,a2
   1cc14:	00068993          	mv	s3,a3
   1cc18:	00c4d493          	srli	s1,s1,0xc
   1cc1c:	01f5da13          	srli	s4,a1,0x1f
   1cc20:	240a8e63          	beqz	s5,1ce7c <__muldf3+0x2a4>
   1cc24:	7ff00793          	li	a5,2047
   1cc28:	2cfa8063          	beq	s5,a5,1cee8 <__muldf3+0x310>
   1cc2c:	00349493          	slli	s1,s1,0x3
   1cc30:	01d55793          	srli	a5,a0,0x1d
   1cc34:	0097e7b3          	or	a5,a5,s1
   1cc38:	008004b7          	lui	s1,0x800
   1cc3c:	0097e4b3          	or	s1,a5,s1
   1cc40:	00351913          	slli	s2,a0,0x3
   1cc44:	c01a8a93          	addi	s5,s5,-1023
   1cc48:	00000b93          	li	s7,0
   1cc4c:	0149d713          	srli	a4,s3,0x14
   1cc50:	00c99413          	slli	s0,s3,0xc
   1cc54:	7ff77713          	andi	a4,a4,2047
   1cc58:	00c45413          	srli	s0,s0,0xc
   1cc5c:	01f9d993          	srli	s3,s3,0x1f
   1cc60:	2c070063          	beqz	a4,1cf20 <__muldf3+0x348>
   1cc64:	7ff00793          	li	a5,2047
   1cc68:	32f70463          	beq	a4,a5,1cf90 <__muldf3+0x3b8>
   1cc6c:	00341413          	slli	s0,s0,0x3
   1cc70:	01db5793          	srli	a5,s6,0x1d
   1cc74:	0087e7b3          	or	a5,a5,s0
   1cc78:	00800437          	lui	s0,0x800
   1cc7c:	0087e433          	or	s0,a5,s0
   1cc80:	c0170693          	addi	a3,a4,-1023
   1cc84:	003b1793          	slli	a5,s6,0x3
   1cc88:	00000713          	li	a4,0
   1cc8c:	00da8ab3          	add	s5,s5,a3
   1cc90:	002b9693          	slli	a3,s7,0x2
   1cc94:	00e6e6b3          	or	a3,a3,a4
   1cc98:	00a00613          	li	a2,10
   1cc9c:	001a8513          	addi	a0,s5,1
   1cca0:	40d64663          	blt	a2,a3,1d0ac <__muldf3+0x4d4>
   1cca4:	00200613          	li	a2,2
   1cca8:	013a45b3          	xor	a1,s4,s3
   1ccac:	30d64e63          	blt	a2,a3,1cfc8 <__muldf3+0x3f0>
   1ccb0:	fff68693          	addi	a3,a3,-1
   1ccb4:	00100613          	li	a2,1
   1ccb8:	32d67a63          	bgeu	a2,a3,1cfec <__muldf3+0x414>
   1ccbc:	00010337          	lui	t1,0x10
   1ccc0:	fff30e13          	addi	t3,t1,-1 # ffff <exit-0xb5>
   1ccc4:	01095713          	srli	a4,s2,0x10
   1ccc8:	0107d893          	srli	a7,a5,0x10
   1cccc:	01c97933          	and	s2,s2,t3
   1ccd0:	01c7ff33          	and	t5,a5,t3
   1ccd4:	03e907b3          	mul	a5,s2,t5
   1ccd8:	03e70eb3          	mul	t4,a4,t5
   1ccdc:	0107d813          	srli	a6,a5,0x10
   1cce0:	03288633          	mul	a2,a7,s2
   1cce4:	01d60633          	add	a2,a2,t4
   1cce8:	00c80833          	add	a6,a6,a2
   1ccec:	031706b3          	mul	a3,a4,a7
   1ccf0:	01d87463          	bgeu	a6,t4,1ccf8 <__muldf3+0x120>
   1ccf4:	006686b3          	add	a3,a3,t1
   1ccf8:	01085293          	srli	t0,a6,0x10
   1ccfc:	01c87833          	and	a6,a6,t3
   1cd00:	01c7f7b3          	and	a5,a5,t3
   1cd04:	01045613          	srli	a2,s0,0x10
   1cd08:	01c47e33          	and	t3,s0,t3
   1cd0c:	01081813          	slli	a6,a6,0x10
   1cd10:	00f80833          	add	a6,a6,a5
   1cd14:	03c90eb3          	mul	t4,s2,t3
   1cd18:	03c707b3          	mul	a5,a4,t3
   1cd1c:	03260933          	mul	s2,a2,s2
   1cd20:	02c70333          	mul	t1,a4,a2
   1cd24:	00f90933          	add	s2,s2,a5
   1cd28:	010ed713          	srli	a4,t4,0x10
   1cd2c:	01270733          	add	a4,a4,s2
   1cd30:	00f77663          	bgeu	a4,a5,1cd3c <__muldf3+0x164>
   1cd34:	000107b7          	lui	a5,0x10
   1cd38:	00f30333          	add	t1,t1,a5
   1cd3c:	00010437          	lui	s0,0x10
   1cd40:	01075793          	srli	a5,a4,0x10
   1cd44:	fff40f93          	addi	t6,s0,-1 # ffff <exit-0xb5>
   1cd48:	00678333          	add	t1,a5,t1
   1cd4c:	01f777b3          	and	a5,a4,t6
   1cd50:	01fefeb3          	and	t4,t4,t6
   1cd54:	01079793          	slli	a5,a5,0x10
   1cd58:	01f4ffb3          	and	t6,s1,t6
   1cd5c:	01d787b3          	add	a5,a5,t4
   1cd60:	0104de93          	srli	t4,s1,0x10
   1cd64:	03ff03b3          	mul	t2,t5,t6
   1cd68:	00f282b3          	add	t0,t0,a5
   1cd6c:	03ee8f33          	mul	t5,t4,t5
   1cd70:	0103d713          	srli	a4,t2,0x10
   1cd74:	03d884b3          	mul	s1,a7,t4
   1cd78:	03f888b3          	mul	a7,a7,t6
   1cd7c:	01e888b3          	add	a7,a7,t5
   1cd80:	01170733          	add	a4,a4,a7
   1cd84:	01e77463          	bgeu	a4,t5,1cd8c <__muldf3+0x1b4>
   1cd88:	008484b3          	add	s1,s1,s0
   1cd8c:	01075f13          	srli	t5,a4,0x10
   1cd90:	009f0f33          	add	t5,t5,s1
   1cd94:	000104b7          	lui	s1,0x10
   1cd98:	fff48413          	addi	s0,s1,-1 # ffff <exit-0xb5>
   1cd9c:	00877733          	and	a4,a4,s0
   1cda0:	0083f3b3          	and	t2,t2,s0
   1cda4:	01071713          	slli	a4,a4,0x10
   1cda8:	007708b3          	add	a7,a4,t2
   1cdac:	03fe03b3          	mul	t2,t3,t6
   1cdb0:	03ce8e33          	mul	t3,t4,t3
   1cdb4:	03d60eb3          	mul	t4,a2,t4
   1cdb8:	03f60633          	mul	a2,a2,t6
   1cdbc:	0103df93          	srli	t6,t2,0x10
   1cdc0:	01c60633          	add	a2,a2,t3
   1cdc4:	00cf8fb3          	add	t6,t6,a2
   1cdc8:	01cff463          	bgeu	t6,t3,1cdd0 <__muldf3+0x1f8>
   1cdcc:	009e8eb3          	add	t4,t4,s1
   1cdd0:	008ff733          	and	a4,t6,s0
   1cdd4:	0083f3b3          	and	t2,t2,s0
   1cdd8:	01071713          	slli	a4,a4,0x10
   1cddc:	005686b3          	add	a3,a3,t0
   1cde0:	00770733          	add	a4,a4,t2
   1cde4:	00670333          	add	t1,a4,t1
   1cde8:	00f6b7b3          	sltu	a5,a3,a5
   1cdec:	00f307b3          	add	a5,t1,a5
   1cdf0:	00e33633          	sltu	a2,t1,a4
   1cdf4:	011688b3          	add	a7,a3,a7
   1cdf8:	0067b333          	sltu	t1,a5,t1
   1cdfc:	00666633          	or	a2,a2,t1
   1ce00:	00d8b6b3          	sltu	a3,a7,a3
   1ce04:	01e78333          	add	t1,a5,t5
   1ce08:	00d306b3          	add	a3,t1,a3
   1ce0c:	00f33733          	sltu	a4,t1,a5
   1ce10:	010fdf93          	srli	t6,t6,0x10
   1ce14:	0066b333          	sltu	t1,a3,t1
   1ce18:	00989793          	slli	a5,a7,0x9
   1ce1c:	01f60633          	add	a2,a2,t6
   1ce20:	00676733          	or	a4,a4,t1
   1ce24:	00c70733          	add	a4,a4,a2
   1ce28:	0107e7b3          	or	a5,a5,a6
   1ce2c:	01d70733          	add	a4,a4,t4
   1ce30:	00f037b3          	snez	a5,a5
   1ce34:	0178d893          	srli	a7,a7,0x17
   1ce38:	00971713          	slli	a4,a4,0x9
   1ce3c:	0176d413          	srli	s0,a3,0x17
   1ce40:	0117e7b3          	or	a5,a5,a7
   1ce44:	00969693          	slli	a3,a3,0x9
   1ce48:	00d7e7b3          	or	a5,a5,a3
   1ce4c:	00771693          	slli	a3,a4,0x7
   1ce50:	00876433          	or	s0,a4,s0
   1ce54:	0206d063          	bgez	a3,1ce74 <__muldf3+0x29c>
   1ce58:	0017d713          	srli	a4,a5,0x1
   1ce5c:	0017f793          	andi	a5,a5,1
   1ce60:	00f76733          	or	a4,a4,a5
   1ce64:	01f41793          	slli	a5,s0,0x1f
   1ce68:	00f767b3          	or	a5,a4,a5
   1ce6c:	00145413          	srli	s0,s0,0x1
   1ce70:	00050a93          	mv	s5,a0
   1ce74:	000a8513          	mv	a0,s5
   1ce78:	18c0006f          	j	1d004 <__muldf3+0x42c>
   1ce7c:	00a4e933          	or	s2,s1,a0
   1ce80:	08090063          	beqz	s2,1cf00 <__muldf3+0x328>
   1ce84:	04048063          	beqz	s1,1cec4 <__muldf3+0x2ec>
   1ce88:	00048513          	mv	a0,s1
   1ce8c:	0b1030ef          	jal	2073c <__clzsi2>
   1ce90:	ff550713          	addi	a4,a0,-11 # 7fff5 <__BSS_END__+0x5d1f5>
   1ce94:	01d00793          	li	a5,29
   1ce98:	ff850693          	addi	a3,a0,-8
   1ce9c:	40e787b3          	sub	a5,a5,a4
   1cea0:	00d494b3          	sll	s1,s1,a3
   1cea4:	00f457b3          	srl	a5,s0,a5
   1cea8:	0097e7b3          	or	a5,a5,s1
   1ceac:	00d414b3          	sll	s1,s0,a3
   1ceb0:	c0d00a93          	li	s5,-1011
   1ceb4:	00048913          	mv	s2,s1
   1ceb8:	40aa8ab3          	sub	s5,s5,a0
   1cebc:	00078493          	mv	s1,a5
   1cec0:	d89ff06f          	j	1cc48 <__muldf3+0x70>
   1cec4:	079030ef          	jal	2073c <__clzsi2>
   1cec8:	00050793          	mv	a5,a0
   1cecc:	01578713          	addi	a4,a5,21 # 10015 <exit-0x9f>
   1ced0:	01c00693          	li	a3,28
   1ced4:	02050513          	addi	a0,a0,32
   1ced8:	fae6dee3          	bge	a3,a4,1ce94 <__muldf3+0x2bc>
   1cedc:	ff878793          	addi	a5,a5,-8
   1cee0:	00f417b3          	sll	a5,s0,a5
   1cee4:	fcdff06f          	j	1ceb0 <__muldf3+0x2d8>
   1cee8:	00a4e933          	or	s2,s1,a0
   1ceec:	02091263          	bnez	s2,1cf10 <__muldf3+0x338>
   1cef0:	00000493          	li	s1,0
   1cef4:	7ff00a93          	li	s5,2047
   1cef8:	00200b93          	li	s7,2
   1cefc:	d51ff06f          	j	1cc4c <__muldf3+0x74>
   1cf00:	00000493          	li	s1,0
   1cf04:	00000a93          	li	s5,0
   1cf08:	00100b93          	li	s7,1
   1cf0c:	d41ff06f          	j	1cc4c <__muldf3+0x74>
   1cf10:	00050913          	mv	s2,a0
   1cf14:	7ff00a93          	li	s5,2047
   1cf18:	00300b93          	li	s7,3
   1cf1c:	d31ff06f          	j	1cc4c <__muldf3+0x74>
   1cf20:	016467b3          	or	a5,s0,s6
   1cf24:	08078263          	beqz	a5,1cfa8 <__muldf3+0x3d0>
   1cf28:	04040063          	beqz	s0,1cf68 <__muldf3+0x390>
   1cf2c:	00040513          	mv	a0,s0
   1cf30:	00d030ef          	jal	2073c <__clzsi2>
   1cf34:	ff550693          	addi	a3,a0,-11
   1cf38:	01d00713          	li	a4,29
   1cf3c:	ff850793          	addi	a5,a0,-8
   1cf40:	40d70733          	sub	a4,a4,a3
   1cf44:	00f41433          	sll	s0,s0,a5
   1cf48:	00eb5733          	srl	a4,s6,a4
   1cf4c:	00876733          	or	a4,a4,s0
   1cf50:	00fb1433          	sll	s0,s6,a5
   1cf54:	c0d00693          	li	a3,-1011
   1cf58:	00040793          	mv	a5,s0
   1cf5c:	40a686b3          	sub	a3,a3,a0
   1cf60:	00070413          	mv	s0,a4
   1cf64:	d25ff06f          	j	1cc88 <__muldf3+0xb0>
   1cf68:	000b0513          	mv	a0,s6
   1cf6c:	7d0030ef          	jal	2073c <__clzsi2>
   1cf70:	00050793          	mv	a5,a0
   1cf74:	01578693          	addi	a3,a5,21
   1cf78:	01c00713          	li	a4,28
   1cf7c:	02050513          	addi	a0,a0,32
   1cf80:	fad75ce3          	bge	a4,a3,1cf38 <__muldf3+0x360>
   1cf84:	ff878793          	addi	a5,a5,-8
   1cf88:	00fb1733          	sll	a4,s6,a5
   1cf8c:	fc9ff06f          	j	1cf54 <__muldf3+0x37c>
   1cf90:	016467b3          	or	a5,s0,s6
   1cf94:	02079263          	bnez	a5,1cfb8 <__muldf3+0x3e0>
   1cf98:	00000413          	li	s0,0
   1cf9c:	7ff00693          	li	a3,2047
   1cfa0:	00200713          	li	a4,2
   1cfa4:	ce9ff06f          	j	1cc8c <__muldf3+0xb4>
   1cfa8:	00000413          	li	s0,0
   1cfac:	00000693          	li	a3,0
   1cfb0:	00100713          	li	a4,1
   1cfb4:	cd9ff06f          	j	1cc8c <__muldf3+0xb4>
   1cfb8:	000b0793          	mv	a5,s6
   1cfbc:	7ff00693          	li	a3,2047
   1cfc0:	00300713          	li	a4,3
   1cfc4:	cc9ff06f          	j	1cc8c <__muldf3+0xb4>
   1cfc8:	00100613          	li	a2,1
   1cfcc:	00d61633          	sll	a2,a2,a3
   1cfd0:	53067693          	andi	a3,a2,1328
   1cfd4:	0e069663          	bnez	a3,1d0c0 <__muldf3+0x4e8>
   1cfd8:	24067813          	andi	a6,a2,576
   1cfdc:	1a081a63          	bnez	a6,1d190 <__muldf3+0x5b8>
   1cfe0:	08867613          	andi	a2,a2,136
   1cfe4:	cc060ce3          	beqz	a2,1ccbc <__muldf3+0xe4>
   1cfe8:	00098593          	mv	a1,s3
   1cfec:	00200693          	li	a3,2
   1cff0:	18d70863          	beq	a4,a3,1d180 <__muldf3+0x5a8>
   1cff4:	00300693          	li	a3,3
   1cff8:	1ad70463          	beq	a4,a3,1d1a0 <__muldf3+0x5c8>
   1cffc:	00100693          	li	a3,1
   1d000:	1ad70663          	beq	a4,a3,1d1ac <__muldf3+0x5d4>
   1d004:	3ff50613          	addi	a2,a0,1023
   1d008:	0cc05463          	blez	a2,1d0d0 <__muldf3+0x4f8>
   1d00c:	0077f713          	andi	a4,a5,7
   1d010:	02070063          	beqz	a4,1d030 <__muldf3+0x458>
   1d014:	00f7f713          	andi	a4,a5,15
   1d018:	00400693          	li	a3,4
   1d01c:	00d70a63          	beq	a4,a3,1d030 <__muldf3+0x458>
   1d020:	00478713          	addi	a4,a5,4
   1d024:	00f736b3          	sltu	a3,a4,a5
   1d028:	00d40433          	add	s0,s0,a3
   1d02c:	00070793          	mv	a5,a4
   1d030:	00741713          	slli	a4,s0,0x7
   1d034:	00075a63          	bgez	a4,1d048 <__muldf3+0x470>
   1d038:	ff000737          	lui	a4,0xff000
   1d03c:	fff70713          	addi	a4,a4,-1 # feffffff <__BSS_END__+0xfefdd1ff>
   1d040:	00e47433          	and	s0,s0,a4
   1d044:	40050613          	addi	a2,a0,1024
   1d048:	7fe00713          	li	a4,2046
   1d04c:	12c74a63          	blt	a4,a2,1d180 <__muldf3+0x5a8>
   1d050:	0037d793          	srli	a5,a5,0x3
   1d054:	01d41693          	slli	a3,s0,0x1d
   1d058:	00f6e6b3          	or	a3,a3,a5
   1d05c:	00345713          	srli	a4,s0,0x3
   1d060:	00c71713          	slli	a4,a4,0xc
   1d064:	02c12083          	lw	ra,44(sp)
   1d068:	02812403          	lw	s0,40(sp)
   1d06c:	01461613          	slli	a2,a2,0x14
   1d070:	00c75713          	srli	a4,a4,0xc
   1d074:	01f59593          	slli	a1,a1,0x1f
   1d078:	00e66633          	or	a2,a2,a4
   1d07c:	00b667b3          	or	a5,a2,a1
   1d080:	02412483          	lw	s1,36(sp)
   1d084:	02012903          	lw	s2,32(sp)
   1d088:	01c12983          	lw	s3,28(sp)
   1d08c:	01812a03          	lw	s4,24(sp)
   1d090:	01412a83          	lw	s5,20(sp)
   1d094:	01012b03          	lw	s6,16(sp)
   1d098:	00c12b83          	lw	s7,12(sp)
   1d09c:	00068513          	mv	a0,a3
   1d0a0:	00078593          	mv	a1,a5
   1d0a4:	03010113          	addi	sp,sp,48
   1d0a8:	00008067          	ret
   1d0ac:	00f00613          	li	a2,15
   1d0b0:	0ec68863          	beq	a3,a2,1d1a0 <__muldf3+0x5c8>
   1d0b4:	00b00613          	li	a2,11
   1d0b8:	000a0593          	mv	a1,s4
   1d0bc:	f2c686e3          	beq	a3,a2,1cfe8 <__muldf3+0x410>
   1d0c0:	00048413          	mv	s0,s1
   1d0c4:	00090793          	mv	a5,s2
   1d0c8:	000b8713          	mv	a4,s7
   1d0cc:	f21ff06f          	j	1cfec <__muldf3+0x414>
   1d0d0:	00100713          	li	a4,1
   1d0d4:	40c70733          	sub	a4,a4,a2
   1d0d8:	03800693          	li	a3,56
   1d0dc:	0ce6c863          	blt	a3,a4,1d1ac <__muldf3+0x5d4>
   1d0e0:	01f00693          	li	a3,31
   1d0e4:	06e6c463          	blt	a3,a4,1d14c <__muldf3+0x574>
   1d0e8:	41e50513          	addi	a0,a0,1054
   1d0ec:	00e7d633          	srl	a2,a5,a4
   1d0f0:	00a416b3          	sll	a3,s0,a0
   1d0f4:	00a79533          	sll	a0,a5,a0
   1d0f8:	00c6e6b3          	or	a3,a3,a2
   1d0fc:	00a03533          	snez	a0,a0
   1d100:	00a6e7b3          	or	a5,a3,a0
   1d104:	00e45733          	srl	a4,s0,a4
   1d108:	0077f693          	andi	a3,a5,7
   1d10c:	02068063          	beqz	a3,1d12c <__muldf3+0x554>
   1d110:	00f7f693          	andi	a3,a5,15
   1d114:	00400613          	li	a2,4
   1d118:	00c68a63          	beq	a3,a2,1d12c <__muldf3+0x554>
   1d11c:	00478693          	addi	a3,a5,4
   1d120:	00f6b633          	sltu	a2,a3,a5
   1d124:	00c70733          	add	a4,a4,a2
   1d128:	00068793          	mv	a5,a3
   1d12c:	00871693          	slli	a3,a4,0x8
   1d130:	0806c463          	bltz	a3,1d1b8 <__muldf3+0x5e0>
   1d134:	01d71693          	slli	a3,a4,0x1d
   1d138:	0037d793          	srli	a5,a5,0x3
   1d13c:	00f6e6b3          	or	a3,a3,a5
   1d140:	00375713          	srli	a4,a4,0x3
   1d144:	00000613          	li	a2,0
   1d148:	f19ff06f          	j	1d060 <__muldf3+0x488>
   1d14c:	fe100693          	li	a3,-31
   1d150:	40c686b3          	sub	a3,a3,a2
   1d154:	02000813          	li	a6,32
   1d158:	00d456b3          	srl	a3,s0,a3
   1d15c:	00000613          	li	a2,0
   1d160:	01070663          	beq	a4,a6,1d16c <__muldf3+0x594>
   1d164:	43e50613          	addi	a2,a0,1086
   1d168:	00c41633          	sll	a2,s0,a2
   1d16c:	00f66633          	or	a2,a2,a5
   1d170:	00c03633          	snez	a2,a2
   1d174:	00c6e7b3          	or	a5,a3,a2
   1d178:	00000713          	li	a4,0
   1d17c:	f8dff06f          	j	1d108 <__muldf3+0x530>
   1d180:	00000713          	li	a4,0
   1d184:	00000693          	li	a3,0
   1d188:	7ff00613          	li	a2,2047
   1d18c:	ed5ff06f          	j	1d060 <__muldf3+0x488>
   1d190:	00080737          	lui	a4,0x80
   1d194:	7ff00613          	li	a2,2047
   1d198:	00000593          	li	a1,0
   1d19c:	ec5ff06f          	j	1d060 <__muldf3+0x488>
   1d1a0:	00080737          	lui	a4,0x80
   1d1a4:	00000693          	li	a3,0
   1d1a8:	fedff06f          	j	1d194 <__muldf3+0x5bc>
   1d1ac:	00000713          	li	a4,0
   1d1b0:	00000693          	li	a3,0
   1d1b4:	f91ff06f          	j	1d144 <__muldf3+0x56c>
   1d1b8:	00000713          	li	a4,0
   1d1bc:	00000693          	li	a3,0
   1d1c0:	00100613          	li	a2,1
   1d1c4:	e9dff06f          	j	1d060 <__muldf3+0x488>

0001d1c8 <__subdf3>:
   1d1c8:	00100837          	lui	a6,0x100
   1d1cc:	fff80813          	addi	a6,a6,-1 # fffff <__BSS_END__+0xdd1ff>
   1d1d0:	fe010113          	addi	sp,sp,-32
   1d1d4:	00b878b3          	and	a7,a6,a1
   1d1d8:	0145d713          	srli	a4,a1,0x14
   1d1dc:	01d55793          	srli	a5,a0,0x1d
   1d1e0:	00d87833          	and	a6,a6,a3
   1d1e4:	01212823          	sw	s2,16(sp)
   1d1e8:	7ff77913          	andi	s2,a4,2047
   1d1ec:	00389713          	slli	a4,a7,0x3
   1d1f0:	0146d893          	srli	a7,a3,0x14
   1d1f4:	00912a23          	sw	s1,20(sp)
   1d1f8:	00e7e7b3          	or	a5,a5,a4
   1d1fc:	01f5d493          	srli	s1,a1,0x1f
   1d200:	01d65713          	srli	a4,a2,0x1d
   1d204:	00381813          	slli	a6,a6,0x3
   1d208:	00112e23          	sw	ra,28(sp)
   1d20c:	00812c23          	sw	s0,24(sp)
   1d210:	01312623          	sw	s3,12(sp)
   1d214:	7ff8f893          	andi	a7,a7,2047
   1d218:	7ff00593          	li	a1,2047
   1d21c:	00351513          	slli	a0,a0,0x3
   1d220:	01f6d693          	srli	a3,a3,0x1f
   1d224:	01076733          	or	a4,a4,a6
   1d228:	00361613          	slli	a2,a2,0x3
   1d22c:	00b89663          	bne	a7,a1,1d238 <__subdf3+0x70>
   1d230:	00c765b3          	or	a1,a4,a2
   1d234:	00059463          	bnez	a1,1d23c <__subdf3+0x74>
   1d238:	0016c693          	xori	a3,a3,1
   1d23c:	41190833          	sub	a6,s2,a7
   1d240:	2a969a63          	bne	a3,s1,1d4f4 <__subdf3+0x32c>
   1d244:	11005c63          	blez	a6,1d35c <__subdf3+0x194>
   1d248:	04089063          	bnez	a7,1d288 <__subdf3+0xc0>
   1d24c:	00c766b3          	or	a3,a4,a2
   1d250:	66068063          	beqz	a3,1d8b0 <__subdf3+0x6e8>
   1d254:	fff80593          	addi	a1,a6,-1
   1d258:	02059063          	bnez	a1,1d278 <__subdf3+0xb0>
   1d25c:	00c50633          	add	a2,a0,a2
   1d260:	00a636b3          	sltu	a3,a2,a0
   1d264:	00e78733          	add	a4,a5,a4
   1d268:	00060513          	mv	a0,a2
   1d26c:	00d707b3          	add	a5,a4,a3
   1d270:	00100913          	li	s2,1
   1d274:	06c0006f          	j	1d2e0 <__subdf3+0x118>
   1d278:	7ff00693          	li	a3,2047
   1d27c:	02d81063          	bne	a6,a3,1d29c <__subdf3+0xd4>
   1d280:	7ff00913          	li	s2,2047
   1d284:	1f80006f          	j	1d47c <__subdf3+0x2b4>
   1d288:	7ff00693          	li	a3,2047
   1d28c:	1ed90863          	beq	s2,a3,1d47c <__subdf3+0x2b4>
   1d290:	008006b7          	lui	a3,0x800
   1d294:	00d76733          	or	a4,a4,a3
   1d298:	00080593          	mv	a1,a6
   1d29c:	03800693          	li	a3,56
   1d2a0:	0ab6c863          	blt	a3,a1,1d350 <__subdf3+0x188>
   1d2a4:	01f00693          	li	a3,31
   1d2a8:	06b6ca63          	blt	a3,a1,1d31c <__subdf3+0x154>
   1d2ac:	02000813          	li	a6,32
   1d2b0:	40b80833          	sub	a6,a6,a1
   1d2b4:	010716b3          	sll	a3,a4,a6
   1d2b8:	00b658b3          	srl	a7,a2,a1
   1d2bc:	01061833          	sll	a6,a2,a6
   1d2c0:	0116e6b3          	or	a3,a3,a7
   1d2c4:	01003833          	snez	a6,a6
   1d2c8:	0106e6b3          	or	a3,a3,a6
   1d2cc:	00b755b3          	srl	a1,a4,a1
   1d2d0:	00a68533          	add	a0,a3,a0
   1d2d4:	00f585b3          	add	a1,a1,a5
   1d2d8:	00d536b3          	sltu	a3,a0,a3
   1d2dc:	00d587b3          	add	a5,a1,a3
   1d2e0:	00879713          	slli	a4,a5,0x8
   1d2e4:	18075c63          	bgez	a4,1d47c <__subdf3+0x2b4>
   1d2e8:	00190913          	addi	s2,s2,1
   1d2ec:	7ff00713          	li	a4,2047
   1d2f0:	5ae90a63          	beq	s2,a4,1d8a4 <__subdf3+0x6dc>
   1d2f4:	ff800737          	lui	a4,0xff800
   1d2f8:	fff70713          	addi	a4,a4,-1 # ff7fffff <__BSS_END__+0xff7dd1ff>
   1d2fc:	00e7f733          	and	a4,a5,a4
   1d300:	00155793          	srli	a5,a0,0x1
   1d304:	00157513          	andi	a0,a0,1
   1d308:	00a7e7b3          	or	a5,a5,a0
   1d30c:	01f71513          	slli	a0,a4,0x1f
   1d310:	00f56533          	or	a0,a0,a5
   1d314:	00175793          	srli	a5,a4,0x1
   1d318:	1640006f          	j	1d47c <__subdf3+0x2b4>
   1d31c:	fe058693          	addi	a3,a1,-32
   1d320:	02000893          	li	a7,32
   1d324:	00d756b3          	srl	a3,a4,a3
   1d328:	00000813          	li	a6,0
   1d32c:	01158863          	beq	a1,a7,1d33c <__subdf3+0x174>
   1d330:	04000813          	li	a6,64
   1d334:	40b80833          	sub	a6,a6,a1
   1d338:	01071833          	sll	a6,a4,a6
   1d33c:	00c86833          	or	a6,a6,a2
   1d340:	01003833          	snez	a6,a6
   1d344:	0106e6b3          	or	a3,a3,a6
   1d348:	00000593          	li	a1,0
   1d34c:	f85ff06f          	j	1d2d0 <__subdf3+0x108>
   1d350:	00c766b3          	or	a3,a4,a2
   1d354:	00d036b3          	snez	a3,a3
   1d358:	ff1ff06f          	j	1d348 <__subdf3+0x180>
   1d35c:	0c080a63          	beqz	a6,1d430 <__subdf3+0x268>
   1d360:	412886b3          	sub	a3,a7,s2
   1d364:	02091463          	bnez	s2,1d38c <__subdf3+0x1c4>
   1d368:	00a7e5b3          	or	a1,a5,a0
   1d36c:	50058e63          	beqz	a1,1d888 <__subdf3+0x6c0>
   1d370:	fff68593          	addi	a1,a3,-1 # 7fffff <__BSS_END__+0x7dd1ff>
   1d374:	ee0584e3          	beqz	a1,1d25c <__subdf3+0x94>
   1d378:	7ff00813          	li	a6,2047
   1d37c:	03069263          	bne	a3,a6,1d3a0 <__subdf3+0x1d8>
   1d380:	00070793          	mv	a5,a4
   1d384:	00060513          	mv	a0,a2
   1d388:	ef9ff06f          	j	1d280 <__subdf3+0xb8>
   1d38c:	7ff00593          	li	a1,2047
   1d390:	feb888e3          	beq	a7,a1,1d380 <__subdf3+0x1b8>
   1d394:	008005b7          	lui	a1,0x800
   1d398:	00b7e7b3          	or	a5,a5,a1
   1d39c:	00068593          	mv	a1,a3
   1d3a0:	03800693          	li	a3,56
   1d3a4:	08b6c063          	blt	a3,a1,1d424 <__subdf3+0x25c>
   1d3a8:	01f00693          	li	a3,31
   1d3ac:	04b6c263          	blt	a3,a1,1d3f0 <__subdf3+0x228>
   1d3b0:	02000813          	li	a6,32
   1d3b4:	40b80833          	sub	a6,a6,a1
   1d3b8:	010796b3          	sll	a3,a5,a6
   1d3bc:	00b55333          	srl	t1,a0,a1
   1d3c0:	01051833          	sll	a6,a0,a6
   1d3c4:	0066e6b3          	or	a3,a3,t1
   1d3c8:	01003833          	snez	a6,a6
   1d3cc:	0106e6b3          	or	a3,a3,a6
   1d3d0:	00b7d5b3          	srl	a1,a5,a1
   1d3d4:	00c68633          	add	a2,a3,a2
   1d3d8:	00e585b3          	add	a1,a1,a4
   1d3dc:	00d636b3          	sltu	a3,a2,a3
   1d3e0:	00060513          	mv	a0,a2
   1d3e4:	00d587b3          	add	a5,a1,a3
   1d3e8:	00088913          	mv	s2,a7
   1d3ec:	ef5ff06f          	j	1d2e0 <__subdf3+0x118>
   1d3f0:	fe058693          	addi	a3,a1,-32 # 7fffe0 <__BSS_END__+0x7dd1e0>
   1d3f4:	02000313          	li	t1,32
   1d3f8:	00d7d6b3          	srl	a3,a5,a3
   1d3fc:	00000813          	li	a6,0
   1d400:	00658863          	beq	a1,t1,1d410 <__subdf3+0x248>
   1d404:	04000813          	li	a6,64
   1d408:	40b80833          	sub	a6,a6,a1
   1d40c:	01079833          	sll	a6,a5,a6
   1d410:	00a86833          	or	a6,a6,a0
   1d414:	01003833          	snez	a6,a6
   1d418:	0106e6b3          	or	a3,a3,a6
   1d41c:	00000593          	li	a1,0
   1d420:	fb5ff06f          	j	1d3d4 <__subdf3+0x20c>
   1d424:	00a7e6b3          	or	a3,a5,a0
   1d428:	00d036b3          	snez	a3,a3
   1d42c:	ff1ff06f          	j	1d41c <__subdf3+0x254>
   1d430:	00190693          	addi	a3,s2,1
   1d434:	7fe6f593          	andi	a1,a3,2046
   1d438:	08059663          	bnez	a1,1d4c4 <__subdf3+0x2fc>
   1d43c:	00a7e6b3          	or	a3,a5,a0
   1d440:	06091263          	bnez	s2,1d4a4 <__subdf3+0x2dc>
   1d444:	44068863          	beqz	a3,1d894 <__subdf3+0x6cc>
   1d448:	00c766b3          	or	a3,a4,a2
   1d44c:	02068863          	beqz	a3,1d47c <__subdf3+0x2b4>
   1d450:	00c50633          	add	a2,a0,a2
   1d454:	00a636b3          	sltu	a3,a2,a0
   1d458:	00e78733          	add	a4,a5,a4
   1d45c:	00d707b3          	add	a5,a4,a3
   1d460:	00879713          	slli	a4,a5,0x8
   1d464:	00060513          	mv	a0,a2
   1d468:	00075a63          	bgez	a4,1d47c <__subdf3+0x2b4>
   1d46c:	ff800737          	lui	a4,0xff800
   1d470:	fff70713          	addi	a4,a4,-1 # ff7fffff <__BSS_END__+0xff7dd1ff>
   1d474:	00e7f7b3          	and	a5,a5,a4
   1d478:	00100913          	li	s2,1
   1d47c:	00757713          	andi	a4,a0,7
   1d480:	44070863          	beqz	a4,1d8d0 <__subdf3+0x708>
   1d484:	00f57713          	andi	a4,a0,15
   1d488:	00400693          	li	a3,4
   1d48c:	44d70263          	beq	a4,a3,1d8d0 <__subdf3+0x708>
   1d490:	00450713          	addi	a4,a0,4
   1d494:	00a736b3          	sltu	a3,a4,a0
   1d498:	00d787b3          	add	a5,a5,a3
   1d49c:	00070513          	mv	a0,a4
   1d4a0:	4300006f          	j	1d8d0 <__subdf3+0x708>
   1d4a4:	ec068ee3          	beqz	a3,1d380 <__subdf3+0x1b8>
   1d4a8:	00c76633          	or	a2,a4,a2
   1d4ac:	dc060ae3          	beqz	a2,1d280 <__subdf3+0xb8>
   1d4b0:	00000493          	li	s1,0
   1d4b4:	004007b7          	lui	a5,0x400
   1d4b8:	00000513          	li	a0,0
   1d4bc:	7ff00913          	li	s2,2047
   1d4c0:	4100006f          	j	1d8d0 <__subdf3+0x708>
   1d4c4:	7ff00593          	li	a1,2047
   1d4c8:	3cb68c63          	beq	a3,a1,1d8a0 <__subdf3+0x6d8>
   1d4cc:	00c50633          	add	a2,a0,a2
   1d4d0:	00a63533          	sltu	a0,a2,a0
   1d4d4:	00e78733          	add	a4,a5,a4
   1d4d8:	00a70733          	add	a4,a4,a0
   1d4dc:	01f71513          	slli	a0,a4,0x1f
   1d4e0:	00165613          	srli	a2,a2,0x1
   1d4e4:	00c56533          	or	a0,a0,a2
   1d4e8:	00175793          	srli	a5,a4,0x1
   1d4ec:	00068913          	mv	s2,a3
   1d4f0:	f8dff06f          	j	1d47c <__subdf3+0x2b4>
   1d4f4:	0f005c63          	blez	a6,1d5ec <__subdf3+0x424>
   1d4f8:	08089e63          	bnez	a7,1d594 <__subdf3+0x3cc>
   1d4fc:	00c766b3          	or	a3,a4,a2
   1d500:	3a068863          	beqz	a3,1d8b0 <__subdf3+0x6e8>
   1d504:	fff80693          	addi	a3,a6,-1
   1d508:	02069063          	bnez	a3,1d528 <__subdf3+0x360>
   1d50c:	40c50633          	sub	a2,a0,a2
   1d510:	00c536b3          	sltu	a3,a0,a2
   1d514:	40e78733          	sub	a4,a5,a4
   1d518:	00060513          	mv	a0,a2
   1d51c:	40d707b3          	sub	a5,a4,a3
   1d520:	00100913          	li	s2,1
   1d524:	0540006f          	j	1d578 <__subdf3+0x3b0>
   1d528:	7ff00593          	li	a1,2047
   1d52c:	d4b80ae3          	beq	a6,a1,1d280 <__subdf3+0xb8>
   1d530:	03800593          	li	a1,56
   1d534:	0ad5c663          	blt	a1,a3,1d5e0 <__subdf3+0x418>
   1d538:	01f00593          	li	a1,31
   1d53c:	06d5c863          	blt	a1,a3,1d5ac <__subdf3+0x3e4>
   1d540:	02000813          	li	a6,32
   1d544:	40d80833          	sub	a6,a6,a3
   1d548:	00d658b3          	srl	a7,a2,a3
   1d54c:	010715b3          	sll	a1,a4,a6
   1d550:	01061833          	sll	a6,a2,a6
   1d554:	0115e5b3          	or	a1,a1,a7
   1d558:	01003833          	snez	a6,a6
   1d55c:	0105e633          	or	a2,a1,a6
   1d560:	00d756b3          	srl	a3,a4,a3
   1d564:	40c50633          	sub	a2,a0,a2
   1d568:	00c53733          	sltu	a4,a0,a2
   1d56c:	40d786b3          	sub	a3,a5,a3
   1d570:	00060513          	mv	a0,a2
   1d574:	40e687b3          	sub	a5,a3,a4
   1d578:	00879713          	slli	a4,a5,0x8
   1d57c:	f00750e3          	bgez	a4,1d47c <__subdf3+0x2b4>
   1d580:	00800437          	lui	s0,0x800
   1d584:	fff40413          	addi	s0,s0,-1 # 7fffff <__BSS_END__+0x7dd1ff>
   1d588:	0087f433          	and	s0,a5,s0
   1d58c:	00050993          	mv	s3,a0
   1d590:	2100006f          	j	1d7a0 <__subdf3+0x5d8>
   1d594:	7ff00693          	li	a3,2047
   1d598:	eed902e3          	beq	s2,a3,1d47c <__subdf3+0x2b4>
   1d59c:	008006b7          	lui	a3,0x800
   1d5a0:	00d76733          	or	a4,a4,a3
   1d5a4:	00080693          	mv	a3,a6
   1d5a8:	f89ff06f          	j	1d530 <__subdf3+0x368>
   1d5ac:	fe068593          	addi	a1,a3,-32 # 7fffe0 <__BSS_END__+0x7dd1e0>
   1d5b0:	02000893          	li	a7,32
   1d5b4:	00b755b3          	srl	a1,a4,a1
   1d5b8:	00000813          	li	a6,0
   1d5bc:	01168863          	beq	a3,a7,1d5cc <__subdf3+0x404>
   1d5c0:	04000813          	li	a6,64
   1d5c4:	40d80833          	sub	a6,a6,a3
   1d5c8:	01071833          	sll	a6,a4,a6
   1d5cc:	00c86833          	or	a6,a6,a2
   1d5d0:	01003833          	snez	a6,a6
   1d5d4:	0105e633          	or	a2,a1,a6
   1d5d8:	00000693          	li	a3,0
   1d5dc:	f89ff06f          	j	1d564 <__subdf3+0x39c>
   1d5e0:	00c76633          	or	a2,a4,a2
   1d5e4:	00c03633          	snez	a2,a2
   1d5e8:	ff1ff06f          	j	1d5d8 <__subdf3+0x410>
   1d5ec:	0e080863          	beqz	a6,1d6dc <__subdf3+0x514>
   1d5f0:	41288833          	sub	a6,a7,s2
   1d5f4:	04091263          	bnez	s2,1d638 <__subdf3+0x470>
   1d5f8:	00a7e5b3          	or	a1,a5,a0
   1d5fc:	2a058e63          	beqz	a1,1d8b8 <__subdf3+0x6f0>
   1d600:	fff80593          	addi	a1,a6,-1
   1d604:	00059e63          	bnez	a1,1d620 <__subdf3+0x458>
   1d608:	40a60533          	sub	a0,a2,a0
   1d60c:	40f70733          	sub	a4,a4,a5
   1d610:	00a63633          	sltu	a2,a2,a0
   1d614:	40c707b3          	sub	a5,a4,a2
   1d618:	00068493          	mv	s1,a3
   1d61c:	f05ff06f          	j	1d520 <__subdf3+0x358>
   1d620:	7ff00313          	li	t1,2047
   1d624:	02681463          	bne	a6,t1,1d64c <__subdf3+0x484>
   1d628:	00070793          	mv	a5,a4
   1d62c:	00060513          	mv	a0,a2
   1d630:	7ff00913          	li	s2,2047
   1d634:	0d00006f          	j	1d704 <__subdf3+0x53c>
   1d638:	7ff00593          	li	a1,2047
   1d63c:	feb886e3          	beq	a7,a1,1d628 <__subdf3+0x460>
   1d640:	008005b7          	lui	a1,0x800
   1d644:	00b7e7b3          	or	a5,a5,a1
   1d648:	00080593          	mv	a1,a6
   1d64c:	03800813          	li	a6,56
   1d650:	08b84063          	blt	a6,a1,1d6d0 <__subdf3+0x508>
   1d654:	01f00813          	li	a6,31
   1d658:	04b84263          	blt	a6,a1,1d69c <__subdf3+0x4d4>
   1d65c:	02000313          	li	t1,32
   1d660:	40b30333          	sub	t1,t1,a1
   1d664:	00b55e33          	srl	t3,a0,a1
   1d668:	00679833          	sll	a6,a5,t1
   1d66c:	00651333          	sll	t1,a0,t1
   1d670:	01c86833          	or	a6,a6,t3
   1d674:	00603333          	snez	t1,t1
   1d678:	00686533          	or	a0,a6,t1
   1d67c:	00b7d5b3          	srl	a1,a5,a1
   1d680:	40a60533          	sub	a0,a2,a0
   1d684:	40b705b3          	sub	a1,a4,a1
   1d688:	00a63633          	sltu	a2,a2,a0
   1d68c:	40c587b3          	sub	a5,a1,a2
   1d690:	00088913          	mv	s2,a7
   1d694:	00068493          	mv	s1,a3
   1d698:	ee1ff06f          	j	1d578 <__subdf3+0x3b0>
   1d69c:	fe058813          	addi	a6,a1,-32 # 7fffe0 <__BSS_END__+0x7dd1e0>
   1d6a0:	02000e13          	li	t3,32
   1d6a4:	0107d833          	srl	a6,a5,a6
   1d6a8:	00000313          	li	t1,0
   1d6ac:	01c58863          	beq	a1,t3,1d6bc <__subdf3+0x4f4>
   1d6b0:	04000313          	li	t1,64
   1d6b4:	40b30333          	sub	t1,t1,a1
   1d6b8:	00679333          	sll	t1,a5,t1
   1d6bc:	00a36333          	or	t1,t1,a0
   1d6c0:	00603333          	snez	t1,t1
   1d6c4:	00686533          	or	a0,a6,t1
   1d6c8:	00000593          	li	a1,0
   1d6cc:	fb5ff06f          	j	1d680 <__subdf3+0x4b8>
   1d6d0:	00a7e533          	or	a0,a5,a0
   1d6d4:	00a03533          	snez	a0,a0
   1d6d8:	ff1ff06f          	j	1d6c8 <__subdf3+0x500>
   1d6dc:	00190593          	addi	a1,s2,1
   1d6e0:	7fe5f593          	andi	a1,a1,2046
   1d6e4:	08059663          	bnez	a1,1d770 <__subdf3+0x5a8>
   1d6e8:	00c765b3          	or	a1,a4,a2
   1d6ec:	00a7e833          	or	a6,a5,a0
   1d6f0:	06091063          	bnez	s2,1d750 <__subdf3+0x588>
   1d6f4:	00081c63          	bnez	a6,1d70c <__subdf3+0x544>
   1d6f8:	10058e63          	beqz	a1,1d814 <__subdf3+0x64c>
   1d6fc:	00070793          	mv	a5,a4
   1d700:	00060513          	mv	a0,a2
   1d704:	00068493          	mv	s1,a3
   1d708:	d75ff06f          	j	1d47c <__subdf3+0x2b4>
   1d70c:	d60588e3          	beqz	a1,1d47c <__subdf3+0x2b4>
   1d710:	40c50833          	sub	a6,a0,a2
   1d714:	010538b3          	sltu	a7,a0,a6
   1d718:	40e785b3          	sub	a1,a5,a4
   1d71c:	411585b3          	sub	a1,a1,a7
   1d720:	00859893          	slli	a7,a1,0x8
   1d724:	0008dc63          	bgez	a7,1d73c <__subdf3+0x574>
   1d728:	40a60533          	sub	a0,a2,a0
   1d72c:	40f70733          	sub	a4,a4,a5
   1d730:	00a63633          	sltu	a2,a2,a0
   1d734:	40c707b3          	sub	a5,a4,a2
   1d738:	fcdff06f          	j	1d704 <__subdf3+0x53c>
   1d73c:	00b86533          	or	a0,a6,a1
   1d740:	18050463          	beqz	a0,1d8c8 <__subdf3+0x700>
   1d744:	00058793          	mv	a5,a1
   1d748:	00080513          	mv	a0,a6
   1d74c:	d31ff06f          	j	1d47c <__subdf3+0x2b4>
   1d750:	00081c63          	bnez	a6,1d768 <__subdf3+0x5a0>
   1d754:	d4058ee3          	beqz	a1,1d4b0 <__subdf3+0x2e8>
   1d758:	00070793          	mv	a5,a4
   1d75c:	00060513          	mv	a0,a2
   1d760:	00068493          	mv	s1,a3
   1d764:	b1dff06f          	j	1d280 <__subdf3+0xb8>
   1d768:	b0058ce3          	beqz	a1,1d280 <__subdf3+0xb8>
   1d76c:	d45ff06f          	j	1d4b0 <__subdf3+0x2e8>
   1d770:	40c505b3          	sub	a1,a0,a2
   1d774:	00b53833          	sltu	a6,a0,a1
   1d778:	40e78433          	sub	s0,a5,a4
   1d77c:	41040433          	sub	s0,s0,a6
   1d780:	00841813          	slli	a6,s0,0x8
   1d784:	00058993          	mv	s3,a1
   1d788:	08085063          	bgez	a6,1d808 <__subdf3+0x640>
   1d78c:	40a609b3          	sub	s3,a2,a0
   1d790:	40f70433          	sub	s0,a4,a5
   1d794:	01363633          	sltu	a2,a2,s3
   1d798:	40c40433          	sub	s0,s0,a2
   1d79c:	00068493          	mv	s1,a3
   1d7a0:	06040e63          	beqz	s0,1d81c <__subdf3+0x654>
   1d7a4:	00040513          	mv	a0,s0
   1d7a8:	795020ef          	jal	2073c <__clzsi2>
   1d7ac:	ff850693          	addi	a3,a0,-8
   1d7b0:	02000793          	li	a5,32
   1d7b4:	40d787b3          	sub	a5,a5,a3
   1d7b8:	00d41433          	sll	s0,s0,a3
   1d7bc:	00f9d7b3          	srl	a5,s3,a5
   1d7c0:	0087e7b3          	or	a5,a5,s0
   1d7c4:	00d99433          	sll	s0,s3,a3
   1d7c8:	0b26c463          	blt	a3,s2,1d870 <__subdf3+0x6a8>
   1d7cc:	412686b3          	sub	a3,a3,s2
   1d7d0:	00168613          	addi	a2,a3,1
   1d7d4:	01f00713          	li	a4,31
   1d7d8:	06c74263          	blt	a4,a2,1d83c <__subdf3+0x674>
   1d7dc:	02000713          	li	a4,32
   1d7e0:	40c70733          	sub	a4,a4,a2
   1d7e4:	00e79533          	sll	a0,a5,a4
   1d7e8:	00c456b3          	srl	a3,s0,a2
   1d7ec:	00e41733          	sll	a4,s0,a4
   1d7f0:	00d56533          	or	a0,a0,a3
   1d7f4:	00e03733          	snez	a4,a4
   1d7f8:	00e56533          	or	a0,a0,a4
   1d7fc:	00c7d7b3          	srl	a5,a5,a2
   1d800:	00000913          	li	s2,0
   1d804:	c79ff06f          	j	1d47c <__subdf3+0x2b4>
   1d808:	0085e5b3          	or	a1,a1,s0
   1d80c:	f8059ae3          	bnez	a1,1d7a0 <__subdf3+0x5d8>
   1d810:	00000913          	li	s2,0
   1d814:	00000493          	li	s1,0
   1d818:	08c0006f          	j	1d8a4 <__subdf3+0x6dc>
   1d81c:	00098513          	mv	a0,s3
   1d820:	71d020ef          	jal	2073c <__clzsi2>
   1d824:	01850693          	addi	a3,a0,24
   1d828:	01f00793          	li	a5,31
   1d82c:	f8d7d2e3          	bge	a5,a3,1d7b0 <__subdf3+0x5e8>
   1d830:	ff850793          	addi	a5,a0,-8
   1d834:	00f997b3          	sll	a5,s3,a5
   1d838:	f91ff06f          	j	1d7c8 <__subdf3+0x600>
   1d83c:	fe168693          	addi	a3,a3,-31
   1d840:	00d7d533          	srl	a0,a5,a3
   1d844:	02000693          	li	a3,32
   1d848:	00000713          	li	a4,0
   1d84c:	00d60863          	beq	a2,a3,1d85c <__subdf3+0x694>
   1d850:	04000713          	li	a4,64
   1d854:	40c70733          	sub	a4,a4,a2
   1d858:	00e79733          	sll	a4,a5,a4
   1d85c:	00e46733          	or	a4,s0,a4
   1d860:	00e03733          	snez	a4,a4
   1d864:	00e56533          	or	a0,a0,a4
   1d868:	00000793          	li	a5,0
   1d86c:	f95ff06f          	j	1d800 <__subdf3+0x638>
   1d870:	ff800737          	lui	a4,0xff800
   1d874:	fff70713          	addi	a4,a4,-1 # ff7fffff <__BSS_END__+0xff7dd1ff>
   1d878:	40d90933          	sub	s2,s2,a3
   1d87c:	00e7f7b3          	and	a5,a5,a4
   1d880:	00040513          	mv	a0,s0
   1d884:	bf9ff06f          	j	1d47c <__subdf3+0x2b4>
   1d888:	00070793          	mv	a5,a4
   1d88c:	00060513          	mv	a0,a2
   1d890:	c5dff06f          	j	1d4ec <__subdf3+0x324>
   1d894:	00070793          	mv	a5,a4
   1d898:	00060513          	mv	a0,a2
   1d89c:	be1ff06f          	j	1d47c <__subdf3+0x2b4>
   1d8a0:	7ff00913          	li	s2,2047
   1d8a4:	00000793          	li	a5,0
   1d8a8:	00000513          	li	a0,0
   1d8ac:	0240006f          	j	1d8d0 <__subdf3+0x708>
   1d8b0:	00080913          	mv	s2,a6
   1d8b4:	bc9ff06f          	j	1d47c <__subdf3+0x2b4>
   1d8b8:	00070793          	mv	a5,a4
   1d8bc:	00060513          	mv	a0,a2
   1d8c0:	00080913          	mv	s2,a6
   1d8c4:	e41ff06f          	j	1d704 <__subdf3+0x53c>
   1d8c8:	00000793          	li	a5,0
   1d8cc:	00000493          	li	s1,0
   1d8d0:	00879713          	slli	a4,a5,0x8
   1d8d4:	00075e63          	bgez	a4,1d8f0 <__subdf3+0x728>
   1d8d8:	00190913          	addi	s2,s2,1
   1d8dc:	7ff00713          	li	a4,2047
   1d8e0:	08e90263          	beq	s2,a4,1d964 <__subdf3+0x79c>
   1d8e4:	ff800737          	lui	a4,0xff800
   1d8e8:	fff70713          	addi	a4,a4,-1 # ff7fffff <__BSS_END__+0xff7dd1ff>
   1d8ec:	00e7f7b3          	and	a5,a5,a4
   1d8f0:	01d79693          	slli	a3,a5,0x1d
   1d8f4:	00355513          	srli	a0,a0,0x3
   1d8f8:	7ff00713          	li	a4,2047
   1d8fc:	00a6e6b3          	or	a3,a3,a0
   1d900:	0037d793          	srli	a5,a5,0x3
   1d904:	00e91e63          	bne	s2,a4,1d920 <__subdf3+0x758>
   1d908:	00f6e6b3          	or	a3,a3,a5
   1d90c:	00000793          	li	a5,0
   1d910:	00068863          	beqz	a3,1d920 <__subdf3+0x758>
   1d914:	000807b7          	lui	a5,0x80
   1d918:	00000693          	li	a3,0
   1d91c:	00000493          	li	s1,0
   1d920:	01491713          	slli	a4,s2,0x14
   1d924:	7ff00637          	lui	a2,0x7ff00
   1d928:	00c79793          	slli	a5,a5,0xc
   1d92c:	00c77733          	and	a4,a4,a2
   1d930:	01c12083          	lw	ra,28(sp)
   1d934:	01812403          	lw	s0,24(sp)
   1d938:	00c7d793          	srli	a5,a5,0xc
   1d93c:	00f767b3          	or	a5,a4,a5
   1d940:	01f49713          	slli	a4,s1,0x1f
   1d944:	00e7e633          	or	a2,a5,a4
   1d948:	01412483          	lw	s1,20(sp)
   1d94c:	01012903          	lw	s2,16(sp)
   1d950:	00c12983          	lw	s3,12(sp)
   1d954:	00068513          	mv	a0,a3
   1d958:	00060593          	mv	a1,a2
   1d95c:	02010113          	addi	sp,sp,32
   1d960:	00008067          	ret
   1d964:	00000793          	li	a5,0
   1d968:	00000513          	li	a0,0
   1d96c:	f85ff06f          	j	1d8f0 <__subdf3+0x728>

0001d970 <__fixdfsi>:
   1d970:	0145d713          	srli	a4,a1,0x14
   1d974:	001006b7          	lui	a3,0x100
   1d978:	fff68793          	addi	a5,a3,-1 # fffff <__BSS_END__+0xdd1ff>
   1d97c:	7ff77713          	andi	a4,a4,2047
   1d980:	3fe00613          	li	a2,1022
   1d984:	00b7f7b3          	and	a5,a5,a1
   1d988:	01f5d593          	srli	a1,a1,0x1f
   1d98c:	04e65e63          	bge	a2,a4,1d9e8 <__fixdfsi+0x78>
   1d990:	41d00613          	li	a2,1053
   1d994:	00e65a63          	bge	a2,a4,1d9a8 <__fixdfsi+0x38>
   1d998:	80000537          	lui	a0,0x80000
   1d99c:	fff50513          	addi	a0,a0,-1 # 7fffffff <__BSS_END__+0x7ffdd1ff>
   1d9a0:	00a58533          	add	a0,a1,a0
   1d9a4:	00008067          	ret
   1d9a8:	00d7e7b3          	or	a5,a5,a3
   1d9ac:	43300693          	li	a3,1075
   1d9b0:	40e686b3          	sub	a3,a3,a4
   1d9b4:	01f00613          	li	a2,31
   1d9b8:	02d64063          	blt	a2,a3,1d9d8 <__fixdfsi+0x68>
   1d9bc:	bed70713          	addi	a4,a4,-1043
   1d9c0:	00e797b3          	sll	a5,a5,a4
   1d9c4:	00d55533          	srl	a0,a0,a3
   1d9c8:	00a7e533          	or	a0,a5,a0
   1d9cc:	02058063          	beqz	a1,1d9ec <__fixdfsi+0x7c>
   1d9d0:	40a00533          	neg	a0,a0
   1d9d4:	00008067          	ret
   1d9d8:	41300693          	li	a3,1043
   1d9dc:	40e68733          	sub	a4,a3,a4
   1d9e0:	00e7d533          	srl	a0,a5,a4
   1d9e4:	fe9ff06f          	j	1d9cc <__fixdfsi+0x5c>
   1d9e8:	00000513          	li	a0,0
   1d9ec:	00008067          	ret

0001d9f0 <__floatsidf>:
   1d9f0:	ff010113          	addi	sp,sp,-16
   1d9f4:	00112623          	sw	ra,12(sp)
   1d9f8:	00812423          	sw	s0,8(sp)
   1d9fc:	00912223          	sw	s1,4(sp)
   1da00:	08050663          	beqz	a0,1da8c <__floatsidf+0x9c>
   1da04:	41f55793          	srai	a5,a0,0x1f
   1da08:	00a7c433          	xor	s0,a5,a0
   1da0c:	40f40433          	sub	s0,s0,a5
   1da10:	01f55493          	srli	s1,a0,0x1f
   1da14:	00040513          	mv	a0,s0
   1da18:	525020ef          	jal	2073c <__clzsi2>
   1da1c:	41e00713          	li	a4,1054
   1da20:	00a00793          	li	a5,10
   1da24:	40a70733          	sub	a4,a4,a0
   1da28:	04a7c863          	blt	a5,a0,1da78 <__floatsidf+0x88>
   1da2c:	00b00793          	li	a5,11
   1da30:	40a787b3          	sub	a5,a5,a0
   1da34:	01550513          	addi	a0,a0,21
   1da38:	00f457b3          	srl	a5,s0,a5
   1da3c:	00a41433          	sll	s0,s0,a0
   1da40:	00048513          	mv	a0,s1
   1da44:	00c79793          	slli	a5,a5,0xc
   1da48:	00c7d793          	srli	a5,a5,0xc
   1da4c:	01471713          	slli	a4,a4,0x14
   1da50:	01f51513          	slli	a0,a0,0x1f
   1da54:	00f76733          	or	a4,a4,a5
   1da58:	00c12083          	lw	ra,12(sp)
   1da5c:	00a767b3          	or	a5,a4,a0
   1da60:	00040513          	mv	a0,s0
   1da64:	00812403          	lw	s0,8(sp)
   1da68:	00412483          	lw	s1,4(sp)
   1da6c:	00078593          	mv	a1,a5
   1da70:	01010113          	addi	sp,sp,16
   1da74:	00008067          	ret
   1da78:	ff550513          	addi	a0,a0,-11
   1da7c:	00a417b3          	sll	a5,s0,a0
   1da80:	00048513          	mv	a0,s1
   1da84:	00000413          	li	s0,0
   1da88:	fbdff06f          	j	1da44 <__floatsidf+0x54>
   1da8c:	00000713          	li	a4,0
   1da90:	00000793          	li	a5,0
   1da94:	ff1ff06f          	j	1da84 <__floatsidf+0x94>

0001da98 <__eqtf2>:
   1da98:	00c52783          	lw	a5,12(a0)
   1da9c:	0005af03          	lw	t5,0(a1)
   1daa0:	0045af83          	lw	t6,4(a1)
   1daa4:	0085a283          	lw	t0,8(a1)
   1daa8:	00c5a583          	lw	a1,12(a1)
   1daac:	00008737          	lui	a4,0x8
   1dab0:	0107d693          	srli	a3,a5,0x10
   1dab4:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1dab8:	01079813          	slli	a6,a5,0x10
   1dabc:	01059e93          	slli	t4,a1,0x10
   1dac0:	01f7d613          	srli	a2,a5,0x1f
   1dac4:	00e6f6b3          	and	a3,a3,a4
   1dac8:	0105d793          	srli	a5,a1,0x10
   1dacc:	00052883          	lw	a7,0(a0)
   1dad0:	00452303          	lw	t1,4(a0)
   1dad4:	00852e03          	lw	t3,8(a0)
   1dad8:	ff010113          	addi	sp,sp,-16
   1dadc:	01085813          	srli	a6,a6,0x10
   1dae0:	010ede93          	srli	t4,t4,0x10
   1dae4:	00e7f7b3          	and	a5,a5,a4
   1dae8:	01f5d593          	srli	a1,a1,0x1f
   1daec:	02e69063          	bne	a3,a4,1db0c <__eqtf2+0x74>
   1daf0:	0068e733          	or	a4,a7,t1
   1daf4:	01c76733          	or	a4,a4,t3
   1daf8:	01076733          	or	a4,a4,a6
   1dafc:	00100513          	li	a0,1
   1db00:	04071a63          	bnez	a4,1db54 <__eqtf2+0xbc>
   1db04:	04d79863          	bne	a5,a3,1db54 <__eqtf2+0xbc>
   1db08:	0080006f          	j	1db10 <__eqtf2+0x78>
   1db0c:	00e79c63          	bne	a5,a4,1db24 <__eqtf2+0x8c>
   1db10:	01ff6733          	or	a4,t5,t6
   1db14:	00576733          	or	a4,a4,t0
   1db18:	01d76733          	or	a4,a4,t4
   1db1c:	00100513          	li	a0,1
   1db20:	02071a63          	bnez	a4,1db54 <__eqtf2+0xbc>
   1db24:	00100513          	li	a0,1
   1db28:	02d79663          	bne	a5,a3,1db54 <__eqtf2+0xbc>
   1db2c:	03e89463          	bne	a7,t5,1db54 <__eqtf2+0xbc>
   1db30:	03f31263          	bne	t1,t6,1db54 <__eqtf2+0xbc>
   1db34:	025e1063          	bne	t3,t0,1db54 <__eqtf2+0xbc>
   1db38:	01d81e63          	bne	a6,t4,1db54 <__eqtf2+0xbc>
   1db3c:	02b60063          	beq	a2,a1,1db5c <__eqtf2+0xc4>
   1db40:	00079a63          	bnez	a5,1db54 <__eqtf2+0xbc>
   1db44:	0068e533          	or	a0,a7,t1
   1db48:	01c56533          	or	a0,a0,t3
   1db4c:	01056533          	or	a0,a0,a6
   1db50:	00a03533          	snez	a0,a0
   1db54:	01010113          	addi	sp,sp,16
   1db58:	00008067          	ret
   1db5c:	00000513          	li	a0,0
   1db60:	ff5ff06f          	j	1db54 <__eqtf2+0xbc>

0001db64 <__getf2>:
   1db64:	00c52783          	lw	a5,12(a0)
   1db68:	00c5a683          	lw	a3,12(a1)
   1db6c:	00008737          	lui	a4,0x8
   1db70:	0107d613          	srli	a2,a5,0x10
   1db74:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1db78:	01079813          	slli	a6,a5,0x10
   1db7c:	01069293          	slli	t0,a3,0x10
   1db80:	00052883          	lw	a7,0(a0)
   1db84:	00452303          	lw	t1,4(a0)
   1db88:	00852e03          	lw	t3,8(a0)
   1db8c:	00e67633          	and	a2,a2,a4
   1db90:	0106d513          	srli	a0,a3,0x10
   1db94:	0005ae83          	lw	t4,0(a1)
   1db98:	0045af03          	lw	t5,4(a1)
   1db9c:	0085af83          	lw	t6,8(a1)
   1dba0:	ff010113          	addi	sp,sp,-16
   1dba4:	01085813          	srli	a6,a6,0x10
   1dba8:	01f7d793          	srli	a5,a5,0x1f
   1dbac:	0102d293          	srli	t0,t0,0x10
   1dbb0:	00e57533          	and	a0,a0,a4
   1dbb4:	01f6d693          	srli	a3,a3,0x1f
   1dbb8:	00e61a63          	bne	a2,a4,1dbcc <__getf2+0x68>
   1dbbc:	01136733          	or	a4,t1,a7
   1dbc0:	01c76733          	or	a4,a4,t3
   1dbc4:	01076733          	or	a4,a4,a6
   1dbc8:	0a071463          	bnez	a4,1dc70 <__getf2+0x10c>
   1dbcc:	00008737          	lui	a4,0x8
   1dbd0:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1dbd4:	00e51a63          	bne	a0,a4,1dbe8 <__getf2+0x84>
   1dbd8:	01eee733          	or	a4,t4,t5
   1dbdc:	01f76733          	or	a4,a4,t6
   1dbe0:	00576733          	or	a4,a4,t0
   1dbe4:	08071663          	bnez	a4,1dc70 <__getf2+0x10c>
   1dbe8:	00000713          	li	a4,0
   1dbec:	00061a63          	bnez	a2,1dc00 <__getf2+0x9c>
   1dbf0:	01136733          	or	a4,t1,a7
   1dbf4:	01c76733          	or	a4,a4,t3
   1dbf8:	01076733          	or	a4,a4,a6
   1dbfc:	00173713          	seqz	a4,a4
   1dc00:	06051c63          	bnez	a0,1dc78 <__getf2+0x114>
   1dc04:	01eee5b3          	or	a1,t4,t5
   1dc08:	01f5e5b3          	or	a1,a1,t6
   1dc0c:	0055e5b3          	or	a1,a1,t0
   1dc10:	00070c63          	beqz	a4,1dc28 <__getf2+0xc4>
   1dc14:	02058063          	beqz	a1,1dc34 <__getf2+0xd0>
   1dc18:	00100513          	li	a0,1
   1dc1c:	00069c63          	bnez	a3,1dc34 <__getf2+0xd0>
   1dc20:	fff00513          	li	a0,-1
   1dc24:	0100006f          	j	1dc34 <__getf2+0xd0>
   1dc28:	04059a63          	bnez	a1,1dc7c <__getf2+0x118>
   1dc2c:	fff00513          	li	a0,-1
   1dc30:	04078e63          	beqz	a5,1dc8c <__getf2+0x128>
   1dc34:	01010113          	addi	sp,sp,16
   1dc38:	00008067          	ret
   1dc3c:	fca64ee3          	blt	a2,a0,1dc18 <__getf2+0xb4>
   1dc40:	ff02e6e3          	bltu	t0,a6,1dc2c <__getf2+0xc8>
   1dc44:	02581063          	bne	a6,t0,1dc64 <__getf2+0x100>
   1dc48:	ffcfe2e3          	bltu	t6,t3,1dc2c <__getf2+0xc8>
   1dc4c:	01fe1c63          	bne	t3,t6,1dc64 <__getf2+0x100>
   1dc50:	fc6f6ee3          	bltu	t5,t1,1dc2c <__getf2+0xc8>
   1dc54:	01e31863          	bne	t1,t5,1dc64 <__getf2+0x100>
   1dc58:	fd1eeae3          	bltu	t4,a7,1dc2c <__getf2+0xc8>
   1dc5c:	00000513          	li	a0,0
   1dc60:	fdd8fae3          	bgeu	a7,t4,1dc34 <__getf2+0xd0>
   1dc64:	00100513          	li	a0,1
   1dc68:	fc0796e3          	bnez	a5,1dc34 <__getf2+0xd0>
   1dc6c:	fb5ff06f          	j	1dc20 <__getf2+0xbc>
   1dc70:	ffe00513          	li	a0,-2
   1dc74:	fc1ff06f          	j	1dc34 <__getf2+0xd0>
   1dc78:	fa0710e3          	bnez	a4,1dc18 <__getf2+0xb4>
   1dc7c:	faf698e3          	bne	a3,a5,1dc2c <__getf2+0xc8>
   1dc80:	fac55ee3          	bge	a0,a2,1dc3c <__getf2+0xd8>
   1dc84:	fff00513          	li	a0,-1
   1dc88:	fa0696e3          	bnez	a3,1dc34 <__getf2+0xd0>
   1dc8c:	00100513          	li	a0,1
   1dc90:	fa5ff06f          	j	1dc34 <__getf2+0xd0>

0001dc94 <__letf2>:
   1dc94:	00c52783          	lw	a5,12(a0)
   1dc98:	00c5a683          	lw	a3,12(a1)
   1dc9c:	00008737          	lui	a4,0x8
   1dca0:	0107d613          	srli	a2,a5,0x10
   1dca4:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1dca8:	01079813          	slli	a6,a5,0x10
   1dcac:	01069293          	slli	t0,a3,0x10
   1dcb0:	00052883          	lw	a7,0(a0)
   1dcb4:	00452303          	lw	t1,4(a0)
   1dcb8:	00852e03          	lw	t3,8(a0)
   1dcbc:	00e67633          	and	a2,a2,a4
   1dcc0:	0106d513          	srli	a0,a3,0x10
   1dcc4:	0005ae83          	lw	t4,0(a1)
   1dcc8:	0045af03          	lw	t5,4(a1)
   1dccc:	0085af83          	lw	t6,8(a1)
   1dcd0:	ff010113          	addi	sp,sp,-16
   1dcd4:	01085813          	srli	a6,a6,0x10
   1dcd8:	01f7d793          	srli	a5,a5,0x1f
   1dcdc:	0102d293          	srli	t0,t0,0x10
   1dce0:	00e57533          	and	a0,a0,a4
   1dce4:	01f6d693          	srli	a3,a3,0x1f
   1dce8:	00e61a63          	bne	a2,a4,1dcfc <__letf2+0x68>
   1dcec:	01136733          	or	a4,t1,a7
   1dcf0:	01c76733          	or	a4,a4,t3
   1dcf4:	01076733          	or	a4,a4,a6
   1dcf8:	0a071463          	bnez	a4,1dda0 <__letf2+0x10c>
   1dcfc:	00008737          	lui	a4,0x8
   1dd00:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1dd04:	00e51a63          	bne	a0,a4,1dd18 <__letf2+0x84>
   1dd08:	01eee733          	or	a4,t4,t5
   1dd0c:	01f76733          	or	a4,a4,t6
   1dd10:	00576733          	or	a4,a4,t0
   1dd14:	08071663          	bnez	a4,1dda0 <__letf2+0x10c>
   1dd18:	00000713          	li	a4,0
   1dd1c:	00061a63          	bnez	a2,1dd30 <__letf2+0x9c>
   1dd20:	01136733          	or	a4,t1,a7
   1dd24:	01c76733          	or	a4,a4,t3
   1dd28:	01076733          	or	a4,a4,a6
   1dd2c:	00173713          	seqz	a4,a4
   1dd30:	06051c63          	bnez	a0,1dda8 <__letf2+0x114>
   1dd34:	01eee5b3          	or	a1,t4,t5
   1dd38:	01f5e5b3          	or	a1,a1,t6
   1dd3c:	0055e5b3          	or	a1,a1,t0
   1dd40:	00070c63          	beqz	a4,1dd58 <__letf2+0xc4>
   1dd44:	02058063          	beqz	a1,1dd64 <__letf2+0xd0>
   1dd48:	00100513          	li	a0,1
   1dd4c:	00069c63          	bnez	a3,1dd64 <__letf2+0xd0>
   1dd50:	fff00513          	li	a0,-1
   1dd54:	0100006f          	j	1dd64 <__letf2+0xd0>
   1dd58:	04059a63          	bnez	a1,1ddac <__letf2+0x118>
   1dd5c:	fff00513          	li	a0,-1
   1dd60:	04078e63          	beqz	a5,1ddbc <__letf2+0x128>
   1dd64:	01010113          	addi	sp,sp,16
   1dd68:	00008067          	ret
   1dd6c:	fca64ee3          	blt	a2,a0,1dd48 <__letf2+0xb4>
   1dd70:	ff02e6e3          	bltu	t0,a6,1dd5c <__letf2+0xc8>
   1dd74:	02581063          	bne	a6,t0,1dd94 <__letf2+0x100>
   1dd78:	ffcfe2e3          	bltu	t6,t3,1dd5c <__letf2+0xc8>
   1dd7c:	01fe1c63          	bne	t3,t6,1dd94 <__letf2+0x100>
   1dd80:	fc6f6ee3          	bltu	t5,t1,1dd5c <__letf2+0xc8>
   1dd84:	01e31863          	bne	t1,t5,1dd94 <__letf2+0x100>
   1dd88:	fd1eeae3          	bltu	t4,a7,1dd5c <__letf2+0xc8>
   1dd8c:	00000513          	li	a0,0
   1dd90:	fdd8fae3          	bgeu	a7,t4,1dd64 <__letf2+0xd0>
   1dd94:	00100513          	li	a0,1
   1dd98:	fc0796e3          	bnez	a5,1dd64 <__letf2+0xd0>
   1dd9c:	fb5ff06f          	j	1dd50 <__letf2+0xbc>
   1dda0:	00200513          	li	a0,2
   1dda4:	fc1ff06f          	j	1dd64 <__letf2+0xd0>
   1dda8:	fa0710e3          	bnez	a4,1dd48 <__letf2+0xb4>
   1ddac:	faf698e3          	bne	a3,a5,1dd5c <__letf2+0xc8>
   1ddb0:	fac55ee3          	bge	a0,a2,1dd6c <__letf2+0xd8>
   1ddb4:	fff00513          	li	a0,-1
   1ddb8:	fa0696e3          	bnez	a3,1dd64 <__letf2+0xd0>
   1ddbc:	00100513          	li	a0,1
   1ddc0:	fa5ff06f          	j	1dd64 <__letf2+0xd0>

0001ddc4 <__multf3>:
   1ddc4:	f5010113          	addi	sp,sp,-176
   1ddc8:	09412c23          	sw	s4,152(sp)
   1ddcc:	00c5aa03          	lw	s4,12(a1)
   1ddd0:	0005a783          	lw	a5,0(a1)
   1ddd4:	0085a683          	lw	a3,8(a1)
   1ddd8:	0a812423          	sw	s0,168(sp)
   1dddc:	00050413          	mv	s0,a0
   1dde0:	0045a503          	lw	a0,4(a1)
   1dde4:	010a1713          	slli	a4,s4,0x10
   1dde8:	0b212023          	sw	s2,160(sp)
   1ddec:	09312e23          	sw	s3,156(sp)
   1ddf0:	00062903          	lw	s2,0(a2) # 7ff00000 <__BSS_END__+0x7fedd200>
   1ddf4:	00c62983          	lw	s3,12(a2)
   1ddf8:	09512a23          	sw	s5,148(sp)
   1ddfc:	09612823          	sw	s6,144(sp)
   1de00:	00862a83          	lw	s5,8(a2)
   1de04:	00462b03          	lw	s6,4(a2)
   1de08:	00008637          	lui	a2,0x8
   1de0c:	0a912223          	sw	s1,164(sp)
   1de10:	01075713          	srli	a4,a4,0x10
   1de14:	010a5493          	srli	s1,s4,0x10
   1de18:	fff60613          	addi	a2,a2,-1 # 7fff <exit-0x80b5>
   1de1c:	05412e23          	sw	s4,92(sp)
   1de20:	0a112623          	sw	ra,172(sp)
   1de24:	09712623          	sw	s7,140(sp)
   1de28:	09812423          	sw	s8,136(sp)
   1de2c:	09912223          	sw	s9,132(sp)
   1de30:	09a12023          	sw	s10,128(sp)
   1de34:	07b12e23          	sw	s11,124(sp)
   1de38:	04f12823          	sw	a5,80(sp)
   1de3c:	04a12a23          	sw	a0,84(sp)
   1de40:	04d12c23          	sw	a3,88(sp)
   1de44:	02f12023          	sw	a5,32(sp)
   1de48:	02a12223          	sw	a0,36(sp)
   1de4c:	02d12423          	sw	a3,40(sp)
   1de50:	02e12623          	sw	a4,44(sp)
   1de54:	00c4f4b3          	and	s1,s1,a2
   1de58:	01fa5a13          	srli	s4,s4,0x1f
   1de5c:	080482e3          	beqz	s1,1e6e0 <__multf3+0x91c>
   1de60:	1ac48ce3          	beq	s1,a2,1e818 <__multf3+0xa54>
   1de64:	000106b7          	lui	a3,0x10
   1de68:	00d76733          	or	a4,a4,a3
   1de6c:	02e12623          	sw	a4,44(sp)
   1de70:	02010593          	addi	a1,sp,32
   1de74:	02c10713          	addi	a4,sp,44
   1de78:	00072683          	lw	a3,0(a4)
   1de7c:	ffc72603          	lw	a2,-4(a4)
   1de80:	ffc70713          	addi	a4,a4,-4
   1de84:	00369693          	slli	a3,a3,0x3
   1de88:	01d65613          	srli	a2,a2,0x1d
   1de8c:	00c6e6b3          	or	a3,a3,a2
   1de90:	00d72223          	sw	a3,4(a4)
   1de94:	fee592e3          	bne	a1,a4,1de78 <__multf3+0xb4>
   1de98:	00379793          	slli	a5,a5,0x3
   1de9c:	02f12023          	sw	a5,32(sp)
   1dea0:	ffffc7b7          	lui	a5,0xffffc
   1dea4:	00178793          	addi	a5,a5,1 # ffffc001 <__BSS_END__+0xfffd9201>
   1dea8:	00f484b3          	add	s1,s1,a5
   1deac:	00000b93          	li	s7,0
   1deb0:	01099513          	slli	a0,s3,0x10
   1deb4:	00008737          	lui	a4,0x8
   1deb8:	0109d793          	srli	a5,s3,0x10
   1debc:	01055513          	srli	a0,a0,0x10
   1dec0:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1dec4:	05312e23          	sw	s3,92(sp)
   1dec8:	05212823          	sw	s2,80(sp)
   1decc:	05612a23          	sw	s6,84(sp)
   1ded0:	05512c23          	sw	s5,88(sp)
   1ded4:	03212823          	sw	s2,48(sp)
   1ded8:	03612a23          	sw	s6,52(sp)
   1dedc:	03512c23          	sw	s5,56(sp)
   1dee0:	02a12e23          	sw	a0,60(sp)
   1dee4:	00e7f7b3          	and	a5,a5,a4
   1dee8:	01f9d993          	srli	s3,s3,0x1f
   1deec:	14078ae3          	beqz	a5,1e840 <__multf3+0xa7c>
   1def0:	28e784e3          	beq	a5,a4,1e978 <__multf3+0xbb4>
   1def4:	00010737          	lui	a4,0x10
   1def8:	00e56533          	or	a0,a0,a4
   1defc:	02a12e23          	sw	a0,60(sp)
   1df00:	03010593          	addi	a1,sp,48
   1df04:	03c10713          	addi	a4,sp,60
   1df08:	00072683          	lw	a3,0(a4) # 10000 <exit-0xb4>
   1df0c:	ffc72603          	lw	a2,-4(a4)
   1df10:	ffc70713          	addi	a4,a4,-4
   1df14:	00369693          	slli	a3,a3,0x3
   1df18:	01d65613          	srli	a2,a2,0x1d
   1df1c:	00c6e6b3          	or	a3,a3,a2
   1df20:	00d72223          	sw	a3,4(a4)
   1df24:	fee592e3          	bne	a1,a4,1df08 <__multf3+0x144>
   1df28:	ffffc737          	lui	a4,0xffffc
   1df2c:	00391913          	slli	s2,s2,0x3
   1df30:	00170713          	addi	a4,a4,1 # ffffc001 <__BSS_END__+0xfffd9201>
   1df34:	03212823          	sw	s2,48(sp)
   1df38:	00e787b3          	add	a5,a5,a4
   1df3c:	00000693          	li	a3,0
   1df40:	009787b3          	add	a5,a5,s1
   1df44:	00f12623          	sw	a5,12(sp)
   1df48:	00178793          	addi	a5,a5,1
   1df4c:	00f12423          	sw	a5,8(sp)
   1df50:	002b9793          	slli	a5,s7,0x2
   1df54:	00d7e7b3          	or	a5,a5,a3
   1df58:	00a00713          	li	a4,10
   1df5c:	28f742e3          	blt	a4,a5,1e9e0 <__multf3+0xc1c>
   1df60:	013a4733          	xor	a4,s4,s3
   1df64:	00e12223          	sw	a4,4(sp)
   1df68:	00200713          	li	a4,2
   1df6c:	22f74ae3          	blt	a4,a5,1e9a0 <__multf3+0xbdc>
   1df70:	fff78793          	addi	a5,a5,-1
   1df74:	00100713          	li	a4,1
   1df78:	28f770e3          	bgeu	a4,a5,1e9f8 <__multf3+0xc34>
   1df7c:	02012883          	lw	a7,32(sp)
   1df80:	03012f03          	lw	t5,48(sp)
   1df84:	000105b7          	lui	a1,0x10
   1df88:	fff58713          	addi	a4,a1,-1 # ffff <exit-0xb5>
   1df8c:	0108d913          	srli	s2,a7,0x10
   1df90:	010f5493          	srli	s1,t5,0x10
   1df94:	00e8f8b3          	and	a7,a7,a4
   1df98:	00ef7f33          	and	t5,t5,a4
   1df9c:	031f0633          	mul	a2,t5,a7
   1dfa0:	03e90533          	mul	a0,s2,t5
   1dfa4:	01065793          	srli	a5,a2,0x10
   1dfa8:	031486b3          	mul	a3,s1,a7
   1dfac:	00a686b3          	add	a3,a3,a0
   1dfb0:	00d787b3          	add	a5,a5,a3
   1dfb4:	02990bb3          	mul	s7,s2,s1
   1dfb8:	00a7f463          	bgeu	a5,a0,1dfc0 <__multf3+0x1fc>
   1dfbc:	00bb8bb3          	add	s7,s7,a1
   1dfc0:	03412e83          	lw	t4,52(sp)
   1dfc4:	0107d693          	srli	a3,a5,0x10
   1dfc8:	00e7f7b3          	and	a5,a5,a4
   1dfcc:	00e67633          	and	a2,a2,a4
   1dfd0:	01079793          	slli	a5,a5,0x10
   1dfd4:	00c787b3          	add	a5,a5,a2
   1dfd8:	010ed293          	srli	t0,t4,0x10
   1dfdc:	00eefeb3          	and	t4,t4,a4
   1dfe0:	03d90633          	mul	a2,s2,t4
   1dfe4:	00f12823          	sw	a5,16(sp)
   1dfe8:	04f12823          	sw	a5,80(sp)
   1dfec:	03128733          	mul	a4,t0,a7
   1dff0:	031e87b3          	mul	a5,t4,a7
   1dff4:	00c70733          	add	a4,a4,a2
   1dff8:	0107d313          	srli	t1,a5,0x10
   1dffc:	00e30333          	add	t1,t1,a4
   1e000:	02590b33          	mul	s6,s2,t0
   1e004:	00c37663          	bgeu	t1,a2,1e010 <__multf3+0x24c>
   1e008:	00010737          	lui	a4,0x10
   1e00c:	00eb0b33          	add	s6,s6,a4
   1e010:	02412803          	lw	a6,36(sp)
   1e014:	00010737          	lui	a4,0x10
   1e018:	fff70613          	addi	a2,a4,-1 # ffff <exit-0xb5>
   1e01c:	01035513          	srli	a0,t1,0x10
   1e020:	00c37333          	and	t1,t1,a2
   1e024:	00c7f7b3          	and	a5,a5,a2
   1e028:	01085393          	srli	t2,a6,0x10
   1e02c:	01031313          	slli	t1,t1,0x10
   1e030:	00c87833          	and	a6,a6,a2
   1e034:	00f30333          	add	t1,t1,a5
   1e038:	03e38fb3          	mul	t6,t2,t5
   1e03c:	006686b3          	add	a3,a3,t1
   1e040:	03e807b3          	mul	a5,a6,t5
   1e044:	03048e33          	mul	t3,s1,a6
   1e048:	0107d613          	srli	a2,a5,0x10
   1e04c:	01fe0e33          	add	t3,t3,t6
   1e050:	01c60633          	add	a2,a2,t3
   1e054:	027485b3          	mul	a1,s1,t2
   1e058:	01f67463          	bgeu	a2,t6,1e060 <__multf3+0x29c>
   1e05c:	00e585b3          	add	a1,a1,a4
   1e060:	01065a93          	srli	s5,a2,0x10
   1e064:	00010737          	lui	a4,0x10
   1e068:	00ba8ab3          	add	s5,s5,a1
   1e06c:	fff70593          	addi	a1,a4,-1 # ffff <exit-0xb5>
   1e070:	00b7f7b3          	and	a5,a5,a1
   1e074:	00b67633          	and	a2,a2,a1
   1e078:	01061613          	slli	a2,a2,0x10
   1e07c:	030e85b3          	mul	a1,t4,a6
   1e080:	00f60633          	add	a2,a2,a5
   1e084:	03d38fb3          	mul	t6,t2,t4
   1e088:	0105d793          	srli	a5,a1,0x10
   1e08c:	03028e33          	mul	t3,t0,a6
   1e090:	01fe0e33          	add	t3,t3,t6
   1e094:	01c787b3          	add	a5,a5,t3
   1e098:	027289b3          	mul	s3,t0,t2
   1e09c:	01f7f463          	bgeu	a5,t6,1e0a4 <__multf3+0x2e0>
   1e0a0:	00e989b3          	add	s3,s3,a4
   1e0a4:	0107d713          	srli	a4,a5,0x10
   1e0a8:	01370733          	add	a4,a4,s3
   1e0ac:	00010a37          	lui	s4,0x10
   1e0b0:	00e12a23          	sw	a4,20(sp)
   1e0b4:	fffa0713          	addi	a4,s4,-1 # ffff <exit-0xb5>
   1e0b8:	03812e03          	lw	t3,56(sp)
   1e0bc:	00e7f7b3          	and	a5,a5,a4
   1e0c0:	00e5f5b3          	and	a1,a1,a4
   1e0c4:	01079793          	slli	a5,a5,0x10
   1e0c8:	00b787b3          	add	a5,a5,a1
   1e0cc:	00f12c23          	sw	a5,24(sp)
   1e0d0:	010e5793          	srli	a5,t3,0x10
   1e0d4:	00ee7e33          	and	t3,t3,a4
   1e0d8:	031e05b3          	mul	a1,t3,a7
   1e0dc:	03c90c33          	mul	s8,s2,t3
   1e0e0:	0105d713          	srli	a4,a1,0x10
   1e0e4:	031789b3          	mul	s3,a5,a7
   1e0e8:	018989b3          	add	s3,s3,s8
   1e0ec:	01370733          	add	a4,a4,s3
   1e0f0:	02f90fb3          	mul	t6,s2,a5
   1e0f4:	01877463          	bgeu	a4,s8,1e0fc <__multf3+0x338>
   1e0f8:	014f8fb3          	add	t6,t6,s4
   1e0fc:	01075993          	srli	s3,a4,0x10
   1e100:	00010cb7          	lui	s9,0x10
   1e104:	01f98fb3          	add	t6,s3,t6
   1e108:	fffc8993          	addi	s3,s9,-1 # ffff <exit-0xb5>
   1e10c:	01377733          	and	a4,a4,s3
   1e110:	0135f5b3          	and	a1,a1,s3
   1e114:	01071713          	slli	a4,a4,0x10
   1e118:	00b70733          	add	a4,a4,a1
   1e11c:	02812583          	lw	a1,40(sp)
   1e120:	01f12e23          	sw	t6,28(sp)
   1e124:	0105df93          	srli	t6,a1,0x10
   1e128:	0135f5b3          	and	a1,a1,s3
   1e12c:	03e58a33          	mul	s4,a1,t5
   1e130:	03ef8d33          	mul	s10,t6,t5
   1e134:	010a5d93          	srli	s11,s4,0x10
   1e138:	02b489b3          	mul	s3,s1,a1
   1e13c:	01a989b3          	add	s3,s3,s10
   1e140:	013d89b3          	add	s3,s11,s3
   1e144:	03f48c33          	mul	s8,s1,t6
   1e148:	01a9f463          	bgeu	s3,s10,1e150 <__multf3+0x38c>
   1e14c:	019c0c33          	add	s8,s8,s9
   1e150:	00db86b3          	add	a3,s7,a3
   1e154:	0066b333          	sltu	t1,a3,t1
   1e158:	0109dd13          	srli	s10,s3,0x10
   1e15c:	00650533          	add	a0,a0,t1
   1e160:	00010cb7          	lui	s9,0x10
   1e164:	01650533          	add	a0,a0,s6
   1e168:	018d0d33          	add	s10,s10,s8
   1e16c:	00c68633          	add	a2,a3,a2
   1e170:	fffc8c13          	addi	s8,s9,-1 # ffff <exit-0xb5>
   1e174:	01550ab3          	add	s5,a0,s5
   1e178:	0189f9b3          	and	s3,s3,s8
   1e17c:	00d636b3          	sltu	a3,a2,a3
   1e180:	00da86b3          	add	a3,s5,a3
   1e184:	01099993          	slli	s3,s3,0x10
   1e188:	018a7a33          	and	s4,s4,s8
   1e18c:	01498a33          	add	s4,s3,s4
   1e190:	00aab9b3          	sltu	s3,s5,a0
   1e194:	0156bab3          	sltu	s5,a3,s5
   1e198:	0159e9b3          	or	s3,s3,s5
   1e19c:	00653533          	sltu	a0,a0,t1
   1e1a0:	00a98533          	add	a0,s3,a0
   1e1a4:	01812303          	lw	t1,24(sp)
   1e1a8:	01412983          	lw	s3,20(sp)
   1e1ac:	04c12a23          	sw	a2,84(sp)
   1e1b0:	00668333          	add	t1,a3,t1
   1e1b4:	01350ab3          	add	s5,a0,s3
   1e1b8:	01c12983          	lw	s3,28(sp)
   1e1bc:	00d336b3          	sltu	a3,t1,a3
   1e1c0:	00da86b3          	add	a3,s5,a3
   1e1c4:	00e30733          	add	a4,t1,a4
   1e1c8:	01368b33          	add	s6,a3,s3
   1e1cc:	00673333          	sltu	t1,a4,t1
   1e1d0:	006b0333          	add	t1,s6,t1
   1e1d4:	01470a33          	add	s4,a4,s4
   1e1d8:	01a30d33          	add	s10,t1,s10
   1e1dc:	00ea3733          	sltu	a4,s4,a4
   1e1e0:	00aab533          	sltu	a0,s5,a0
   1e1e4:	00ed0733          	add	a4,s10,a4
   1e1e8:	0156bab3          	sltu	s5,a3,s5
   1e1ec:	00db36b3          	sltu	a3,s6,a3
   1e1f0:	01633b33          	sltu	s6,t1,s6
   1e1f4:	0166e6b3          	or	a3,a3,s6
   1e1f8:	006d39b3          	sltu	s3,s10,t1
   1e1fc:	01556ab3          	or	s5,a0,s5
   1e200:	01a73d33          	sltu	s10,a4,s10
   1e204:	00da8ab3          	add	s5,s5,a3
   1e208:	01a9e9b3          	or	s3,s3,s10
   1e20c:	015989b3          	add	s3,s3,s5
   1e210:	03c12a83          	lw	s5,60(sp)
   1e214:	05412c23          	sw	s4,88(sp)
   1e218:	010adb13          	srli	s6,s5,0x10
   1e21c:	018afab3          	and	s5,s5,s8
   1e220:	031a86b3          	mul	a3,s5,a7
   1e224:	03590533          	mul	a0,s2,s5
   1e228:	031b08b3          	mul	a7,s6,a7
   1e22c:	00a88333          	add	t1,a7,a0
   1e230:	0106d893          	srli	a7,a3,0x10
   1e234:	006888b3          	add	a7,a7,t1
   1e238:	03690933          	mul	s2,s2,s6
   1e23c:	00a8f463          	bgeu	a7,a0,1e244 <__multf3+0x480>
   1e240:	01990933          	add	s2,s2,s9
   1e244:	02c12b83          	lw	s7,44(sp)
   1e248:	0108d513          	srli	a0,a7,0x10
   1e24c:	01250533          	add	a0,a0,s2
   1e250:	00010c37          	lui	s8,0x10
   1e254:	00a12a23          	sw	a0,20(sp)
   1e258:	fffc0513          	addi	a0,s8,-1 # ffff <exit-0xb5>
   1e25c:	010bd913          	srli	s2,s7,0x10
   1e260:	00abfbb3          	and	s7,s7,a0
   1e264:	00a6f6b3          	and	a3,a3,a0
   1e268:	00a8f8b3          	and	a7,a7,a0
   1e26c:	03248333          	mul	t1,s1,s2
   1e270:	01089893          	slli	a7,a7,0x10
   1e274:	00d888b3          	add	a7,a7,a3
   1e278:	03eb8533          	mul	a0,s7,t5
   1e27c:	037484b3          	mul	s1,s1,s7
   1e280:	01055693          	srli	a3,a0,0x10
   1e284:	03e90f33          	mul	t5,s2,t5
   1e288:	01e484b3          	add	s1,s1,t5
   1e28c:	009686b3          	add	a3,a3,s1
   1e290:	01e6f463          	bgeu	a3,t5,1e298 <__multf3+0x4d4>
   1e294:	01830333          	add	t1,t1,s8
   1e298:	0106df13          	srli	t5,a3,0x10
   1e29c:	006f0333          	add	t1,t5,t1
   1e2a0:	00010cb7          	lui	s9,0x10
   1e2a4:	00612c23          	sw	t1,24(sp)
   1e2a8:	fffc8313          	addi	t1,s9,-1 # ffff <exit-0xb5>
   1e2ac:	00657533          	and	a0,a0,t1
   1e2b0:	0066f6b3          	and	a3,a3,t1
   1e2b4:	03c38f33          	mul	t5,t2,t3
   1e2b8:	01069693          	slli	a3,a3,0x10
   1e2bc:	00a686b3          	add	a3,a3,a0
   1e2c0:	03c80333          	mul	t1,a6,t3
   1e2c4:	030784b3          	mul	s1,a5,a6
   1e2c8:	01035513          	srli	a0,t1,0x10
   1e2cc:	01e484b3          	add	s1,s1,t5
   1e2d0:	00950533          	add	a0,a0,s1
   1e2d4:	02f38c33          	mul	s8,t2,a5
   1e2d8:	01e57463          	bgeu	a0,t5,1e2e0 <__multf3+0x51c>
   1e2dc:	019c0c33          	add	s8,s8,s9
   1e2e0:	00010d37          	lui	s10,0x10
   1e2e4:	fffd0f13          	addi	t5,s10,-1 # ffff <exit-0xb5>
   1e2e8:	01055493          	srli	s1,a0,0x10
   1e2ec:	01e57533          	and	a0,a0,t5
   1e2f0:	01e37333          	and	t1,t1,t5
   1e2f4:	01051513          	slli	a0,a0,0x10
   1e2f8:	018484b3          	add	s1,s1,s8
   1e2fc:	02b28f33          	mul	t5,t0,a1
   1e300:	00650533          	add	a0,a0,t1
   1e304:	03df8c33          	mul	s8,t6,t4
   1e308:	02be8333          	mul	t1,t4,a1
   1e30c:	018f0f33          	add	t5,t5,s8
   1e310:	01035d93          	srli	s11,t1,0x10
   1e314:	01ed8f33          	add	t5,s11,t5
   1e318:	03f28cb3          	mul	s9,t0,t6
   1e31c:	018f7463          	bgeu	t5,s8,1e324 <__multf3+0x560>
   1e320:	01ac8cb3          	add	s9,s9,s10
   1e324:	010f5c13          	srli	s8,t5,0x10
   1e328:	019c0c33          	add	s8,s8,s9
   1e32c:	00010cb7          	lui	s9,0x10
   1e330:	fffc8d13          	addi	s10,s9,-1 # ffff <exit-0xb5>
   1e334:	01af7f33          	and	t5,t5,s10
   1e338:	010f1f13          	slli	t5,t5,0x10
   1e33c:	01a37333          	and	t1,t1,s10
   1e340:	006f0333          	add	t1,t5,t1
   1e344:	01412f03          	lw	t5,20(sp)
   1e348:	011708b3          	add	a7,a4,a7
   1e34c:	01812d03          	lw	s10,24(sp)
   1e350:	01e98f33          	add	t5,s3,t5
   1e354:	00e8b733          	sltu	a4,a7,a4
   1e358:	00ef0733          	add	a4,t5,a4
   1e35c:	00d886b3          	add	a3,a7,a3
   1e360:	01a70d33          	add	s10,a4,s10
   1e364:	0116b8b3          	sltu	a7,a3,a7
   1e368:	011d08b3          	add	a7,s10,a7
   1e36c:	00a68533          	add	a0,a3,a0
   1e370:	009884b3          	add	s1,a7,s1
   1e374:	00d536b3          	sltu	a3,a0,a3
   1e378:	00d486b3          	add	a3,s1,a3
   1e37c:	013f39b3          	sltu	s3,t5,s3
   1e380:	01e73f33          	sltu	t5,a4,t5
   1e384:	00ed3733          	sltu	a4,s10,a4
   1e388:	01a8bd33          	sltu	s10,a7,s10
   1e38c:	01e9ef33          	or	t5,s3,t5
   1e390:	0114b8b3          	sltu	a7,s1,a7
   1e394:	01a76733          	or	a4,a4,s10
   1e398:	0096b4b3          	sltu	s1,a3,s1
   1e39c:	01868c33          	add	s8,a3,s8
   1e3a0:	00ef0733          	add	a4,t5,a4
   1e3a4:	0098e8b3          	or	a7,a7,s1
   1e3a8:	00e884b3          	add	s1,a7,a4
   1e3ac:	03cf8f33          	mul	t5,t6,t3
   1e3b0:	00dc38b3          	sltu	a7,s8,a3
   1e3b4:	00650333          	add	t1,a0,t1
   1e3b8:	00a33533          	sltu	a0,t1,a0
   1e3bc:	00ac0533          	add	a0,s8,a0
   1e3c0:	01853c33          	sltu	s8,a0,s8
   1e3c4:	0188e8b3          	or	a7,a7,s8
   1e3c8:	04612e23          	sw	t1,92(sp)
   1e3cc:	009888b3          	add	a7,a7,s1
   1e3d0:	02be06b3          	mul	a3,t3,a1
   1e3d4:	02b789b3          	mul	s3,a5,a1
   1e3d8:	0106d713          	srli	a4,a3,0x10
   1e3dc:	01e989b3          	add	s3,s3,t5
   1e3e0:	01370733          	add	a4,a4,s3
   1e3e4:	03f784b3          	mul	s1,a5,t6
   1e3e8:	01e77463          	bgeu	a4,t5,1e3f0 <__multf3+0x62c>
   1e3ec:	019484b3          	add	s1,s1,s9
   1e3f0:	01075f13          	srli	t5,a4,0x10
   1e3f4:	009f0f33          	add	t5,t5,s1
   1e3f8:	000104b7          	lui	s1,0x10
   1e3fc:	fff48993          	addi	s3,s1,-1 # ffff <exit-0xb5>
   1e400:	01377733          	and	a4,a4,s3
   1e404:	0136f6b3          	and	a3,a3,s3
   1e408:	01071713          	slli	a4,a4,0x10
   1e40c:	035389b3          	mul	s3,t2,s5
   1e410:	00d70733          	add	a4,a4,a3
   1e414:	030a86b3          	mul	a3,s5,a6
   1e418:	030b0833          	mul	a6,s6,a6
   1e41c:	01380c33          	add	s8,a6,s3
   1e420:	0106d813          	srli	a6,a3,0x10
   1e424:	01880833          	add	a6,a6,s8
   1e428:	036383b3          	mul	t2,t2,s6
   1e42c:	01387463          	bgeu	a6,s3,1e434 <__multf3+0x670>
   1e430:	009383b3          	add	t2,t2,s1
   1e434:	01085493          	srli	s1,a6,0x10
   1e438:	00010c37          	lui	s8,0x10
   1e43c:	007483b3          	add	t2,s1,t2
   1e440:	fffc0493          	addi	s1,s8,-1 # ffff <exit-0xb5>
   1e444:	0096f6b3          	and	a3,a3,s1
   1e448:	00987833          	and	a6,a6,s1
   1e44c:	01081813          	slli	a6,a6,0x10
   1e450:	03db89b3          	mul	s3,s7,t4
   1e454:	00d80833          	add	a6,a6,a3
   1e458:	03d90eb3          	mul	t4,s2,t4
   1e45c:	0109d693          	srli	a3,s3,0x10
   1e460:	032284b3          	mul	s1,t0,s2
   1e464:	037282b3          	mul	t0,t0,s7
   1e468:	01d282b3          	add	t0,t0,t4
   1e46c:	005686b3          	add	a3,a3,t0
   1e470:	01d6f463          	bgeu	a3,t4,1e478 <__multf3+0x6b4>
   1e474:	018484b3          	add	s1,s1,s8
   1e478:	0106de93          	srli	t4,a3,0x10
   1e47c:	009e8eb3          	add	t4,t4,s1
   1e480:	000104b7          	lui	s1,0x10
   1e484:	fff48293          	addi	t0,s1,-1 # ffff <exit-0xb5>
   1e488:	0056f6b3          	and	a3,a3,t0
   1e48c:	0059f9b3          	and	s3,s3,t0
   1e490:	01069693          	slli	a3,a3,0x10
   1e494:	02ba82b3          	mul	t0,s5,a1
   1e498:	013686b3          	add	a3,a3,s3
   1e49c:	02bb05b3          	mul	a1,s6,a1
   1e4a0:	035f89b3          	mul	s3,t6,s5
   1e4a4:	01358c33          	add	s8,a1,s3
   1e4a8:	0102d593          	srli	a1,t0,0x10
   1e4ac:	018585b3          	add	a1,a1,s8
   1e4b0:	036f8fb3          	mul	t6,t6,s6
   1e4b4:	0135f463          	bgeu	a1,s3,1e4bc <__multf3+0x6f8>
   1e4b8:	009f8fb3          	add	t6,t6,s1
   1e4bc:	0105d493          	srli	s1,a1,0x10
   1e4c0:	01f48fb3          	add	t6,s1,t6
   1e4c4:	000104b7          	lui	s1,0x10
   1e4c8:	fff48993          	addi	s3,s1,-1 # ffff <exit-0xb5>
   1e4cc:	0135f5b3          	and	a1,a1,s3
   1e4d0:	0132f2b3          	and	t0,t0,s3
   1e4d4:	01059593          	slli	a1,a1,0x10
   1e4d8:	032789b3          	mul	s3,a5,s2
   1e4dc:	005585b3          	add	a1,a1,t0
   1e4e0:	037787b3          	mul	a5,a5,s7
   1e4e4:	03cb82b3          	mul	t0,s7,t3
   1e4e8:	03c90e33          	mul	t3,s2,t3
   1e4ec:	0102dc13          	srli	s8,t0,0x10
   1e4f0:	01c787b3          	add	a5,a5,t3
   1e4f4:	00fc07b3          	add	a5,s8,a5
   1e4f8:	01c7f463          	bgeu	a5,t3,1e500 <__multf3+0x73c>
   1e4fc:	009989b3          	add	s3,s3,s1
   1e500:	00e50733          	add	a4,a0,a4
   1e504:	01070833          	add	a6,a4,a6
   1e508:	01e88f33          	add	t5,a7,t5
   1e50c:	00a73533          	sltu	a0,a4,a0
   1e510:	00af0533          	add	a0,t5,a0
   1e514:	00d806b3          	add	a3,a6,a3
   1e518:	007503b3          	add	t2,a0,t2
   1e51c:	00e83733          	sltu	a4,a6,a4
   1e520:	06d12023          	sw	a3,96(sp)
   1e524:	0106b6b3          	sltu	a3,a3,a6
   1e528:	037a8833          	mul	a6,s5,s7
   1e52c:	00e38733          	add	a4,t2,a4
   1e530:	01d70eb3          	add	t4,a4,t4
   1e534:	00de86b3          	add	a3,t4,a3
   1e538:	0107de13          	srli	t3,a5,0x10
   1e53c:	011f38b3          	sltu	a7,t5,a7
   1e540:	000104b7          	lui	s1,0x10
   1e544:	01e53f33          	sltu	t5,a0,t5
   1e548:	00a3b533          	sltu	a0,t2,a0
   1e54c:	007733b3          	sltu	t2,a4,t2
   1e550:	03590ab3          	mul	s5,s2,s5
   1e554:	013e0e33          	add	t3,t3,s3
   1e558:	00756533          	or	a0,a0,t2
   1e55c:	fff48993          	addi	s3,s1,-1 # ffff <exit-0xb5>
   1e560:	00eeb733          	sltu	a4,t4,a4
   1e564:	01e8e8b3          	or	a7,a7,t5
   1e568:	01d6beb3          	sltu	t4,a3,t4
   1e56c:	00a888b3          	add	a7,a7,a0
   1e570:	0137f7b3          	and	a5,a5,s3
   1e574:	01d76733          	or	a4,a4,t4
   1e578:	032b0933          	mul	s2,s6,s2
   1e57c:	00b685b3          	add	a1,a3,a1
   1e580:	01170733          	add	a4,a4,a7
   1e584:	01079793          	slli	a5,a5,0x10
   1e588:	0132f2b3          	and	t0,t0,s3
   1e58c:	01f70fb3          	add	t6,a4,t6
   1e590:	005787b3          	add	a5,a5,t0
   1e594:	00d5b6b3          	sltu	a3,a1,a3
   1e598:	00df86b3          	add	a3,t6,a3
   1e59c:	00f587b3          	add	a5,a1,a5
   1e5a0:	037b0b33          	mul	s6,s6,s7
   1e5a4:	00efb733          	sltu	a4,t6,a4
   1e5a8:	01c68e33          	add	t3,a3,t3
   1e5ac:	01f6bfb3          	sltu	t6,a3,t6
   1e5b0:	06f12223          	sw	a5,100(sp)
   1e5b4:	00b7b7b3          	sltu	a5,a5,a1
   1e5b8:	01f76533          	or	a0,a4,t6
   1e5bc:	00fe07b3          	add	a5,t3,a5
   1e5c0:	01085713          	srli	a4,a6,0x10
   1e5c4:	00de36b3          	sltu	a3,t3,a3
   1e5c8:	015b0b33          	add	s6,s6,s5
   1e5cc:	01c7be33          	sltu	t3,a5,t3
   1e5d0:	01670733          	add	a4,a4,s6
   1e5d4:	01c6e6b3          	or	a3,a3,t3
   1e5d8:	01577463          	bgeu	a4,s5,1e5e0 <__multf3+0x81c>
   1e5dc:	00990933          	add	s2,s2,s1
   1e5e0:	01075593          	srli	a1,a4,0x10
   1e5e4:	00a585b3          	add	a1,a1,a0
   1e5e8:	00010537          	lui	a0,0x10
   1e5ec:	fff50513          	addi	a0,a0,-1 # ffff <exit-0xb5>
   1e5f0:	00a77733          	and	a4,a4,a0
   1e5f4:	01071713          	slli	a4,a4,0x10
   1e5f8:	00a87833          	and	a6,a6,a0
   1e5fc:	01070733          	add	a4,a4,a6
   1e600:	00e78733          	add	a4,a5,a4
   1e604:	00d586b3          	add	a3,a1,a3
   1e608:	00f737b3          	sltu	a5,a4,a5
   1e60c:	00f687b3          	add	a5,a3,a5
   1e610:	012787b3          	add	a5,a5,s2
   1e614:	06f12623          	sw	a5,108(sp)
   1e618:	01012783          	lw	a5,16(sp)
   1e61c:	00d31313          	slli	t1,t1,0xd
   1e620:	06e12423          	sw	a4,104(sp)
   1e624:	00c7e7b3          	or	a5,a5,a2
   1e628:	0147e7b3          	or	a5,a5,s4
   1e62c:	00f36333          	or	t1,t1,a5
   1e630:	06010613          	addi	a2,sp,96
   1e634:	05010793          	addi	a5,sp,80
   1e638:	00c7a703          	lw	a4,12(a5)
   1e63c:	0107a683          	lw	a3,16(a5)
   1e640:	00478793          	addi	a5,a5,4
   1e644:	01375713          	srli	a4,a4,0x13
   1e648:	00d69693          	slli	a3,a3,0xd
   1e64c:	00d76733          	or	a4,a4,a3
   1e650:	fee7ae23          	sw	a4,-4(a5)
   1e654:	fef612e3          	bne	a2,a5,1e638 <__multf3+0x874>
   1e658:	05012783          	lw	a5,80(sp)
   1e65c:	00603333          	snez	t1,t1
   1e660:	05c12703          	lw	a4,92(sp)
   1e664:	00f36333          	or	t1,t1,a5
   1e668:	05812783          	lw	a5,88(sp)
   1e66c:	04e12623          	sw	a4,76(sp)
   1e670:	04612023          	sw	t1,64(sp)
   1e674:	04f12423          	sw	a5,72(sp)
   1e678:	05412783          	lw	a5,84(sp)
   1e67c:	04f12223          	sw	a5,68(sp)
   1e680:	00b71793          	slli	a5,a4,0xb
   1e684:	0407d863          	bgez	a5,1e6d4 <__multf3+0x910>
   1e688:	01f31313          	slli	t1,t1,0x1f
   1e68c:	04010793          	addi	a5,sp,64
   1e690:	04c10593          	addi	a1,sp,76
   1e694:	0007a683          	lw	a3,0(a5)
   1e698:	0047a603          	lw	a2,4(a5)
   1e69c:	00478793          	addi	a5,a5,4
   1e6a0:	0016d693          	srli	a3,a3,0x1
   1e6a4:	01f61613          	slli	a2,a2,0x1f
   1e6a8:	00c6e6b3          	or	a3,a3,a2
   1e6ac:	fed7ae23          	sw	a3,-4(a5)
   1e6b0:	fef592e3          	bne	a1,a5,1e694 <__multf3+0x8d0>
   1e6b4:	04012783          	lw	a5,64(sp)
   1e6b8:	00603333          	snez	t1,t1
   1e6bc:	00175713          	srli	a4,a4,0x1
   1e6c0:	0067e7b3          	or	a5,a5,t1
   1e6c4:	04f12023          	sw	a5,64(sp)
   1e6c8:	00812783          	lw	a5,8(sp)
   1e6cc:	04e12623          	sw	a4,76(sp)
   1e6d0:	00f12623          	sw	a5,12(sp)
   1e6d4:	00c12783          	lw	a5,12(sp)
   1e6d8:	00f12423          	sw	a5,8(sp)
   1e6dc:	3780006f          	j	1ea54 <__multf3+0xc90>
   1e6e0:	00a7e633          	or	a2,a5,a0
   1e6e4:	00d66633          	or	a2,a2,a3
   1e6e8:	00e66633          	or	a2,a2,a4
   1e6ec:	14060463          	beqz	a2,1e834 <__multf3+0xa70>
   1e6f0:	0a070063          	beqz	a4,1e790 <__multf3+0x9cc>
   1e6f4:	00070513          	mv	a0,a4
   1e6f8:	044020ef          	jal	2073c <__clzsi2>
   1e6fc:	ff450713          	addi	a4,a0,-12
   1e700:	40575593          	srai	a1,a4,0x5
   1e704:	01f77713          	andi	a4,a4,31
   1e708:	0a070e63          	beqz	a4,1e7c4 <__multf3+0xa00>
   1e70c:	ffc00693          	li	a3,-4
   1e710:	02d586b3          	mul	a3,a1,a3
   1e714:	02000813          	li	a6,32
   1e718:	02010313          	addi	t1,sp,32
   1e71c:	40e80833          	sub	a6,a6,a4
   1e720:	00c68793          	addi	a5,a3,12 # 1000c <exit-0xa8>
   1e724:	00f307b3          	add	a5,t1,a5
   1e728:	40d006b3          	neg	a3,a3
   1e72c:	0cf31463          	bne	t1,a5,1e7f4 <__multf3+0xa30>
   1e730:	fff58793          	addi	a5,a1,-1
   1e734:	00259593          	slli	a1,a1,0x2
   1e738:	05058693          	addi	a3,a1,80
   1e73c:	02010613          	addi	a2,sp,32
   1e740:	00c685b3          	add	a1,a3,a2
   1e744:	02012683          	lw	a3,32(sp)
   1e748:	00e69733          	sll	a4,a3,a4
   1e74c:	fae5a823          	sw	a4,-80(a1)
   1e750:	00178793          	addi	a5,a5,1
   1e754:	00279793          	slli	a5,a5,0x2
   1e758:	00800693          	li	a3,8
   1e75c:	02010713          	addi	a4,sp,32
   1e760:	00d7ea63          	bltu	a5,a3,1e774 <__multf3+0x9b0>
   1e764:	02012023          	sw	zero,32(sp)
   1e768:	00072223          	sw	zero,4(a4)
   1e76c:	ff878793          	addi	a5,a5,-8
   1e770:	02810713          	addi	a4,sp,40
   1e774:	00400693          	li	a3,4
   1e778:	00d7e463          	bltu	a5,a3,1e780 <__multf3+0x9bc>
   1e77c:	00072023          	sw	zero,0(a4)
   1e780:	ffffc4b7          	lui	s1,0xffffc
   1e784:	01148493          	addi	s1,s1,17 # ffffc011 <__BSS_END__+0xfffd9211>
   1e788:	40a484b3          	sub	s1,s1,a0
   1e78c:	f20ff06f          	j	1deac <__multf3+0xe8>
   1e790:	00068a63          	beqz	a3,1e7a4 <__multf3+0x9e0>
   1e794:	00068513          	mv	a0,a3
   1e798:	7a5010ef          	jal	2073c <__clzsi2>
   1e79c:	02050513          	addi	a0,a0,32
   1e7a0:	f5dff06f          	j	1e6fc <__multf3+0x938>
   1e7a4:	00050863          	beqz	a0,1e7b4 <__multf3+0x9f0>
   1e7a8:	795010ef          	jal	2073c <__clzsi2>
   1e7ac:	04050513          	addi	a0,a0,64
   1e7b0:	f4dff06f          	j	1e6fc <__multf3+0x938>
   1e7b4:	00078513          	mv	a0,a5
   1e7b8:	785010ef          	jal	2073c <__clzsi2>
   1e7bc:	06050513          	addi	a0,a0,96
   1e7c0:	f3dff06f          	j	1e6fc <__multf3+0x938>
   1e7c4:	ffc00693          	li	a3,-4
   1e7c8:	02d586b3          	mul	a3,a1,a3
   1e7cc:	02c10793          	addi	a5,sp,44
   1e7d0:	00300713          	li	a4,3
   1e7d4:	00d78633          	add	a2,a5,a3
   1e7d8:	00062603          	lw	a2,0(a2)
   1e7dc:	fff70713          	addi	a4,a4,-1
   1e7e0:	ffc78793          	addi	a5,a5,-4
   1e7e4:	00c7a223          	sw	a2,4(a5)
   1e7e8:	feb756e3          	bge	a4,a1,1e7d4 <__multf3+0xa10>
   1e7ec:	fff58793          	addi	a5,a1,-1
   1e7f0:	f61ff06f          	j	1e750 <__multf3+0x98c>
   1e7f4:	0007a603          	lw	a2,0(a5)
   1e7f8:	ffc7a883          	lw	a7,-4(a5)
   1e7fc:	00d78e33          	add	t3,a5,a3
   1e800:	00e61633          	sll	a2,a2,a4
   1e804:	0108d8b3          	srl	a7,a7,a6
   1e808:	01166633          	or	a2,a2,a7
   1e80c:	00ce2023          	sw	a2,0(t3)
   1e810:	ffc78793          	addi	a5,a5,-4
   1e814:	f19ff06f          	j	1e72c <__multf3+0x968>
   1e818:	00a7e7b3          	or	a5,a5,a0
   1e81c:	00d7e7b3          	or	a5,a5,a3
   1e820:	00e7e7b3          	or	a5,a5,a4
   1e824:	00200b93          	li	s7,2
   1e828:	e8078463          	beqz	a5,1deb0 <__multf3+0xec>
   1e82c:	00300b93          	li	s7,3
   1e830:	e80ff06f          	j	1deb0 <__multf3+0xec>
   1e834:	00000493          	li	s1,0
   1e838:	00100b93          	li	s7,1
   1e83c:	e74ff06f          	j	1deb0 <__multf3+0xec>
   1e840:	016967b3          	or	a5,s2,s6
   1e844:	0157e7b3          	or	a5,a5,s5
   1e848:	00a7e7b3          	or	a5,a5,a0
   1e84c:	14078463          	beqz	a5,1e994 <__multf3+0xbd0>
   1e850:	08050e63          	beqz	a0,1e8ec <__multf3+0xb28>
   1e854:	6e9010ef          	jal	2073c <__clzsi2>
   1e858:	ff450693          	addi	a3,a0,-12
   1e85c:	4056d793          	srai	a5,a3,0x5
   1e860:	01f6f693          	andi	a3,a3,31
   1e864:	0c068063          	beqz	a3,1e924 <__multf3+0xb60>
   1e868:	ffc00613          	li	a2,-4
   1e86c:	02c78633          	mul	a2,a5,a2
   1e870:	02000813          	li	a6,32
   1e874:	03010313          	addi	t1,sp,48
   1e878:	40d80833          	sub	a6,a6,a3
   1e87c:	00c60713          	addi	a4,a2,12
   1e880:	00e30733          	add	a4,t1,a4
   1e884:	40c00633          	neg	a2,a2
   1e888:	0ce31663          	bne	t1,a4,1e954 <__multf3+0xb90>
   1e88c:	fff78713          	addi	a4,a5,-1
   1e890:	00279793          	slli	a5,a5,0x2
   1e894:	02010613          	addi	a2,sp,32
   1e898:	05078793          	addi	a5,a5,80
   1e89c:	00c787b3          	add	a5,a5,a2
   1e8a0:	03012603          	lw	a2,48(sp)
   1e8a4:	00d616b3          	sll	a3,a2,a3
   1e8a8:	fcd7a023          	sw	a3,-64(a5)
   1e8ac:	00170793          	addi	a5,a4,1
   1e8b0:	00279793          	slli	a5,a5,0x2
   1e8b4:	00800693          	li	a3,8
   1e8b8:	03010713          	addi	a4,sp,48
   1e8bc:	00d7ea63          	bltu	a5,a3,1e8d0 <__multf3+0xb0c>
   1e8c0:	02012823          	sw	zero,48(sp)
   1e8c4:	00072223          	sw	zero,4(a4)
   1e8c8:	ff878793          	addi	a5,a5,-8
   1e8cc:	03810713          	addi	a4,sp,56
   1e8d0:	00400693          	li	a3,4
   1e8d4:	00d7e463          	bltu	a5,a3,1e8dc <__multf3+0xb18>
   1e8d8:	00072023          	sw	zero,0(a4)
   1e8dc:	ffffc7b7          	lui	a5,0xffffc
   1e8e0:	01178793          	addi	a5,a5,17 # ffffc011 <__BSS_END__+0xfffd9211>
   1e8e4:	40a787b3          	sub	a5,a5,a0
   1e8e8:	e54ff06f          	j	1df3c <__multf3+0x178>
   1e8ec:	000a8a63          	beqz	s5,1e900 <__multf3+0xb3c>
   1e8f0:	000a8513          	mv	a0,s5
   1e8f4:	649010ef          	jal	2073c <__clzsi2>
   1e8f8:	02050513          	addi	a0,a0,32
   1e8fc:	f5dff06f          	j	1e858 <__multf3+0xa94>
   1e900:	000b0a63          	beqz	s6,1e914 <__multf3+0xb50>
   1e904:	000b0513          	mv	a0,s6
   1e908:	635010ef          	jal	2073c <__clzsi2>
   1e90c:	04050513          	addi	a0,a0,64
   1e910:	f49ff06f          	j	1e858 <__multf3+0xa94>
   1e914:	00090513          	mv	a0,s2
   1e918:	625010ef          	jal	2073c <__clzsi2>
   1e91c:	06050513          	addi	a0,a0,96
   1e920:	f39ff06f          	j	1e858 <__multf3+0xa94>
   1e924:	ffc00613          	li	a2,-4
   1e928:	02c78633          	mul	a2,a5,a2
   1e92c:	03c10713          	addi	a4,sp,60
   1e930:	00300693          	li	a3,3
   1e934:	00c705b3          	add	a1,a4,a2
   1e938:	0005a583          	lw	a1,0(a1)
   1e93c:	fff68693          	addi	a3,a3,-1
   1e940:	ffc70713          	addi	a4,a4,-4
   1e944:	00b72223          	sw	a1,4(a4)
   1e948:	fef6d6e3          	bge	a3,a5,1e934 <__multf3+0xb70>
   1e94c:	fff78713          	addi	a4,a5,-1
   1e950:	f5dff06f          	j	1e8ac <__multf3+0xae8>
   1e954:	00072583          	lw	a1,0(a4)
   1e958:	ffc72883          	lw	a7,-4(a4)
   1e95c:	00c70e33          	add	t3,a4,a2
   1e960:	00d595b3          	sll	a1,a1,a3
   1e964:	0108d8b3          	srl	a7,a7,a6
   1e968:	0115e5b3          	or	a1,a1,a7
   1e96c:	00be2023          	sw	a1,0(t3)
   1e970:	ffc70713          	addi	a4,a4,-4
   1e974:	f15ff06f          	j	1e888 <__multf3+0xac4>
   1e978:	01696933          	or	s2,s2,s6
   1e97c:	01596933          	or	s2,s2,s5
   1e980:	00a96933          	or	s2,s2,a0
   1e984:	00200693          	li	a3,2
   1e988:	da090c63          	beqz	s2,1df40 <__multf3+0x17c>
   1e98c:	00300693          	li	a3,3
   1e990:	db0ff06f          	j	1df40 <__multf3+0x17c>
   1e994:	00000793          	li	a5,0
   1e998:	00100693          	li	a3,1
   1e99c:	da4ff06f          	j	1df40 <__multf3+0x17c>
   1e9a0:	00100713          	li	a4,1
   1e9a4:	00f717b3          	sll	a5,a4,a5
   1e9a8:	5307f713          	andi	a4,a5,1328
   1e9ac:	06071863          	bnez	a4,1ea1c <__multf3+0xc58>
   1e9b0:	0887f713          	andi	a4,a5,136
   1e9b4:	04071063          	bnez	a4,1e9f4 <__multf3+0xc30>
   1e9b8:	2407f793          	andi	a5,a5,576
   1e9bc:	dc078063          	beqz	a5,1df7c <__multf3+0x1b8>
   1e9c0:	000087b7          	lui	a5,0x8
   1e9c4:	04f12623          	sw	a5,76(sp)
   1e9c8:	04012423          	sw	zero,72(sp)
   1e9cc:	04012223          	sw	zero,68(sp)
   1e9d0:	04012023          	sw	zero,64(sp)
   1e9d4:	fff78793          	addi	a5,a5,-1 # 7fff <exit-0x80b5>
   1e9d8:	00012223          	sw	zero,4(sp)
   1e9dc:	1440006f          	j	1eb20 <__multf3+0xd5c>
   1e9e0:	00f00713          	li	a4,15
   1e9e4:	fce78ee3          	beq	a5,a4,1e9c0 <__multf3+0xbfc>
   1e9e8:	00b00713          	li	a4,11
   1e9ec:	01412223          	sw	s4,4(sp)
   1e9f0:	02e79663          	bne	a5,a4,1ea1c <__multf3+0xc58>
   1e9f4:	01312223          	sw	s3,4(sp)
   1e9f8:	03012783          	lw	a5,48(sp)
   1e9fc:	00068b93          	mv	s7,a3
   1ea00:	04f12023          	sw	a5,64(sp)
   1ea04:	03412783          	lw	a5,52(sp)
   1ea08:	04f12223          	sw	a5,68(sp)
   1ea0c:	03812783          	lw	a5,56(sp)
   1ea10:	04f12423          	sw	a5,72(sp)
   1ea14:	03c12783          	lw	a5,60(sp)
   1ea18:	0200006f          	j	1ea38 <__multf3+0xc74>
   1ea1c:	02012783          	lw	a5,32(sp)
   1ea20:	04f12023          	sw	a5,64(sp)
   1ea24:	02412783          	lw	a5,36(sp)
   1ea28:	04f12223          	sw	a5,68(sp)
   1ea2c:	02812783          	lw	a5,40(sp)
   1ea30:	04f12423          	sw	a5,72(sp)
   1ea34:	02c12783          	lw	a5,44(sp)
   1ea38:	04f12623          	sw	a5,76(sp)
   1ea3c:	00200793          	li	a5,2
   1ea40:	36fb8663          	beq	s7,a5,1edac <__multf3+0xfe8>
   1ea44:	00300793          	li	a5,3
   1ea48:	f6fb8ce3          	beq	s7,a5,1e9c0 <__multf3+0xbfc>
   1ea4c:	00100793          	li	a5,1
   1ea50:	34fb8463          	beq	s7,a5,1ed98 <__multf3+0xfd4>
   1ea54:	00812703          	lw	a4,8(sp)
   1ea58:	000047b7          	lui	a5,0x4
   1ea5c:	fff78793          	addi	a5,a5,-1 # 3fff <exit-0xc0b5>
   1ea60:	00f707b3          	add	a5,a4,a5
   1ea64:	12f05a63          	blez	a5,1eb98 <__multf3+0xdd4>
   1ea68:	04012703          	lw	a4,64(sp)
   1ea6c:	00777693          	andi	a3,a4,7
   1ea70:	04068463          	beqz	a3,1eab8 <__multf3+0xcf4>
   1ea74:	00f77693          	andi	a3,a4,15
   1ea78:	00400613          	li	a2,4
   1ea7c:	02c68e63          	beq	a3,a2,1eab8 <__multf3+0xcf4>
   1ea80:	00470713          	addi	a4,a4,4
   1ea84:	00473693          	sltiu	a3,a4,4
   1ea88:	04e12023          	sw	a4,64(sp)
   1ea8c:	04412703          	lw	a4,68(sp)
   1ea90:	00e68733          	add	a4,a3,a4
   1ea94:	04e12223          	sw	a4,68(sp)
   1ea98:	00d73733          	sltu	a4,a4,a3
   1ea9c:	04812683          	lw	a3,72(sp)
   1eaa0:	00e68733          	add	a4,a3,a4
   1eaa4:	04e12423          	sw	a4,72(sp)
   1eaa8:	00d73733          	sltu	a4,a4,a3
   1eaac:	04c12683          	lw	a3,76(sp)
   1eab0:	00d70733          	add	a4,a4,a3
   1eab4:	04e12623          	sw	a4,76(sp)
   1eab8:	04c12703          	lw	a4,76(sp)
   1eabc:	00b71693          	slli	a3,a4,0xb
   1eac0:	0206d063          	bgez	a3,1eae0 <__multf3+0xd1c>
   1eac4:	fff007b7          	lui	a5,0xfff00
   1eac8:	fff78793          	addi	a5,a5,-1 # ffefffff <__BSS_END__+0xffedd1ff>
   1eacc:	00f77733          	and	a4,a4,a5
   1ead0:	04e12623          	sw	a4,76(sp)
   1ead4:	00812703          	lw	a4,8(sp)
   1ead8:	000047b7          	lui	a5,0x4
   1eadc:	00f707b3          	add	a5,a4,a5
   1eae0:	04010713          	addi	a4,sp,64
   1eae4:	04c10593          	addi	a1,sp,76
   1eae8:	00072683          	lw	a3,0(a4)
   1eaec:	00472603          	lw	a2,4(a4)
   1eaf0:	00470713          	addi	a4,a4,4
   1eaf4:	0036d693          	srli	a3,a3,0x3
   1eaf8:	01d61613          	slli	a2,a2,0x1d
   1eafc:	00c6e6b3          	or	a3,a3,a2
   1eb00:	fed72e23          	sw	a3,-4(a4)
   1eb04:	fee592e3          	bne	a1,a4,1eae8 <__multf3+0xd24>
   1eb08:	000086b7          	lui	a3,0x8
   1eb0c:	ffe68693          	addi	a3,a3,-2 # 7ffe <exit-0x80b6>
   1eb10:	04c12703          	lw	a4,76(sp)
   1eb14:	28f6cc63          	blt	a3,a5,1edac <__multf3+0xfe8>
   1eb18:	00375713          	srli	a4,a4,0x3
   1eb1c:	04e12623          	sw	a4,76(sp)
   1eb20:	04c12703          	lw	a4,76(sp)
   1eb24:	0ac12083          	lw	ra,172(sp)
   1eb28:	00040513          	mv	a0,s0
   1eb2c:	04e11e23          	sh	a4,92(sp)
   1eb30:	00412703          	lw	a4,4(sp)
   1eb34:	0a412483          	lw	s1,164(sp)
   1eb38:	0a012903          	lw	s2,160(sp)
   1eb3c:	00f71713          	slli	a4,a4,0xf
   1eb40:	00f767b3          	or	a5,a4,a5
   1eb44:	04f11f23          	sh	a5,94(sp)
   1eb48:	04012783          	lw	a5,64(sp)
   1eb4c:	09c12983          	lw	s3,156(sp)
   1eb50:	09812a03          	lw	s4,152(sp)
   1eb54:	00f42023          	sw	a5,0(s0)
   1eb58:	04412783          	lw	a5,68(sp)
   1eb5c:	09412a83          	lw	s5,148(sp)
   1eb60:	09012b03          	lw	s6,144(sp)
   1eb64:	00f42223          	sw	a5,4(s0)
   1eb68:	04812783          	lw	a5,72(sp)
   1eb6c:	08c12b83          	lw	s7,140(sp)
   1eb70:	08812c03          	lw	s8,136(sp)
   1eb74:	00f42423          	sw	a5,8(s0)
   1eb78:	05c12783          	lw	a5,92(sp)
   1eb7c:	08412c83          	lw	s9,132(sp)
   1eb80:	08012d03          	lw	s10,128(sp)
   1eb84:	00f42623          	sw	a5,12(s0)
   1eb88:	0a812403          	lw	s0,168(sp)
   1eb8c:	07c12d83          	lw	s11,124(sp)
   1eb90:	0b010113          	addi	sp,sp,176
   1eb94:	00008067          	ret
   1eb98:	00100693          	li	a3,1
   1eb9c:	40f686b3          	sub	a3,a3,a5
   1eba0:	07400793          	li	a5,116
   1eba4:	1cd7ca63          	blt	a5,a3,1ed78 <__multf3+0xfb4>
   1eba8:	04010613          	addi	a2,sp,64
   1ebac:	4056d713          	srai	a4,a3,0x5
   1ebb0:	00060513          	mv	a0,a2
   1ebb4:	01f6f693          	andi	a3,a3,31
   1ebb8:	00000793          	li	a5,0
   1ebbc:	00000593          	li	a1,0
   1ebc0:	02e59e63          	bne	a1,a4,1ebfc <__multf3+0xe38>
   1ebc4:	00300593          	li	a1,3
   1ebc8:	40e585b3          	sub	a1,a1,a4
   1ebcc:	00271513          	slli	a0,a4,0x2
   1ebd0:	04069063          	bnez	a3,1ec10 <__multf3+0xe4c>
   1ebd4:	00060813          	mv	a6,a2
   1ebd8:	00a808b3          	add	a7,a6,a0
   1ebdc:	0008a883          	lw	a7,0(a7) # 10000 <exit-0xb4>
   1ebe0:	00168693          	addi	a3,a3,1
   1ebe4:	00480813          	addi	a6,a6,4
   1ebe8:	ff182e23          	sw	a7,-4(a6)
   1ebec:	fed5d6e3          	bge	a1,a3,1ebd8 <__multf3+0xe14>
   1ebf0:	00400693          	li	a3,4
   1ebf4:	40e68733          	sub	a4,a3,a4
   1ebf8:	06c0006f          	j	1ec64 <__multf3+0xea0>
   1ebfc:	00052803          	lw	a6,0(a0)
   1ec00:	00158593          	addi	a1,a1,1
   1ec04:	00450513          	addi	a0,a0,4
   1ec08:	0107e7b3          	or	a5,a5,a6
   1ec0c:	fb5ff06f          	j	1ebc0 <__multf3+0xdfc>
   1ec10:	05050813          	addi	a6,a0,80
   1ec14:	02010893          	addi	a7,sp,32
   1ec18:	01180833          	add	a6,a6,a7
   1ec1c:	fd082803          	lw	a6,-48(a6)
   1ec20:	02000313          	li	t1,32
   1ec24:	40d30333          	sub	t1,t1,a3
   1ec28:	00681833          	sll	a6,a6,t1
   1ec2c:	0107e7b3          	or	a5,a5,a6
   1ec30:	00000e13          	li	t3,0
   1ec34:	00a60833          	add	a6,a2,a0
   1ec38:	40a00533          	neg	a0,a0
   1ec3c:	0ebe4063          	blt	t3,a1,1ed1c <__multf3+0xf58>
   1ec40:	00400513          	li	a0,4
   1ec44:	00259593          	slli	a1,a1,0x2
   1ec48:	40e50733          	sub	a4,a0,a4
   1ec4c:	05058593          	addi	a1,a1,80
   1ec50:	02010513          	addi	a0,sp,32
   1ec54:	00a585b3          	add	a1,a1,a0
   1ec58:	04c12503          	lw	a0,76(sp)
   1ec5c:	00d556b3          	srl	a3,a0,a3
   1ec60:	fcd5a823          	sw	a3,-48(a1)
   1ec64:	00400693          	li	a3,4
   1ec68:	40e686b3          	sub	a3,a3,a4
   1ec6c:	00269693          	slli	a3,a3,0x2
   1ec70:	00271713          	slli	a4,a4,0x2
   1ec74:	00800593          	li	a1,8
   1ec78:	00e60733          	add	a4,a2,a4
   1ec7c:	00b6ea63          	bltu	a3,a1,1ec90 <__multf3+0xecc>
   1ec80:	00072023          	sw	zero,0(a4)
   1ec84:	00072223          	sw	zero,4(a4)
   1ec88:	ff868693          	addi	a3,a3,-8
   1ec8c:	00870713          	addi	a4,a4,8
   1ec90:	00400593          	li	a1,4
   1ec94:	00b6e463          	bltu	a3,a1,1ec9c <__multf3+0xed8>
   1ec98:	00072023          	sw	zero,0(a4)
   1ec9c:	04012703          	lw	a4,64(sp)
   1eca0:	00f037b3          	snez	a5,a5
   1eca4:	00e7e7b3          	or	a5,a5,a4
   1eca8:	04f12023          	sw	a5,64(sp)
   1ecac:	0077f713          	andi	a4,a5,7
   1ecb0:	04070463          	beqz	a4,1ecf8 <__multf3+0xf34>
   1ecb4:	00f7f713          	andi	a4,a5,15
   1ecb8:	00400693          	li	a3,4
   1ecbc:	02d70e63          	beq	a4,a3,1ecf8 <__multf3+0xf34>
   1ecc0:	04412703          	lw	a4,68(sp)
   1ecc4:	00478793          	addi	a5,a5,4 # 4004 <exit-0xc0b0>
   1ecc8:	04f12023          	sw	a5,64(sp)
   1eccc:	0047b793          	sltiu	a5,a5,4
   1ecd0:	00f707b3          	add	a5,a4,a5
   1ecd4:	04f12223          	sw	a5,68(sp)
   1ecd8:	00e7b7b3          	sltu	a5,a5,a4
   1ecdc:	04812703          	lw	a4,72(sp)
   1ece0:	00f707b3          	add	a5,a4,a5
   1ece4:	04f12423          	sw	a5,72(sp)
   1ece8:	00e7b7b3          	sltu	a5,a5,a4
   1ecec:	04c12703          	lw	a4,76(sp)
   1ecf0:	00e787b3          	add	a5,a5,a4
   1ecf4:	04f12623          	sw	a5,76(sp)
   1ecf8:	04c12703          	lw	a4,76(sp)
   1ecfc:	00c71793          	slli	a5,a4,0xc
   1ed00:	0407d263          	bgez	a5,1ed44 <__multf3+0xf80>
   1ed04:	04012623          	sw	zero,76(sp)
   1ed08:	04012423          	sw	zero,72(sp)
   1ed0c:	04012223          	sw	zero,68(sp)
   1ed10:	04012023          	sw	zero,64(sp)
   1ed14:	00100793          	li	a5,1
   1ed18:	e09ff06f          	j	1eb20 <__multf3+0xd5c>
   1ed1c:	00082883          	lw	a7,0(a6)
   1ed20:	00482e83          	lw	t4,4(a6)
   1ed24:	00a80f33          	add	t5,a6,a0
   1ed28:	00d8d8b3          	srl	a7,a7,a3
   1ed2c:	006e9eb3          	sll	t4,t4,t1
   1ed30:	01d8e8b3          	or	a7,a7,t4
   1ed34:	011f2023          	sw	a7,0(t5)
   1ed38:	001e0e13          	addi	t3,t3,1
   1ed3c:	00480813          	addi	a6,a6,4
   1ed40:	efdff06f          	j	1ec3c <__multf3+0xe78>
   1ed44:	00c60593          	addi	a1,a2,12
   1ed48:	00062783          	lw	a5,0(a2)
   1ed4c:	00462683          	lw	a3,4(a2)
   1ed50:	00460613          	addi	a2,a2,4
   1ed54:	0037d793          	srli	a5,a5,0x3
   1ed58:	01d69693          	slli	a3,a3,0x1d
   1ed5c:	00d7e7b3          	or	a5,a5,a3
   1ed60:	fef62e23          	sw	a5,-4(a2)
   1ed64:	fec592e3          	bne	a1,a2,1ed48 <__multf3+0xf84>
   1ed68:	00375713          	srli	a4,a4,0x3
   1ed6c:	04e12623          	sw	a4,76(sp)
   1ed70:	00000793          	li	a5,0
   1ed74:	dadff06f          	j	1eb20 <__multf3+0xd5c>
   1ed78:	04412703          	lw	a4,68(sp)
   1ed7c:	04012783          	lw	a5,64(sp)
   1ed80:	00e7e7b3          	or	a5,a5,a4
   1ed84:	04812703          	lw	a4,72(sp)
   1ed88:	00e7e7b3          	or	a5,a5,a4
   1ed8c:	04c12703          	lw	a4,76(sp)
   1ed90:	00e7e7b3          	or	a5,a5,a4
   1ed94:	fc078ee3          	beqz	a5,1ed70 <__multf3+0xfac>
   1ed98:	04012623          	sw	zero,76(sp)
   1ed9c:	04012423          	sw	zero,72(sp)
   1eda0:	04012223          	sw	zero,68(sp)
   1eda4:	04012023          	sw	zero,64(sp)
   1eda8:	fc9ff06f          	j	1ed70 <__multf3+0xfac>
   1edac:	000087b7          	lui	a5,0x8
   1edb0:	04012623          	sw	zero,76(sp)
   1edb4:	04012423          	sw	zero,72(sp)
   1edb8:	04012223          	sw	zero,68(sp)
   1edbc:	04012023          	sw	zero,64(sp)
   1edc0:	fff78793          	addi	a5,a5,-1 # 7fff <exit-0x80b5>
   1edc4:	d5dff06f          	j	1eb20 <__multf3+0xd5c>

0001edc8 <__subtf3>:
   1edc8:	f9010113          	addi	sp,sp,-112
   1edcc:	0085a703          	lw	a4,8(a1)
   1edd0:	05512a23          	sw	s5,84(sp)
   1edd4:	00c5aa83          	lw	s5,12(a1)
   1edd8:	0005a783          	lw	a5,0(a1)
   1eddc:	0045a683          	lw	a3,4(a1)
   1ede0:	02e12c23          	sw	a4,56(sp)
   1ede4:	00e12c23          	sw	a4,24(sp)
   1ede8:	010a9713          	slli	a4,s5,0x10
   1edec:	06912223          	sw	s1,100(sp)
   1edf0:	01075713          	srli	a4,a4,0x10
   1edf4:	001a9493          	slli	s1,s5,0x1
   1edf8:	00462803          	lw	a6,4(a2)
   1edfc:	00862583          	lw	a1,8(a2)
   1ee00:	06812423          	sw	s0,104(sp)
   1ee04:	07212023          	sw	s2,96(sp)
   1ee08:	00062403          	lw	s0,0(a2)
   1ee0c:	00c62903          	lw	s2,12(a2)
   1ee10:	05412c23          	sw	s4,88(sp)
   1ee14:	03512e23          	sw	s5,60(sp)
   1ee18:	00050a13          	mv	s4,a0
   1ee1c:	06112623          	sw	ra,108(sp)
   1ee20:	05312e23          	sw	s3,92(sp)
   1ee24:	05612823          	sw	s6,80(sp)
   1ee28:	05712623          	sw	s7,76(sp)
   1ee2c:	05812423          	sw	s8,72(sp)
   1ee30:	02f12823          	sw	a5,48(sp)
   1ee34:	02d12a23          	sw	a3,52(sp)
   1ee38:	00f12823          	sw	a5,16(sp)
   1ee3c:	00d12a23          	sw	a3,20(sp)
   1ee40:	00e12e23          	sw	a4,28(sp)
   1ee44:	0114d493          	srli	s1,s1,0x11
   1ee48:	01fada93          	srli	s5,s5,0x1f
   1ee4c:	01010513          	addi	a0,sp,16
   1ee50:	01c10613          	addi	a2,sp,28
   1ee54:	00062703          	lw	a4,0(a2)
   1ee58:	ffc62683          	lw	a3,-4(a2)
   1ee5c:	ffc60613          	addi	a2,a2,-4
   1ee60:	00371713          	slli	a4,a4,0x3
   1ee64:	01d6d693          	srli	a3,a3,0x1d
   1ee68:	00d76733          	or	a4,a4,a3
   1ee6c:	00e62223          	sw	a4,4(a2)
   1ee70:	fec512e3          	bne	a0,a2,1ee54 <__subtf3+0x8c>
   1ee74:	01091713          	slli	a4,s2,0x10
   1ee78:	00191b93          	slli	s7,s2,0x1
   1ee7c:	00379793          	slli	a5,a5,0x3
   1ee80:	01075713          	srli	a4,a4,0x10
   1ee84:	03012a23          	sw	a6,52(sp)
   1ee88:	03212e23          	sw	s2,60(sp)
   1ee8c:	03012223          	sw	a6,36(sp)
   1ee90:	00f12823          	sw	a5,16(sp)
   1ee94:	02812823          	sw	s0,48(sp)
   1ee98:	02b12c23          	sw	a1,56(sp)
   1ee9c:	02812023          	sw	s0,32(sp)
   1eea0:	02b12423          	sw	a1,40(sp)
   1eea4:	02e12623          	sw	a4,44(sp)
   1eea8:	011bdb93          	srli	s7,s7,0x11
   1eeac:	01f95913          	srli	s2,s2,0x1f
   1eeb0:	02010813          	addi	a6,sp,32
   1eeb4:	02c10313          	addi	t1,sp,44
   1eeb8:	00032703          	lw	a4,0(t1)
   1eebc:	ffc32683          	lw	a3,-4(t1)
   1eec0:	ffc30313          	addi	t1,t1,-4
   1eec4:	00371713          	slli	a4,a4,0x3
   1eec8:	01d6d693          	srli	a3,a3,0x1d
   1eecc:	00d76733          	or	a4,a4,a3
   1eed0:	00e32223          	sw	a4,4(t1)
   1eed4:	fe6812e3          	bne	a6,t1,1eeb8 <__subtf3+0xf0>
   1eed8:	00341413          	slli	s0,s0,0x3
   1eedc:	00008737          	lui	a4,0x8
   1eee0:	02812023          	sw	s0,32(sp)
   1eee4:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1eee8:	02eb9063          	bne	s7,a4,1ef08 <__subtf3+0x140>
   1eeec:	02812683          	lw	a3,40(sp)
   1eef0:	02412703          	lw	a4,36(sp)
   1eef4:	00d76733          	or	a4,a4,a3
   1eef8:	02c12683          	lw	a3,44(sp)
   1eefc:	00d76733          	or	a4,a4,a3
   1ef00:	00876733          	or	a4,a4,s0
   1ef04:	00071463          	bnez	a4,1ef0c <__subtf3+0x144>
   1ef08:	00194913          	xori	s2,s2,1
   1ef0c:	417488b3          	sub	a7,s1,s7
   1ef10:	095916e3          	bne	s2,s5,1f79c <__subtf3+0x9d4>
   1ef14:	45105263          	blez	a7,1f358 <__subtf3+0x590>
   1ef18:	01412903          	lw	s2,20(sp)
   1ef1c:	01812983          	lw	s3,24(sp)
   1ef20:	01c12b03          	lw	s6,28(sp)
   1ef24:	0a0b9263          	bnez	s7,1efc8 <__subtf3+0x200>
   1ef28:	02412683          	lw	a3,36(sp)
   1ef2c:	02812703          	lw	a4,40(sp)
   1ef30:	02c12583          	lw	a1,44(sp)
   1ef34:	00e6e633          	or	a2,a3,a4
   1ef38:	00b66633          	or	a2,a2,a1
   1ef3c:	00866633          	or	a2,a2,s0
   1ef40:	00061e63          	bnez	a2,1ef5c <__subtf3+0x194>
   1ef44:	02f12823          	sw	a5,48(sp)
   1ef48:	03212a23          	sw	s2,52(sp)
   1ef4c:	03312c23          	sw	s3,56(sp)
   1ef50:	03612e23          	sw	s6,60(sp)
   1ef54:	00088493          	mv	s1,a7
   1ef58:	08c0006f          	j	1efe4 <__subtf3+0x21c>
   1ef5c:	fff88613          	addi	a2,a7,-1
   1ef60:	04061863          	bnez	a2,1efb0 <__subtf3+0x1e8>
   1ef64:	00878433          	add	s0,a5,s0
   1ef68:	01268933          	add	s2,a3,s2
   1ef6c:	02812823          	sw	s0,48(sp)
   1ef70:	00f43433          	sltu	s0,s0,a5
   1ef74:	00890433          	add	s0,s2,s0
   1ef78:	02812a23          	sw	s0,52(sp)
   1ef7c:	00d936b3          	sltu	a3,s2,a3
   1ef80:	01243433          	sltu	s0,s0,s2
   1ef84:	013709b3          	add	s3,a4,s3
   1ef88:	0086e6b3          	or	a3,a3,s0
   1ef8c:	00d986b3          	add	a3,s3,a3
   1ef90:	02d12c23          	sw	a3,56(sp)
   1ef94:	00e9b7b3          	sltu	a5,s3,a4
   1ef98:	0136b6b3          	sltu	a3,a3,s3
   1ef9c:	00d7e7b3          	or	a5,a5,a3
   1efa0:	016585b3          	add	a1,a1,s6
   1efa4:	00b787b3          	add	a5,a5,a1
   1efa8:	00100493          	li	s1,1
   1efac:	2fc0006f          	j	1f2a8 <__subtf3+0x4e0>
   1efb0:	00008737          	lui	a4,0x8
   1efb4:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1efb8:	00e88463          	beq	a7,a4,1efc0 <__subtf3+0x1f8>
   1efbc:	2500106f          	j	2020c <__subtf3+0x1444>
   1efc0:	02f12823          	sw	a5,48(sp)
   1efc4:	4400006f          	j	1f404 <__subtf3+0x63c>
   1efc8:	00008737          	lui	a4,0x8
   1efcc:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1efd0:	16e49a63          	bne	s1,a4,1f144 <__subtf3+0x37c>
   1efd4:	02f12823          	sw	a5,48(sp)
   1efd8:	03212a23          	sw	s2,52(sp)
   1efdc:	03312c23          	sw	s3,56(sp)
   1efe0:	03612e23          	sw	s6,60(sp)
   1efe4:	03012783          	lw	a5,48(sp)
   1efe8:	0077f713          	andi	a4,a5,7
   1efec:	04070463          	beqz	a4,1f034 <__subtf3+0x26c>
   1eff0:	00f7f713          	andi	a4,a5,15
   1eff4:	00400693          	li	a3,4
   1eff8:	02d70e63          	beq	a4,a3,1f034 <__subtf3+0x26c>
   1effc:	03412703          	lw	a4,52(sp)
   1f000:	00478793          	addi	a5,a5,4
   1f004:	02f12823          	sw	a5,48(sp)
   1f008:	0047b793          	sltiu	a5,a5,4
   1f00c:	00f707b3          	add	a5,a4,a5
   1f010:	02f12a23          	sw	a5,52(sp)
   1f014:	00e7b7b3          	sltu	a5,a5,a4
   1f018:	03812703          	lw	a4,56(sp)
   1f01c:	00f707b3          	add	a5,a4,a5
   1f020:	02f12c23          	sw	a5,56(sp)
   1f024:	00e7b7b3          	sltu	a5,a5,a4
   1f028:	03c12703          	lw	a4,60(sp)
   1f02c:	00e787b3          	add	a5,a5,a4
   1f030:	02f12e23          	sw	a5,60(sp)
   1f034:	03c12783          	lw	a5,60(sp)
   1f038:	00c79713          	slli	a4,a5,0xc
   1f03c:	02075463          	bgez	a4,1f064 <__subtf3+0x29c>
   1f040:	00008737          	lui	a4,0x8
   1f044:	00148493          	addi	s1,s1,1
   1f048:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1f04c:	00e49463          	bne	s1,a4,1f054 <__subtf3+0x28c>
   1f050:	1a80106f          	j	201f8 <__subtf3+0x1430>
   1f054:	fff80737          	lui	a4,0xfff80
   1f058:	fff70713          	addi	a4,a4,-1 # fff7ffff <__BSS_END__+0xfff5d1ff>
   1f05c:	00e7f7b3          	and	a5,a5,a4
   1f060:	02f12e23          	sw	a5,60(sp)
   1f064:	03010793          	addi	a5,sp,48
   1f068:	03c10613          	addi	a2,sp,60
   1f06c:	0007a703          	lw	a4,0(a5)
   1f070:	0047a683          	lw	a3,4(a5)
   1f074:	00478793          	addi	a5,a5,4
   1f078:	00375713          	srli	a4,a4,0x3
   1f07c:	01d69693          	slli	a3,a3,0x1d
   1f080:	00d76733          	or	a4,a4,a3
   1f084:	fee7ae23          	sw	a4,-4(a5)
   1f088:	fec792e3          	bne	a5,a2,1f06c <__subtf3+0x2a4>
   1f08c:	03c12703          	lw	a4,60(sp)
   1f090:	000086b7          	lui	a3,0x8
   1f094:	fff68793          	addi	a5,a3,-1 # 7fff <exit-0x80b5>
   1f098:	00375713          	srli	a4,a4,0x3
   1f09c:	02e12e23          	sw	a4,60(sp)
   1f0a0:	02f49a63          	bne	s1,a5,1f0d4 <__subtf3+0x30c>
   1f0a4:	03412603          	lw	a2,52(sp)
   1f0a8:	03012783          	lw	a5,48(sp)
   1f0ac:	00c7e7b3          	or	a5,a5,a2
   1f0b0:	03812603          	lw	a2,56(sp)
   1f0b4:	00c7e7b3          	or	a5,a5,a2
   1f0b8:	00e7e7b3          	or	a5,a5,a4
   1f0bc:	00078c63          	beqz	a5,1f0d4 <__subtf3+0x30c>
   1f0c0:	02d12e23          	sw	a3,60(sp)
   1f0c4:	02012c23          	sw	zero,56(sp)
   1f0c8:	02012a23          	sw	zero,52(sp)
   1f0cc:	02012823          	sw	zero,48(sp)
   1f0d0:	00000a93          	li	s5,0
   1f0d4:	03c12783          	lw	a5,60(sp)
   1f0d8:	01149493          	slli	s1,s1,0x11
   1f0dc:	0114d493          	srli	s1,s1,0x11
   1f0e0:	00f11623          	sh	a5,12(sp)
   1f0e4:	03012783          	lw	a5,48(sp)
   1f0e8:	00fa9a93          	slli	s5,s5,0xf
   1f0ec:	009aeab3          	or	s5,s5,s1
   1f0f0:	00fa2023          	sw	a5,0(s4)
   1f0f4:	03412783          	lw	a5,52(sp)
   1f0f8:	01511723          	sh	s5,14(sp)
   1f0fc:	06c12083          	lw	ra,108(sp)
   1f100:	00fa2223          	sw	a5,4(s4)
   1f104:	03812783          	lw	a5,56(sp)
   1f108:	06812403          	lw	s0,104(sp)
   1f10c:	06412483          	lw	s1,100(sp)
   1f110:	00fa2423          	sw	a5,8(s4)
   1f114:	00c12783          	lw	a5,12(sp)
   1f118:	06012903          	lw	s2,96(sp)
   1f11c:	05c12983          	lw	s3,92(sp)
   1f120:	00fa2623          	sw	a5,12(s4)
   1f124:	05412a83          	lw	s5,84(sp)
   1f128:	05012b03          	lw	s6,80(sp)
   1f12c:	04c12b83          	lw	s7,76(sp)
   1f130:	04812c03          	lw	s8,72(sp)
   1f134:	000a0513          	mv	a0,s4
   1f138:	05812a03          	lw	s4,88(sp)
   1f13c:	07010113          	addi	sp,sp,112
   1f140:	00008067          	ret
   1f144:	02c12703          	lw	a4,44(sp)
   1f148:	000806b7          	lui	a3,0x80
   1f14c:	00d76733          	or	a4,a4,a3
   1f150:	02e12623          	sw	a4,44(sp)
   1f154:	07400713          	li	a4,116
   1f158:	01175463          	bge	a4,a7,1f160 <__subtf3+0x398>
   1f15c:	0bc0106f          	j	20218 <__subtf3+0x1450>
   1f160:	00088613          	mv	a2,a7
   1f164:	40565693          	srai	a3,a2,0x5
   1f168:	00030513          	mv	a0,t1
   1f16c:	01f67613          	andi	a2,a2,31
   1f170:	00000713          	li	a4,0
   1f174:	00000593          	li	a1,0
   1f178:	02d59c63          	bne	a1,a3,1f1b0 <__subtf3+0x3e8>
   1f17c:	00300593          	li	a1,3
   1f180:	40d585b3          	sub	a1,a1,a3
   1f184:	00269513          	slli	a0,a3,0x2
   1f188:	02061e63          	bnez	a2,1f1c4 <__subtf3+0x3fc>
   1f18c:	00a308b3          	add	a7,t1,a0
   1f190:	0008a883          	lw	a7,0(a7)
   1f194:	00160613          	addi	a2,a2,1
   1f198:	00430313          	addi	t1,t1,4
   1f19c:	ff132e23          	sw	a7,-4(t1)
   1f1a0:	fec5d6e3          	bge	a1,a2,1f18c <__subtf3+0x3c4>
   1f1a4:	00400613          	li	a2,4
   1f1a8:	40d606b3          	sub	a3,a2,a3
   1f1ac:	0640006f          	j	1f210 <__subtf3+0x448>
   1f1b0:	00052883          	lw	a7,0(a0)
   1f1b4:	00158593          	addi	a1,a1,1
   1f1b8:	00450513          	addi	a0,a0,4
   1f1bc:	01176733          	or	a4,a4,a7
   1f1c0:	fb9ff06f          	j	1f178 <__subtf3+0x3b0>
   1f1c4:	04050893          	addi	a7,a0,64
   1f1c8:	002888b3          	add	a7,a7,sp
   1f1cc:	fe08a883          	lw	a7,-32(a7)
   1f1d0:	02000e13          	li	t3,32
   1f1d4:	40ce0e33          	sub	t3,t3,a2
   1f1d8:	01c898b3          	sll	a7,a7,t3
   1f1dc:	01176733          	or	a4,a4,a7
   1f1e0:	00000e93          	li	t4,0
   1f1e4:	00a808b3          	add	a7,a6,a0
   1f1e8:	40a00533          	neg	a0,a0
   1f1ec:	14bec263          	blt	t4,a1,1f330 <__subtf3+0x568>
   1f1f0:	00400513          	li	a0,4
   1f1f4:	40d506b3          	sub	a3,a0,a3
   1f1f8:	02c12503          	lw	a0,44(sp)
   1f1fc:	00259593          	slli	a1,a1,0x2
   1f200:	04058593          	addi	a1,a1,64
   1f204:	002585b3          	add	a1,a1,sp
   1f208:	00c55633          	srl	a2,a0,a2
   1f20c:	fec5a023          	sw	a2,-32(a1)
   1f210:	00400613          	li	a2,4
   1f214:	40d60633          	sub	a2,a2,a3
   1f218:	00261613          	slli	a2,a2,0x2
   1f21c:	00269693          	slli	a3,a3,0x2
   1f220:	00800593          	li	a1,8
   1f224:	00d806b3          	add	a3,a6,a3
   1f228:	00b66a63          	bltu	a2,a1,1f23c <__subtf3+0x474>
   1f22c:	0006a023          	sw	zero,0(a3) # 80000 <__BSS_END__+0x5d200>
   1f230:	0006a223          	sw	zero,4(a3)
   1f234:	ff860613          	addi	a2,a2,-8
   1f238:	00868693          	addi	a3,a3,8
   1f23c:	00400593          	li	a1,4
   1f240:	00b66463          	bltu	a2,a1,1f248 <__subtf3+0x480>
   1f244:	0006a023          	sw	zero,0(a3)
   1f248:	02012683          	lw	a3,32(sp)
   1f24c:	00e03733          	snez	a4,a4
   1f250:	00d76733          	or	a4,a4,a3
   1f254:	02412683          	lw	a3,36(sp)
   1f258:	02e12023          	sw	a4,32(sp)
   1f25c:	00e78733          	add	a4,a5,a4
   1f260:	01268933          	add	s2,a3,s2
   1f264:	02e12823          	sw	a4,48(sp)
   1f268:	00f73733          	sltu	a4,a4,a5
   1f26c:	02812783          	lw	a5,40(sp)
   1f270:	00e90733          	add	a4,s2,a4
   1f274:	02e12a23          	sw	a4,52(sp)
   1f278:	00d936b3          	sltu	a3,s2,a3
   1f27c:	01273733          	sltu	a4,a4,s2
   1f280:	013789b3          	add	s3,a5,s3
   1f284:	00e6e733          	or	a4,a3,a4
   1f288:	00e98733          	add	a4,s3,a4
   1f28c:	02e12c23          	sw	a4,56(sp)
   1f290:	00f9b7b3          	sltu	a5,s3,a5
   1f294:	01373733          	sltu	a4,a4,s3
   1f298:	00e7e7b3          	or	a5,a5,a4
   1f29c:	02c12703          	lw	a4,44(sp)
   1f2a0:	00eb0733          	add	a4,s6,a4
   1f2a4:	00e787b3          	add	a5,a5,a4
   1f2a8:	02f12e23          	sw	a5,60(sp)
   1f2ac:	00c79713          	slli	a4,a5,0xc
   1f2b0:	d2075ae3          	bgez	a4,1efe4 <__subtf3+0x21c>
   1f2b4:	03012683          	lw	a3,48(sp)
   1f2b8:	fff80737          	lui	a4,0xfff80
   1f2bc:	fff70713          	addi	a4,a4,-1 # fff7ffff <__BSS_END__+0xfff5d1ff>
   1f2c0:	00e7f7b3          	and	a5,a5,a4
   1f2c4:	02f12e23          	sw	a5,60(sp)
   1f2c8:	00148493          	addi	s1,s1,1
   1f2cc:	01f69693          	slli	a3,a3,0x1f
   1f2d0:	03010713          	addi	a4,sp,48
   1f2d4:	03c10513          	addi	a0,sp,60
   1f2d8:	00072603          	lw	a2,0(a4)
   1f2dc:	00472583          	lw	a1,4(a4)
   1f2e0:	00470713          	addi	a4,a4,4
   1f2e4:	00165613          	srli	a2,a2,0x1
   1f2e8:	01f59593          	slli	a1,a1,0x1f
   1f2ec:	00b66633          	or	a2,a2,a1
   1f2f0:	fec72e23          	sw	a2,-4(a4)
   1f2f4:	fee512e3          	bne	a0,a4,1f2d8 <__subtf3+0x510>
   1f2f8:	03012703          	lw	a4,48(sp)
   1f2fc:	0017d793          	srli	a5,a5,0x1
   1f300:	02f12e23          	sw	a5,60(sp)
   1f304:	00d037b3          	snez	a5,a3
   1f308:	00f767b3          	or	a5,a4,a5
   1f30c:	02f12823          	sw	a5,48(sp)
   1f310:	000087b7          	lui	a5,0x8
   1f314:	fff78793          	addi	a5,a5,-1 # 7fff <exit-0x80b5>
   1f318:	ccf496e3          	bne	s1,a5,1efe4 <__subtf3+0x21c>
   1f31c:	02012e23          	sw	zero,60(sp)
   1f320:	02012c23          	sw	zero,56(sp)
   1f324:	02012a23          	sw	zero,52(sp)
   1f328:	02012823          	sw	zero,48(sp)
   1f32c:	cb9ff06f          	j	1efe4 <__subtf3+0x21c>
   1f330:	0008a303          	lw	t1,0(a7)
   1f334:	0048af03          	lw	t5,4(a7)
   1f338:	00a88fb3          	add	t6,a7,a0
   1f33c:	00c35333          	srl	t1,t1,a2
   1f340:	01cf1f33          	sll	t5,t5,t3
   1f344:	01e36333          	or	t1,t1,t5
   1f348:	006fa023          	sw	t1,0(t6)
   1f34c:	001e8e93          	addi	t4,t4,1
   1f350:	00488893          	addi	a7,a7,4
   1f354:	e99ff06f          	j	1f1ec <__subtf3+0x424>
   1f358:	02412903          	lw	s2,36(sp)
   1f35c:	02812983          	lw	s3,40(sp)
   1f360:	02c12b03          	lw	s6,44(sp)
   1f364:	26088263          	beqz	a7,1f5c8 <__subtf3+0x800>
   1f368:	409b8833          	sub	a6,s7,s1
   1f36c:	0a049c63          	bnez	s1,1f424 <__subtf3+0x65c>
   1f370:	01412683          	lw	a3,20(sp)
   1f374:	01812703          	lw	a4,24(sp)
   1f378:	01c12883          	lw	a7,28(sp)
   1f37c:	00e6e5b3          	or	a1,a3,a4
   1f380:	0115e5b3          	or	a1,a1,a7
   1f384:	00f5e5b3          	or	a1,a1,a5
   1f388:	00059e63          	bnez	a1,1f3a4 <__subtf3+0x5dc>
   1f38c:	02812823          	sw	s0,48(sp)
   1f390:	03212a23          	sw	s2,52(sp)
   1f394:	03312c23          	sw	s3,56(sp)
   1f398:	03612e23          	sw	s6,60(sp)
   1f39c:	00080493          	mv	s1,a6
   1f3a0:	c45ff06f          	j	1efe4 <__subtf3+0x21c>
   1f3a4:	fff80593          	addi	a1,a6,-1
   1f3a8:	04059663          	bnez	a1,1f3f4 <__subtf3+0x62c>
   1f3ac:	00878433          	add	s0,a5,s0
   1f3b0:	01268933          	add	s2,a3,s2
   1f3b4:	02812823          	sw	s0,48(sp)
   1f3b8:	00f43433          	sltu	s0,s0,a5
   1f3bc:	00890433          	add	s0,s2,s0
   1f3c0:	02812a23          	sw	s0,52(sp)
   1f3c4:	00d936b3          	sltu	a3,s2,a3
   1f3c8:	01243433          	sltu	s0,s0,s2
   1f3cc:	013709b3          	add	s3,a4,s3
   1f3d0:	0086e6b3          	or	a3,a3,s0
   1f3d4:	00d986b3          	add	a3,s3,a3
   1f3d8:	02d12c23          	sw	a3,56(sp)
   1f3dc:	00e9b7b3          	sltu	a5,s3,a4
   1f3e0:	0136b6b3          	sltu	a3,a3,s3
   1f3e4:	00d7e7b3          	or	a5,a5,a3
   1f3e8:	016888b3          	add	a7,a7,s6
   1f3ec:	011787b3          	add	a5,a5,a7
   1f3f0:	bb9ff06f          	j	1efa8 <__subtf3+0x1e0>
   1f3f4:	000087b7          	lui	a5,0x8
   1f3f8:	fff78793          	addi	a5,a5,-1 # 7fff <exit-0x80b5>
   1f3fc:	62f818e3          	bne	a6,a5,2022c <__subtf3+0x1464>
   1f400:	02812823          	sw	s0,48(sp)
   1f404:	03212a23          	sw	s2,52(sp)
   1f408:	03312c23          	sw	s3,56(sp)
   1f40c:	03612e23          	sw	s6,60(sp)
   1f410:	000084b7          	lui	s1,0x8
   1f414:	fff48493          	addi	s1,s1,-1 # 7fff <exit-0x80b5>
   1f418:	bcdff06f          	j	1efe4 <__subtf3+0x21c>
   1f41c:	00078413          	mv	s0,a5
   1f420:	fe1ff06f          	j	1f400 <__subtf3+0x638>
   1f424:	000087b7          	lui	a5,0x8
   1f428:	fff78793          	addi	a5,a5,-1 # 7fff <exit-0x80b5>
   1f42c:	fcfb8ae3          	beq	s7,a5,1f400 <__subtf3+0x638>
   1f430:	01c12783          	lw	a5,28(sp)
   1f434:	00080737          	lui	a4,0x80
   1f438:	00e7e7b3          	or	a5,a5,a4
   1f43c:	00f12e23          	sw	a5,28(sp)
   1f440:	07400793          	li	a5,116
   1f444:	5f07c8e3          	blt	a5,a6,20234 <__subtf3+0x146c>
   1f448:	00080593          	mv	a1,a6
   1f44c:	02000713          	li	a4,32
   1f450:	02e5c733          	div	a4,a1,a4
   1f454:	00060693          	mv	a3,a2
   1f458:	00000493          	li	s1,0
   1f45c:	00000793          	li	a5,0
   1f460:	02e7ce63          	blt	a5,a4,1f49c <__subtf3+0x6d4>
   1f464:	00300793          	li	a5,3
   1f468:	01f5f893          	andi	a7,a1,31
   1f46c:	40e787b3          	sub	a5,a5,a4
   1f470:	00271813          	slli	a6,a4,0x2
   1f474:	02089e63          	bnez	a7,1f4b0 <__subtf3+0x6e8>
   1f478:	010606b3          	add	a3,a2,a6
   1f47c:	0006a683          	lw	a3,0(a3)
   1f480:	00188893          	addi	a7,a7,1
   1f484:	00460613          	addi	a2,a2,4
   1f488:	fed62e23          	sw	a3,-4(a2)
   1f48c:	ff17d6e3          	bge	a5,a7,1f478 <__subtf3+0x6b0>
   1f490:	00400793          	li	a5,4
   1f494:	40e78733          	sub	a4,a5,a4
   1f498:	0780006f          	j	1f510 <__subtf3+0x748>
   1f49c:	0006a803          	lw	a6,0(a3)
   1f4a0:	00178793          	addi	a5,a5,1
   1f4a4:	00468693          	addi	a3,a3,4
   1f4a8:	0104e4b3          	or	s1,s1,a6
   1f4ac:	fb5ff06f          	j	1f460 <__subtf3+0x698>
   1f4b0:	02000693          	li	a3,32
   1f4b4:	02d5e5b3          	rem	a1,a1,a3
   1f4b8:	40b685b3          	sub	a1,a3,a1
   1f4bc:	00070693          	mv	a3,a4
   1f4c0:	00075463          	bgez	a4,1f4c8 <__subtf3+0x700>
   1f4c4:	00000693          	li	a3,0
   1f4c8:	00269693          	slli	a3,a3,0x2
   1f4cc:	04068693          	addi	a3,a3,64
   1f4d0:	002686b3          	add	a3,a3,sp
   1f4d4:	fd06a683          	lw	a3,-48(a3)
   1f4d8:	00000313          	li	t1,0
   1f4dc:	00b696b3          	sll	a3,a3,a1
   1f4e0:	00d4e4b3          	or	s1,s1,a3
   1f4e4:	010506b3          	add	a3,a0,a6
   1f4e8:	41000833          	neg	a6,a6
   1f4ec:	0af34a63          	blt	t1,a5,1f5a0 <__subtf3+0x7d8>
   1f4f0:	00400693          	li	a3,4
   1f4f4:	40e68733          	sub	a4,a3,a4
   1f4f8:	01c12683          	lw	a3,28(sp)
   1f4fc:	00279793          	slli	a5,a5,0x2
   1f500:	04078793          	addi	a5,a5,64
   1f504:	002787b3          	add	a5,a5,sp
   1f508:	0116d6b3          	srl	a3,a3,a7
   1f50c:	fcd7a823          	sw	a3,-48(a5)
   1f510:	00572793          	slti	a5,a4,5
   1f514:	00000613          	li	a2,0
   1f518:	00078863          	beqz	a5,1f528 <__subtf3+0x760>
   1f51c:	00400613          	li	a2,4
   1f520:	40e60633          	sub	a2,a2,a4
   1f524:	00261613          	slli	a2,a2,0x2
   1f528:	00271713          	slli	a4,a4,0x2
   1f52c:	00e50533          	add	a0,a0,a4
   1f530:	00000593          	li	a1,0
   1f534:	fd0f10ef          	jal	10d04 <memset>
   1f538:	01012703          	lw	a4,16(sp)
   1f53c:	009037b3          	snez	a5,s1
   1f540:	00e7e7b3          	or	a5,a5,a4
   1f544:	01412683          	lw	a3,20(sp)
   1f548:	00f12823          	sw	a5,16(sp)
   1f54c:	00f407b3          	add	a5,s0,a5
   1f550:	01268933          	add	s2,a3,s2
   1f554:	02f12823          	sw	a5,48(sp)
   1f558:	0087b7b3          	sltu	a5,a5,s0
   1f55c:	00f90733          	add	a4,s2,a5
   1f560:	01812783          	lw	a5,24(sp)
   1f564:	02e12a23          	sw	a4,52(sp)
   1f568:	00d936b3          	sltu	a3,s2,a3
   1f56c:	01273733          	sltu	a4,a4,s2
   1f570:	013789b3          	add	s3,a5,s3
   1f574:	00e6e733          	or	a4,a3,a4
   1f578:	00e98733          	add	a4,s3,a4
   1f57c:	02e12c23          	sw	a4,56(sp)
   1f580:	00f9b7b3          	sltu	a5,s3,a5
   1f584:	01373733          	sltu	a4,a4,s3
   1f588:	00e7e7b3          	or	a5,a5,a4
   1f58c:	01c12703          	lw	a4,28(sp)
   1f590:	000b8493          	mv	s1,s7
   1f594:	00eb0733          	add	a4,s6,a4
   1f598:	00e787b3          	add	a5,a5,a4
   1f59c:	d0dff06f          	j	1f2a8 <__subtf3+0x4e0>
   1f5a0:	0006a603          	lw	a2,0(a3)
   1f5a4:	0046ae03          	lw	t3,4(a3)
   1f5a8:	01068eb3          	add	t4,a3,a6
   1f5ac:	01165633          	srl	a2,a2,a7
   1f5b0:	00be1e33          	sll	t3,t3,a1
   1f5b4:	01c66633          	or	a2,a2,t3
   1f5b8:	00cea023          	sw	a2,0(t4)
   1f5bc:	00130313          	addi	t1,t1,1
   1f5c0:	00468693          	addi	a3,a3,4
   1f5c4:	f29ff06f          	j	1f4ec <__subtf3+0x724>
   1f5c8:	00148813          	addi	a6,s1,1
   1f5cc:	01181893          	slli	a7,a6,0x11
   1f5d0:	0128d893          	srli	a7,a7,0x12
   1f5d4:	01412683          	lw	a3,20(sp)
   1f5d8:	01812703          	lw	a4,24(sp)
   1f5dc:	01c12603          	lw	a2,28(sp)
   1f5e0:	03010593          	addi	a1,sp,48
   1f5e4:	03c10513          	addi	a0,sp,60
   1f5e8:	10089e63          	bnez	a7,1f704 <__subtf3+0x93c>
   1f5ec:	00e6e833          	or	a6,a3,a4
   1f5f0:	00c86833          	or	a6,a6,a2
   1f5f4:	00f86833          	or	a6,a6,a5
   1f5f8:	0a049863          	bnez	s1,1f6a8 <__subtf3+0x8e0>
   1f5fc:	00081e63          	bnez	a6,1f618 <__subtf3+0x850>
   1f600:	02812823          	sw	s0,48(sp)
   1f604:	03212a23          	sw	s2,52(sp)
   1f608:	03312c23          	sw	s3,56(sp)
   1f60c:	03612e23          	sw	s6,60(sp)
   1f610:	00000493          	li	s1,0
   1f614:	9d1ff06f          	j	1efe4 <__subtf3+0x21c>
   1f618:	013965b3          	or	a1,s2,s3
   1f61c:	0165e5b3          	or	a1,a1,s6
   1f620:	0085e5b3          	or	a1,a1,s0
   1f624:	00059c63          	bnez	a1,1f63c <__subtf3+0x874>
   1f628:	02f12823          	sw	a5,48(sp)
   1f62c:	02d12a23          	sw	a3,52(sp)
   1f630:	02e12c23          	sw	a4,56(sp)
   1f634:	02c12e23          	sw	a2,60(sp)
   1f638:	9adff06f          	j	1efe4 <__subtf3+0x21c>
   1f63c:	00878433          	add	s0,a5,s0
   1f640:	01268933          	add	s2,a3,s2
   1f644:	02812823          	sw	s0,48(sp)
   1f648:	00f43433          	sltu	s0,s0,a5
   1f64c:	00890433          	add	s0,s2,s0
   1f650:	02812a23          	sw	s0,52(sp)
   1f654:	00d936b3          	sltu	a3,s2,a3
   1f658:	01243433          	sltu	s0,s0,s2
   1f65c:	013709b3          	add	s3,a4,s3
   1f660:	0086e6b3          	or	a3,a3,s0
   1f664:	00d986b3          	add	a3,s3,a3
   1f668:	02d12c23          	sw	a3,56(sp)
   1f66c:	00e9b733          	sltu	a4,s3,a4
   1f670:	0136b6b3          	sltu	a3,a3,s3
   1f674:	00d76733          	or	a4,a4,a3
   1f678:	01660633          	add	a2,a2,s6
   1f67c:	00c70733          	add	a4,a4,a2
   1f680:	00c71793          	slli	a5,a4,0xc
   1f684:	0007c663          	bltz	a5,1f690 <__subtf3+0x8c8>
   1f688:	02e12e23          	sw	a4,60(sp)
   1f68c:	959ff06f          	j	1efe4 <__subtf3+0x21c>
   1f690:	fff807b7          	lui	a5,0xfff80
   1f694:	fff78793          	addi	a5,a5,-1 # fff7ffff <__BSS_END__+0xfff5d1ff>
   1f698:	00f77733          	and	a4,a4,a5
   1f69c:	02e12e23          	sw	a4,60(sp)
   1f6a0:	00100493          	li	s1,1
   1f6a4:	941ff06f          	j	1efe4 <__subtf3+0x21c>
   1f6a8:	1a080463          	beqz	a6,1f850 <__subtf3+0xa88>
   1f6ac:	01396933          	or	s2,s2,s3
   1f6b0:	01696933          	or	s2,s2,s6
   1f6b4:	00896933          	or	s2,s2,s0
   1f6b8:	1a090063          	beqz	s2,1f858 <__subtf3+0xa90>
   1f6bc:	000087b7          	lui	a5,0x8
   1f6c0:	02f12e23          	sw	a5,60(sp)
   1f6c4:	02012c23          	sw	zero,56(sp)
   1f6c8:	02012a23          	sw	zero,52(sp)
   1f6cc:	02012823          	sw	zero,48(sp)
   1f6d0:	00050793          	mv	a5,a0
   1f6d4:	0007a703          	lw	a4,0(a5) # 8000 <exit-0x80b4>
   1f6d8:	ffc7a683          	lw	a3,-4(a5)
   1f6dc:	ffc78793          	addi	a5,a5,-4
   1f6e0:	00371713          	slli	a4,a4,0x3
   1f6e4:	01d6d693          	srli	a3,a3,0x1d
   1f6e8:	00d76733          	or	a4,a4,a3
   1f6ec:	00e7a223          	sw	a4,4(a5)
   1f6f0:	fef592e3          	bne	a1,a5,1f6d4 <__subtf3+0x90c>
   1f6f4:	000084b7          	lui	s1,0x8
   1f6f8:	fff48493          	addi	s1,s1,-1 # 7fff <exit-0x80b5>
   1f6fc:	00000a93          	li	s5,0
   1f700:	8e5ff06f          	j	1efe4 <__subtf3+0x21c>
   1f704:	00878433          	add	s0,a5,s0
   1f708:	01268933          	add	s2,a3,s2
   1f70c:	02812823          	sw	s0,48(sp)
   1f710:	00f43433          	sltu	s0,s0,a5
   1f714:	00890433          	add	s0,s2,s0
   1f718:	02812a23          	sw	s0,52(sp)
   1f71c:	00d936b3          	sltu	a3,s2,a3
   1f720:	01243433          	sltu	s0,s0,s2
   1f724:	013709b3          	add	s3,a4,s3
   1f728:	0086e6b3          	or	a3,a3,s0
   1f72c:	00d986b3          	add	a3,s3,a3
   1f730:	02d12c23          	sw	a3,56(sp)
   1f734:	00e9b733          	sltu	a4,s3,a4
   1f738:	0136b6b3          	sltu	a3,a3,s3
   1f73c:	00d76733          	or	a4,a4,a3
   1f740:	01660633          	add	a2,a2,s6
   1f744:	00c70733          	add	a4,a4,a2
   1f748:	02e12e23          	sw	a4,60(sp)
   1f74c:	00058793          	mv	a5,a1
   1f750:	0007a683          	lw	a3,0(a5)
   1f754:	0047a603          	lw	a2,4(a5)
   1f758:	00478793          	addi	a5,a5,4
   1f75c:	0016d693          	srli	a3,a3,0x1
   1f760:	01f61613          	slli	a2,a2,0x1f
   1f764:	00c6e6b3          	or	a3,a3,a2
   1f768:	fed7ae23          	sw	a3,-4(a5)
   1f76c:	fef512e3          	bne	a0,a5,1f750 <__subtf3+0x988>
   1f770:	000087b7          	lui	a5,0x8
   1f774:	fff78793          	addi	a5,a5,-1 # 7fff <exit-0x80b5>
   1f778:	00f80863          	beq	a6,a5,1f788 <__subtf3+0x9c0>
   1f77c:	00175713          	srli	a4,a4,0x1
   1f780:	02e12e23          	sw	a4,60(sp)
   1f784:	c19ff06f          	j	1f39c <__subtf3+0x5d4>
   1f788:	02012e23          	sw	zero,60(sp)
   1f78c:	02012c23          	sw	zero,56(sp)
   1f790:	02012a23          	sw	zero,52(sp)
   1f794:	02012823          	sw	zero,48(sp)
   1f798:	c05ff06f          	j	1f39c <__subtf3+0x5d4>
   1f79c:	29105c63          	blez	a7,1fa34 <__subtf3+0xc6c>
   1f7a0:	01412903          	lw	s2,20(sp)
   1f7a4:	01812983          	lw	s3,24(sp)
   1f7a8:	01c12b03          	lw	s6,28(sp)
   1f7ac:	0a0b9e63          	bnez	s7,1f868 <__subtf3+0xaa0>
   1f7b0:	02412e03          	lw	t3,36(sp)
   1f7b4:	02812503          	lw	a0,40(sp)
   1f7b8:	02c12683          	lw	a3,44(sp)
   1f7bc:	00ae6733          	or	a4,t3,a0
   1f7c0:	00d76733          	or	a4,a4,a3
   1f7c4:	00876733          	or	a4,a4,s0
   1f7c8:	f6070e63          	beqz	a4,1ef44 <__subtf3+0x17c>
   1f7cc:	fff88e93          	addi	t4,a7,-1
   1f7d0:	040e9c63          	bnez	t4,1f828 <__subtf3+0xa60>
   1f7d4:	40878733          	sub	a4,a5,s0
   1f7d8:	41c90633          	sub	a2,s2,t3
   1f7dc:	00e7b5b3          	sltu	a1,a5,a4
   1f7e0:	00c93833          	sltu	a6,s2,a2
   1f7e4:	40b60633          	sub	a2,a2,a1
   1f7e8:	00000593          	li	a1,0
   1f7ec:	00e7f663          	bgeu	a5,a4,1f7f8 <__subtf3+0xa30>
   1f7f0:	412e0e33          	sub	t3,t3,s2
   1f7f4:	001e3593          	seqz	a1,t3
   1f7f8:	0105e7b3          	or	a5,a1,a6
   1f7fc:	40a985b3          	sub	a1,s3,a0
   1f800:	00b9b833          	sltu	a6,s3,a1
   1f804:	40f585b3          	sub	a1,a1,a5
   1f808:	00078663          	beqz	a5,1f814 <__subtf3+0xa4c>
   1f80c:	41350533          	sub	a0,a0,s3
   1f810:	00153e93          	seqz	t4,a0
   1f814:	40db07b3          	sub	a5,s6,a3
   1f818:	010ee6b3          	or	a3,t4,a6
   1f81c:	40d787b3          	sub	a5,a5,a3
   1f820:	00100493          	li	s1,1
   1f824:	1bc0006f          	j	1f9e0 <__subtf3+0xc18>
   1f828:	00008737          	lui	a4,0x8
   1f82c:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1f830:	f8e88863          	beq	a7,a4,1efc0 <__subtf3+0x1f8>
   1f834:	07400713          	li	a4,116
   1f838:	05d75c63          	bge	a4,t4,1f890 <__subtf3+0xac8>
   1f83c:	02012623          	sw	zero,44(sp)
   1f840:	02012423          	sw	zero,40(sp)
   1f844:	02012223          	sw	zero,36(sp)
   1f848:	00100713          	li	a4,1
   1f84c:	1340006f          	j	1f980 <__subtf3+0xbb8>
   1f850:	00040793          	mv	a5,s0
   1f854:	f6cff06f          	j	1efc0 <__subtf3+0x1f8>
   1f858:	00068913          	mv	s2,a3
   1f85c:	00070993          	mv	s3,a4
   1f860:	00060b13          	mv	s6,a2
   1f864:	f5cff06f          	j	1efc0 <__subtf3+0x1f8>
   1f868:	00008737          	lui	a4,0x8
   1f86c:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   1f870:	bae486e3          	beq	s1,a4,1f41c <__subtf3+0x654>
   1f874:	02c12703          	lw	a4,44(sp)
   1f878:	000806b7          	lui	a3,0x80
   1f87c:	00d76733          	or	a4,a4,a3
   1f880:	02e12623          	sw	a4,44(sp)
   1f884:	07400713          	li	a4,116
   1f888:	fb174ae3          	blt	a4,a7,1f83c <__subtf3+0xa74>
   1f88c:	00088e93          	mv	t4,a7
   1f890:	405ed693          	srai	a3,t4,0x5
   1f894:	00030593          	mv	a1,t1
   1f898:	01fefe93          	andi	t4,t4,31
   1f89c:	00000713          	li	a4,0
   1f8a0:	00000613          	li	a2,0
   1f8a4:	02d61c63          	bne	a2,a3,1f8dc <__subtf3+0xb14>
   1f8a8:	00300613          	li	a2,3
   1f8ac:	40d60633          	sub	a2,a2,a3
   1f8b0:	00269593          	slli	a1,a3,0x2
   1f8b4:	020e9e63          	bnez	t4,1f8f0 <__subtf3+0xb28>
   1f8b8:	00b30533          	add	a0,t1,a1
   1f8bc:	00052503          	lw	a0,0(a0)
   1f8c0:	001e8e93          	addi	t4,t4,1
   1f8c4:	00430313          	addi	t1,t1,4
   1f8c8:	fea32e23          	sw	a0,-4(t1)
   1f8cc:	ffd656e3          	bge	a2,t4,1f8b8 <__subtf3+0xaf0>
   1f8d0:	00400613          	li	a2,4
   1f8d4:	40d606b3          	sub	a3,a2,a3
   1f8d8:	0640006f          	j	1f93c <__subtf3+0xb74>
   1f8dc:	0005a503          	lw	a0,0(a1)
   1f8e0:	00160613          	addi	a2,a2,1
   1f8e4:	00458593          	addi	a1,a1,4
   1f8e8:	00a76733          	or	a4,a4,a0
   1f8ec:	fb9ff06f          	j	1f8a4 <__subtf3+0xadc>
   1f8f0:	04058513          	addi	a0,a1,64
   1f8f4:	00250533          	add	a0,a0,sp
   1f8f8:	fe052503          	lw	a0,-32(a0)
   1f8fc:	02000313          	li	t1,32
   1f900:	41d30333          	sub	t1,t1,t4
   1f904:	00651533          	sll	a0,a0,t1
   1f908:	00a76733          	or	a4,a4,a0
   1f90c:	00000e13          	li	t3,0
   1f910:	00b80533          	add	a0,a6,a1
   1f914:	40b005b3          	neg	a1,a1
   1f918:	0ece4a63          	blt	t3,a2,1fa0c <__subtf3+0xc44>
   1f91c:	00400593          	li	a1,4
   1f920:	40d586b3          	sub	a3,a1,a3
   1f924:	02c12583          	lw	a1,44(sp)
   1f928:	00261613          	slli	a2,a2,0x2
   1f92c:	04060613          	addi	a2,a2,64
   1f930:	00260633          	add	a2,a2,sp
   1f934:	01d5d5b3          	srl	a1,a1,t4
   1f938:	feb62023          	sw	a1,-32(a2)
   1f93c:	00400613          	li	a2,4
   1f940:	40d60633          	sub	a2,a2,a3
   1f944:	00261613          	slli	a2,a2,0x2
   1f948:	00269693          	slli	a3,a3,0x2
   1f94c:	00800593          	li	a1,8
   1f950:	00d806b3          	add	a3,a6,a3
   1f954:	00b66a63          	bltu	a2,a1,1f968 <__subtf3+0xba0>
   1f958:	0006a023          	sw	zero,0(a3) # 80000 <__BSS_END__+0x5d200>
   1f95c:	0006a223          	sw	zero,4(a3)
   1f960:	ff860613          	addi	a2,a2,-8
   1f964:	00868693          	addi	a3,a3,8
   1f968:	00400593          	li	a1,4
   1f96c:	00b66463          	bltu	a2,a1,1f974 <__subtf3+0xbac>
   1f970:	0006a023          	sw	zero,0(a3)
   1f974:	02012683          	lw	a3,32(sp)
   1f978:	00e03733          	snez	a4,a4
   1f97c:	00d76733          	or	a4,a4,a3
   1f980:	02412583          	lw	a1,36(sp)
   1f984:	02e12023          	sw	a4,32(sp)
   1f988:	40e78733          	sub	a4,a5,a4
   1f98c:	40b90633          	sub	a2,s2,a1
   1f990:	00e7b6b3          	sltu	a3,a5,a4
   1f994:	00c93533          	sltu	a0,s2,a2
   1f998:	40d60633          	sub	a2,a2,a3
   1f99c:	00000693          	li	a3,0
   1f9a0:	00e7f663          	bgeu	a5,a4,1f9ac <__subtf3+0xbe4>
   1f9a4:	412585b3          	sub	a1,a1,s2
   1f9a8:	0015b693          	seqz	a3,a1
   1f9ac:	00a6e7b3          	or	a5,a3,a0
   1f9b0:	02812503          	lw	a0,40(sp)
   1f9b4:	00000693          	li	a3,0
   1f9b8:	40a985b3          	sub	a1,s3,a0
   1f9bc:	00b9b833          	sltu	a6,s3,a1
   1f9c0:	40f585b3          	sub	a1,a1,a5
   1f9c4:	00078663          	beqz	a5,1f9d0 <__subtf3+0xc08>
   1f9c8:	41350533          	sub	a0,a0,s3
   1f9cc:	00153693          	seqz	a3,a0
   1f9d0:	02c12783          	lw	a5,44(sp)
   1f9d4:	0106e6b3          	or	a3,a3,a6
   1f9d8:	40fb07b3          	sub	a5,s6,a5
   1f9dc:	40d787b3          	sub	a5,a5,a3
   1f9e0:	02e12823          	sw	a4,48(sp)
   1f9e4:	02f12e23          	sw	a5,60(sp)
   1f9e8:	02b12c23          	sw	a1,56(sp)
   1f9ec:	02c12a23          	sw	a2,52(sp)
   1f9f0:	00c79713          	slli	a4,a5,0xc
   1f9f4:	de075863          	bgez	a4,1efe4 <__subtf3+0x21c>
   1f9f8:	00080737          	lui	a4,0x80
   1f9fc:	fff70713          	addi	a4,a4,-1 # 7ffff <__BSS_END__+0x5d1ff>
   1fa00:	00e7f7b3          	and	a5,a5,a4
   1fa04:	02f12e23          	sw	a5,60(sp)
   1fa08:	5700006f          	j	1ff78 <__subtf3+0x11b0>
   1fa0c:	00052883          	lw	a7,0(a0)
   1fa10:	00452f03          	lw	t5,4(a0)
   1fa14:	00b50fb3          	add	t6,a0,a1
   1fa18:	01d8d8b3          	srl	a7,a7,t4
   1fa1c:	006f1f33          	sll	t5,t5,t1
   1fa20:	01e8e8b3          	or	a7,a7,t5
   1fa24:	011fa023          	sw	a7,0(t6)
   1fa28:	001e0e13          	addi	t3,t3,1
   1fa2c:	00450513          	addi	a0,a0,4
   1fa30:	ee9ff06f          	j	1f918 <__subtf3+0xb50>
   1fa34:	02412c03          	lw	s8,36(sp)
   1fa38:	02812b03          	lw	s6,40(sp)
   1fa3c:	02c12983          	lw	s3,44(sp)
   1fa40:	28088463          	beqz	a7,1fcc8 <__subtf3+0xf00>
   1fa44:	409b8333          	sub	t1,s7,s1
   1fa48:	0a049e63          	bnez	s1,1fb04 <__subtf3+0xd3c>
   1fa4c:	01412583          	lw	a1,20(sp)
   1fa50:	01812803          	lw	a6,24(sp)
   1fa54:	01c12683          	lw	a3,28(sp)
   1fa58:	0105e8b3          	or	a7,a1,a6
   1fa5c:	00d8e8b3          	or	a7,a7,a3
   1fa60:	00f8e8b3          	or	a7,a7,a5
   1fa64:	02089063          	bnez	a7,1fa84 <__subtf3+0xcbc>
   1fa68:	02812823          	sw	s0,48(sp)
   1fa6c:	03812a23          	sw	s8,52(sp)
   1fa70:	03612c23          	sw	s6,56(sp)
   1fa74:	03312e23          	sw	s3,60(sp)
   1fa78:	00030493          	mv	s1,t1
   1fa7c:	00090a93          	mv	s5,s2
   1fa80:	d64ff06f          	j	1efe4 <__subtf3+0x21c>
   1fa84:	fff30893          	addi	a7,t1,-1
   1fa88:	04089c63          	bnez	a7,1fae0 <__subtf3+0xd18>
   1fa8c:	40f40733          	sub	a4,s0,a5
   1fa90:	40bc0633          	sub	a2,s8,a1
   1fa94:	00e437b3          	sltu	a5,s0,a4
   1fa98:	00cc3533          	sltu	a0,s8,a2
   1fa9c:	40f60633          	sub	a2,a2,a5
   1faa0:	00000793          	li	a5,0
   1faa4:	00e47663          	bgeu	s0,a4,1fab0 <__subtf3+0xce8>
   1faa8:	418585b3          	sub	a1,a1,s8
   1faac:	0015b793          	seqz	a5,a1
   1fab0:	00a7e7b3          	or	a5,a5,a0
   1fab4:	410b05b3          	sub	a1,s6,a6
   1fab8:	00bb3533          	sltu	a0,s6,a1
   1fabc:	40f585b3          	sub	a1,a1,a5
   1fac0:	00078663          	beqz	a5,1facc <__subtf3+0xd04>
   1fac4:	41680833          	sub	a6,a6,s6
   1fac8:	00183893          	seqz	a7,a6
   1facc:	40d987b3          	sub	a5,s3,a3
   1fad0:	00a8e6b3          	or	a3,a7,a0
   1fad4:	40d787b3          	sub	a5,a5,a3
   1fad8:	00090a93          	mv	s5,s2
   1fadc:	d45ff06f          	j	1f820 <__subtf3+0xa58>
   1fae0:	000087b7          	lui	a5,0x8
   1fae4:	fff78793          	addi	a5,a5,-1 # 7fff <exit-0x80b5>
   1fae8:	76f31063          	bne	t1,a5,20248 <__subtf3+0x1480>
   1faec:	02812823          	sw	s0,48(sp)
   1faf0:	03812a23          	sw	s8,52(sp)
   1faf4:	03612c23          	sw	s6,56(sp)
   1faf8:	03312e23          	sw	s3,60(sp)
   1fafc:	00090a93          	mv	s5,s2
   1fb00:	911ff06f          	j	1f410 <__subtf3+0x648>
   1fb04:	000087b7          	lui	a5,0x8
   1fb08:	fff78793          	addi	a5,a5,-1 # 7fff <exit-0x80b5>
   1fb0c:	fefb80e3          	beq	s7,a5,1faec <__subtf3+0xd24>
   1fb10:	01c12783          	lw	a5,28(sp)
   1fb14:	00080737          	lui	a4,0x80
   1fb18:	00e7e7b3          	or	a5,a5,a4
   1fb1c:	00f12e23          	sw	a5,28(sp)
   1fb20:	07400793          	li	a5,116
   1fb24:	1867c863          	blt	a5,t1,1fcb4 <__subtf3+0xeec>
   1fb28:	02000793          	li	a5,32
   1fb2c:	02f347b3          	div	a5,t1,a5
   1fb30:	00060693          	mv	a3,a2
   1fb34:	00000493          	li	s1,0
   1fb38:	00000713          	li	a4,0
   1fb3c:	02f74e63          	blt	a4,a5,1fb78 <__subtf3+0xdb0>
   1fb40:	00300713          	li	a4,3
   1fb44:	01f37893          	andi	a7,t1,31
   1fb48:	40f70e33          	sub	t3,a4,a5
   1fb4c:	00279593          	slli	a1,a5,0x2
   1fb50:	02089e63          	bnez	a7,1fb8c <__subtf3+0xdc4>
   1fb54:	00b60733          	add	a4,a2,a1
   1fb58:	00072703          	lw	a4,0(a4) # 80000 <__BSS_END__+0x5d200>
   1fb5c:	00188893          	addi	a7,a7,1
   1fb60:	00460613          	addi	a2,a2,4
   1fb64:	fee62e23          	sw	a4,-4(a2)
   1fb68:	ff1e56e3          	bge	t3,a7,1fb54 <__subtf3+0xd8c>
   1fb6c:	00400713          	li	a4,4
   1fb70:	40f707b3          	sub	a5,a4,a5
   1fb74:	0780006f          	j	1fbec <__subtf3+0xe24>
   1fb78:	0006a583          	lw	a1,0(a3)
   1fb7c:	00170713          	addi	a4,a4,1
   1fb80:	00468693          	addi	a3,a3,4
   1fb84:	00b4e4b3          	or	s1,s1,a1
   1fb88:	fb5ff06f          	j	1fb3c <__subtf3+0xd74>
   1fb8c:	02000613          	li	a2,32
   1fb90:	02c36733          	rem	a4,t1,a2
   1fb94:	00078693          	mv	a3,a5
   1fb98:	40e60633          	sub	a2,a2,a4
   1fb9c:	0007d463          	bgez	a5,1fba4 <__subtf3+0xddc>
   1fba0:	00000693          	li	a3,0
   1fba4:	00269693          	slli	a3,a3,0x2
   1fba8:	04068713          	addi	a4,a3,64
   1fbac:	002706b3          	add	a3,a4,sp
   1fbb0:	fd06a703          	lw	a4,-48(a3)
   1fbb4:	00b506b3          	add	a3,a0,a1
   1fbb8:	40b005b3          	neg	a1,a1
   1fbbc:	00c71733          	sll	a4,a4,a2
   1fbc0:	00e4e4b3          	or	s1,s1,a4
   1fbc4:	00000713          	li	a4,0
   1fbc8:	0dc74263          	blt	a4,t3,1fc8c <__subtf3+0xec4>
   1fbcc:	01c12683          	lw	a3,28(sp)
   1fbd0:	00400713          	li	a4,4
   1fbd4:	40f707b3          	sub	a5,a4,a5
   1fbd8:	002e1713          	slli	a4,t3,0x2
   1fbdc:	04070713          	addi	a4,a4,64
   1fbe0:	00270733          	add	a4,a4,sp
   1fbe4:	0116d6b3          	srl	a3,a3,a7
   1fbe8:	fcd72823          	sw	a3,-48(a4)
   1fbec:	0057a713          	slti	a4,a5,5
   1fbf0:	00000613          	li	a2,0
   1fbf4:	00070863          	beqz	a4,1fc04 <__subtf3+0xe3c>
   1fbf8:	00400613          	li	a2,4
   1fbfc:	40f60633          	sub	a2,a2,a5
   1fc00:	00261613          	slli	a2,a2,0x2
   1fc04:	00279793          	slli	a5,a5,0x2
   1fc08:	00f50533          	add	a0,a0,a5
   1fc0c:	00000593          	li	a1,0
   1fc10:	8f4f10ef          	jal	10d04 <memset>
   1fc14:	01012783          	lw	a5,16(sp)
   1fc18:	00903733          	snez	a4,s1
   1fc1c:	00f76733          	or	a4,a4,a5
   1fc20:	01412683          	lw	a3,20(sp)
   1fc24:	00e12823          	sw	a4,16(sp)
   1fc28:	40e40733          	sub	a4,s0,a4
   1fc2c:	40dc0633          	sub	a2,s8,a3
   1fc30:	00e437b3          	sltu	a5,s0,a4
   1fc34:	00cc35b3          	sltu	a1,s8,a2
   1fc38:	40f60633          	sub	a2,a2,a5
   1fc3c:	00000793          	li	a5,0
   1fc40:	00e47663          	bgeu	s0,a4,1fc4c <__subtf3+0xe84>
   1fc44:	418686b3          	sub	a3,a3,s8
   1fc48:	0016b793          	seqz	a5,a3
   1fc4c:	01812503          	lw	a0,24(sp)
   1fc50:	00b7e7b3          	or	a5,a5,a1
   1fc54:	00000693          	li	a3,0
   1fc58:	40ab05b3          	sub	a1,s6,a0
   1fc5c:	00bb3833          	sltu	a6,s6,a1
   1fc60:	40f585b3          	sub	a1,a1,a5
   1fc64:	00078663          	beqz	a5,1fc70 <__subtf3+0xea8>
   1fc68:	41650533          	sub	a0,a0,s6
   1fc6c:	00153693          	seqz	a3,a0
   1fc70:	01c12783          	lw	a5,28(sp)
   1fc74:	0106e6b3          	or	a3,a3,a6
   1fc78:	000b8493          	mv	s1,s7
   1fc7c:	40f987b3          	sub	a5,s3,a5
   1fc80:	40d787b3          	sub	a5,a5,a3
   1fc84:	00090a93          	mv	s5,s2
   1fc88:	d59ff06f          	j	1f9e0 <__subtf3+0xc18>
   1fc8c:	0006a803          	lw	a6,0(a3)
   1fc90:	0046a303          	lw	t1,4(a3)
   1fc94:	00b68eb3          	add	t4,a3,a1
   1fc98:	01185833          	srl	a6,a6,a7
   1fc9c:	00c31333          	sll	t1,t1,a2
   1fca0:	00686833          	or	a6,a6,t1
   1fca4:	010ea023          	sw	a6,0(t4)
   1fca8:	00170713          	addi	a4,a4,1
   1fcac:	00468693          	addi	a3,a3,4
   1fcb0:	f19ff06f          	j	1fbc8 <__subtf3+0xe00>
   1fcb4:	00012e23          	sw	zero,28(sp)
   1fcb8:	00012c23          	sw	zero,24(sp)
   1fcbc:	00012a23          	sw	zero,20(sp)
   1fcc0:	00100713          	li	a4,1
   1fcc4:	f5dff06f          	j	1fc20 <__subtf3+0xe58>
   1fcc8:	00148593          	addi	a1,s1,1
   1fccc:	01159513          	slli	a0,a1,0x11
   1fcd0:	01255513          	srli	a0,a0,0x12
   1fcd4:	01412683          	lw	a3,20(sp)
   1fcd8:	01812603          	lw	a2,24(sp)
   1fcdc:	01c12703          	lw	a4,28(sp)
   1fce0:	00008837          	lui	a6,0x8
   1fce4:	1c051e63          	bnez	a0,1fec0 <__subtf3+0x10f8>
   1fce8:	016c6533          	or	a0,s8,s6
   1fcec:	00c6e5b3          	or	a1,a3,a2
   1fcf0:	01356533          	or	a0,a0,s3
   1fcf4:	00e5e5b3          	or	a1,a1,a4
   1fcf8:	00856533          	or	a0,a0,s0
   1fcfc:	00f5e5b3          	or	a1,a1,a5
   1fd00:	10049863          	bnez	s1,1fe10 <__subtf3+0x1048>
   1fd04:	02059263          	bnez	a1,1fd28 <__subtf3+0xf60>
   1fd08:	02812823          	sw	s0,48(sp)
   1fd0c:	03812a23          	sw	s8,52(sp)
   1fd10:	03612c23          	sw	s6,56(sp)
   1fd14:	03312e23          	sw	s3,60(sp)
   1fd18:	00090a93          	mv	s5,s2
   1fd1c:	ac051463          	bnez	a0,1efe4 <__subtf3+0x21c>
   1fd20:	00000493          	li	s1,0
   1fd24:	9d9ff06f          	j	1f6fc <__subtf3+0x934>
   1fd28:	00051c63          	bnez	a0,1fd40 <__subtf3+0xf78>
   1fd2c:	02f12823          	sw	a5,48(sp)
   1fd30:	02d12a23          	sw	a3,52(sp)
   1fd34:	02c12c23          	sw	a2,56(sp)
   1fd38:	02e12e23          	sw	a4,60(sp)
   1fd3c:	8d5ff06f          	j	1f610 <__subtf3+0x848>
   1fd40:	40878533          	sub	a0,a5,s0
   1fd44:	41868e33          	sub	t3,a3,s8
   1fd48:	00a7b833          	sltu	a6,a5,a0
   1fd4c:	01c6b8b3          	sltu	a7,a3,t3
   1fd50:	410e0833          	sub	a6,t3,a6
   1fd54:	00000593          	li	a1,0
   1fd58:	00a7f463          	bgeu	a5,a0,1fd60 <__subtf3+0xf98>
   1fd5c:	001e3593          	seqz	a1,t3
   1fd60:	0115e5b3          	or	a1,a1,a7
   1fd64:	416608b3          	sub	a7,a2,s6
   1fd68:	01163f33          	sltu	t5,a2,a7
   1fd6c:	40b88eb3          	sub	t4,a7,a1
   1fd70:	00000313          	li	t1,0
   1fd74:	00058463          	beqz	a1,1fd7c <__subtf3+0xfb4>
   1fd78:	0018b313          	seqz	t1,a7
   1fd7c:	01e36333          	or	t1,t1,t5
   1fd80:	413705b3          	sub	a1,a4,s3
   1fd84:	406585b3          	sub	a1,a1,t1
   1fd88:	02b12e23          	sw	a1,60(sp)
   1fd8c:	03d12c23          	sw	t4,56(sp)
   1fd90:	03012a23          	sw	a6,52(sp)
   1fd94:	02a12823          	sw	a0,48(sp)
   1fd98:	00c59313          	slli	t1,a1,0xc
   1fd9c:	06035063          	bgez	t1,1fdfc <__subtf3+0x1034>
   1fda0:	40f407b3          	sub	a5,s0,a5
   1fda4:	40dc06b3          	sub	a3,s8,a3
   1fda8:	00f435b3          	sltu	a1,s0,a5
   1fdac:	00dc3c33          	sltu	s8,s8,a3
   1fdb0:	40b686b3          	sub	a3,a3,a1
   1fdb4:	00000593          	li	a1,0
   1fdb8:	00f47463          	bgeu	s0,a5,1fdc0 <__subtf3+0xff8>
   1fdbc:	001e3593          	seqz	a1,t3
   1fdc0:	40cb0633          	sub	a2,s6,a2
   1fdc4:	0185ec33          	or	s8,a1,s8
   1fdc8:	00cb3b33          	sltu	s6,s6,a2
   1fdcc:	00000513          	li	a0,0
   1fdd0:	41860633          	sub	a2,a2,s8
   1fdd4:	000c0463          	beqz	s8,1fddc <__subtf3+0x1014>
   1fdd8:	0018b513          	seqz	a0,a7
   1fddc:	40e98733          	sub	a4,s3,a4
   1fde0:	01656533          	or	a0,a0,s6
   1fde4:	40a70733          	sub	a4,a4,a0
   1fde8:	02e12e23          	sw	a4,60(sp)
   1fdec:	02c12c23          	sw	a2,56(sp)
   1fdf0:	02d12a23          	sw	a3,52(sp)
   1fdf4:	02f12823          	sw	a5,48(sp)
   1fdf8:	c85ff06f          	j	1fa7c <__subtf3+0xcb4>
   1fdfc:	01056533          	or	a0,a0,a6
   1fe00:	01d56533          	or	a0,a0,t4
   1fe04:	00b56533          	or	a0,a0,a1
   1fe08:	f0050ce3          	beqz	a0,1fd20 <__subtf3+0xf58>
   1fe0c:	805ff06f          	j	1f610 <__subtf3+0x848>
   1fe10:	03010893          	addi	a7,sp,48
   1fe14:	04059e63          	bnez	a1,1fe70 <__subtf3+0x10a8>
   1fe18:	02051e63          	bnez	a0,1fe54 <__subtf3+0x108c>
   1fe1c:	03012e23          	sw	a6,60(sp)
   1fe20:	02012c23          	sw	zero,56(sp)
   1fe24:	02012a23          	sw	zero,52(sp)
   1fe28:	02012823          	sw	zero,48(sp)
   1fe2c:	03c10793          	addi	a5,sp,60
   1fe30:	0007a703          	lw	a4,0(a5)
   1fe34:	ffc7a683          	lw	a3,-4(a5)
   1fe38:	ffc78793          	addi	a5,a5,-4
   1fe3c:	00371713          	slli	a4,a4,0x3
   1fe40:	01d6d693          	srli	a3,a3,0x1d
   1fe44:	00d76733          	or	a4,a4,a3
   1fe48:	00e7a223          	sw	a4,4(a5)
   1fe4c:	fef892e3          	bne	a7,a5,1fe30 <__subtf3+0x1068>
   1fe50:	8a5ff06f          	j	1f6f4 <__subtf3+0x92c>
   1fe54:	02812823          	sw	s0,48(sp)
   1fe58:	03812a23          	sw	s8,52(sp)
   1fe5c:	03612c23          	sw	s6,56(sp)
   1fe60:	03312e23          	sw	s3,60(sp)
   1fe64:	00090a93          	mv	s5,s2
   1fe68:	fff80493          	addi	s1,a6,-1 # 7fff <exit-0x80b5>
   1fe6c:	978ff06f          	j	1efe4 <__subtf3+0x21c>
   1fe70:	00051c63          	bnez	a0,1fe88 <__subtf3+0x10c0>
   1fe74:	02f12823          	sw	a5,48(sp)
   1fe78:	02d12a23          	sw	a3,52(sp)
   1fe7c:	02c12c23          	sw	a2,56(sp)
   1fe80:	02e12e23          	sw	a4,60(sp)
   1fe84:	fe5ff06f          	j	1fe68 <__subtf3+0x10a0>
   1fe88:	03012e23          	sw	a6,60(sp)
   1fe8c:	02012c23          	sw	zero,56(sp)
   1fe90:	02012a23          	sw	zero,52(sp)
   1fe94:	02012823          	sw	zero,48(sp)
   1fe98:	03c10793          	addi	a5,sp,60
   1fe9c:	0007a703          	lw	a4,0(a5)
   1fea0:	ffc7a683          	lw	a3,-4(a5)
   1fea4:	ffc78793          	addi	a5,a5,-4
   1fea8:	00371713          	slli	a4,a4,0x3
   1feac:	01d6d693          	srli	a3,a3,0x1d
   1feb0:	00d76733          	or	a4,a4,a3
   1feb4:	00e7a223          	sw	a4,4(a5)
   1feb8:	fef892e3          	bne	a7,a5,1fe9c <__subtf3+0x10d4>
   1febc:	839ff06f          	j	1f6f4 <__subtf3+0x92c>
   1fec0:	40878533          	sub	a0,a5,s0
   1fec4:	41868eb3          	sub	t4,a3,s8
   1fec8:	00a7b833          	sltu	a6,a5,a0
   1fecc:	01d6b333          	sltu	t1,a3,t4
   1fed0:	410e8833          	sub	a6,t4,a6
   1fed4:	00000593          	li	a1,0
   1fed8:	00a7f463          	bgeu	a5,a0,1fee0 <__subtf3+0x1118>
   1fedc:	001eb593          	seqz	a1,t4
   1fee0:	0065e5b3          	or	a1,a1,t1
   1fee4:	41660333          	sub	t1,a2,s6
   1fee8:	00663fb3          	sltu	t6,a2,t1
   1feec:	40b30f33          	sub	t5,t1,a1
   1fef0:	00000e13          	li	t3,0
   1fef4:	00058463          	beqz	a1,1fefc <__subtf3+0x1134>
   1fef8:	00133e13          	seqz	t3,t1
   1fefc:	01fe6e33          	or	t3,t3,t6
   1ff00:	413705b3          	sub	a1,a4,s3
   1ff04:	41c585b3          	sub	a1,a1,t3
   1ff08:	02b12e23          	sw	a1,60(sp)
   1ff0c:	03e12c23          	sw	t5,56(sp)
   1ff10:	03012a23          	sw	a6,52(sp)
   1ff14:	02a12823          	sw	a0,48(sp)
   1ff18:	00c59e13          	slli	t3,a1,0xc
   1ff1c:	140e5663          	bgez	t3,20068 <__subtf3+0x12a0>
   1ff20:	40f407b3          	sub	a5,s0,a5
   1ff24:	40dc06b3          	sub	a3,s8,a3
   1ff28:	00f435b3          	sltu	a1,s0,a5
   1ff2c:	00dc3c33          	sltu	s8,s8,a3
   1ff30:	40b686b3          	sub	a3,a3,a1
   1ff34:	00000593          	li	a1,0
   1ff38:	00f47463          	bgeu	s0,a5,1ff40 <__subtf3+0x1178>
   1ff3c:	001eb593          	seqz	a1,t4
   1ff40:	40cb0633          	sub	a2,s6,a2
   1ff44:	0185ec33          	or	s8,a1,s8
   1ff48:	00cb3b33          	sltu	s6,s6,a2
   1ff4c:	41860633          	sub	a2,a2,s8
   1ff50:	000c0463          	beqz	s8,1ff58 <__subtf3+0x1190>
   1ff54:	00133893          	seqz	a7,t1
   1ff58:	40e985b3          	sub	a1,s3,a4
   1ff5c:	0168e733          	or	a4,a7,s6
   1ff60:	40e58733          	sub	a4,a1,a4
   1ff64:	02e12e23          	sw	a4,60(sp)
   1ff68:	02c12c23          	sw	a2,56(sp)
   1ff6c:	02d12a23          	sw	a3,52(sp)
   1ff70:	02f12823          	sw	a5,48(sp)
   1ff74:	00090a93          	mv	s5,s2
   1ff78:	03c12503          	lw	a0,60(sp)
   1ff7c:	10050063          	beqz	a0,2007c <__subtf3+0x12b4>
   1ff80:	7bc000ef          	jal	2073c <__clzsi2>
   1ff84:	ff450513          	addi	a0,a0,-12
   1ff88:	02000713          	li	a4,32
   1ff8c:	01f57813          	andi	a6,a0,31
   1ff90:	02e547b3          	div	a5,a0,a4
   1ff94:	12080063          	beqz	a6,200b4 <__subtf3+0x12ec>
   1ff98:	03010313          	addi	t1,sp,48
   1ff9c:	02e566b3          	rem	a3,a0,a4
   1ffa0:	40d70633          	sub	a2,a4,a3
   1ffa4:	ffc00693          	li	a3,-4
   1ffa8:	02d786b3          	mul	a3,a5,a3
   1ffac:	00c68713          	addi	a4,a3,12
   1ffb0:	00e30733          	add	a4,t1,a4
   1ffb4:	40d006b3          	neg	a3,a3
   1ffb8:	12e31663          	bne	t1,a4,200e4 <__subtf3+0x131c>
   1ffbc:	03012683          	lw	a3,48(sp)
   1ffc0:	fff78713          	addi	a4,a5,-1
   1ffc4:	00279793          	slli	a5,a5,0x2
   1ffc8:	04078793          	addi	a5,a5,64
   1ffcc:	002787b3          	add	a5,a5,sp
   1ffd0:	010696b3          	sll	a3,a3,a6
   1ffd4:	fed7a823          	sw	a3,-16(a5)
   1ffd8:	00170713          	addi	a4,a4,1
   1ffdc:	03010793          	addi	a5,sp,48
   1ffe0:	00271713          	slli	a4,a4,0x2
   1ffe4:	00800693          	li	a3,8
   1ffe8:	00078893          	mv	a7,a5
   1ffec:	00d76a63          	bltu	a4,a3,20000 <__subtf3+0x1238>
   1fff0:	02012823          	sw	zero,48(sp)
   1fff4:	0007a223          	sw	zero,4(a5)
   1fff8:	ff870713          	addi	a4,a4,-8
   1fffc:	03810793          	addi	a5,sp,56
   20000:	00400693          	li	a3,4
   20004:	00d76463          	bltu	a4,a3,2000c <__subtf3+0x1244>
   20008:	0007a023          	sw	zero,0(a5)
   2000c:	1c954863          	blt	a0,s1,201dc <__subtf3+0x1414>
   20010:	40950533          	sub	a0,a0,s1
   20014:	00150513          	addi	a0,a0,1
   20018:	40555713          	srai	a4,a0,0x5
   2001c:	01f57793          	andi	a5,a0,31
   20020:	00088593          	mv	a1,a7
   20024:	00088613          	mv	a2,a7
   20028:	00000313          	li	t1,0
   2002c:	00000693          	li	a3,0
   20030:	0ce69c63          	bne	a3,a4,20108 <__subtf3+0x1340>
   20034:	00300693          	li	a3,3
   20038:	40e686b3          	sub	a3,a3,a4
   2003c:	00271613          	slli	a2,a4,0x2
   20040:	0c079e63          	bnez	a5,2011c <__subtf3+0x1354>
   20044:	00c58533          	add	a0,a1,a2
   20048:	00052503          	lw	a0,0(a0)
   2004c:	00178793          	addi	a5,a5,1
   20050:	00458593          	addi	a1,a1,4
   20054:	fea5ae23          	sw	a0,-4(a1)
   20058:	fef6d6e3          	bge	a3,a5,20044 <__subtf3+0x127c>
   2005c:	00400793          	li	a5,4
   20060:	40e78733          	sub	a4,a5,a4
   20064:	1040006f          	j	20168 <__subtf3+0x13a0>
   20068:	01056533          	or	a0,a0,a6
   2006c:	01e56533          	or	a0,a0,t5
   20070:	00b56533          	or	a0,a0,a1
   20074:	ca0506e3          	beqz	a0,1fd20 <__subtf3+0xf58>
   20078:	f01ff06f          	j	1ff78 <__subtf3+0x11b0>
   2007c:	03812503          	lw	a0,56(sp)
   20080:	00050863          	beqz	a0,20090 <__subtf3+0x12c8>
   20084:	6b8000ef          	jal	2073c <__clzsi2>
   20088:	02050513          	addi	a0,a0,32
   2008c:	ef9ff06f          	j	1ff84 <__subtf3+0x11bc>
   20090:	03412503          	lw	a0,52(sp)
   20094:	00050863          	beqz	a0,200a4 <__subtf3+0x12dc>
   20098:	6a4000ef          	jal	2073c <__clzsi2>
   2009c:	04050513          	addi	a0,a0,64
   200a0:	ee5ff06f          	j	1ff84 <__subtf3+0x11bc>
   200a4:	03012503          	lw	a0,48(sp)
   200a8:	694000ef          	jal	2073c <__clzsi2>
   200ac:	06050513          	addi	a0,a0,96
   200b0:	ed5ff06f          	j	1ff84 <__subtf3+0x11bc>
   200b4:	ffc00613          	li	a2,-4
   200b8:	02c78633          	mul	a2,a5,a2
   200bc:	03c10713          	addi	a4,sp,60
   200c0:	00300693          	li	a3,3
   200c4:	00c705b3          	add	a1,a4,a2
   200c8:	0005a583          	lw	a1,0(a1)
   200cc:	fff68693          	addi	a3,a3,-1
   200d0:	ffc70713          	addi	a4,a4,-4
   200d4:	00b72223          	sw	a1,4(a4)
   200d8:	fef6d6e3          	bge	a3,a5,200c4 <__subtf3+0x12fc>
   200dc:	fff78713          	addi	a4,a5,-1
   200e0:	ef9ff06f          	j	1ffd8 <__subtf3+0x1210>
   200e4:	00072583          	lw	a1,0(a4)
   200e8:	ffc72883          	lw	a7,-4(a4)
   200ec:	00d70e33          	add	t3,a4,a3
   200f0:	010595b3          	sll	a1,a1,a6
   200f4:	00c8d8b3          	srl	a7,a7,a2
   200f8:	0115e5b3          	or	a1,a1,a7
   200fc:	00be2023          	sw	a1,0(t3)
   20100:	ffc70713          	addi	a4,a4,-4
   20104:	eb5ff06f          	j	1ffb8 <__subtf3+0x11f0>
   20108:	00062503          	lw	a0,0(a2)
   2010c:	00168693          	addi	a3,a3,1
   20110:	00460613          	addi	a2,a2,4
   20114:	00a36333          	or	t1,t1,a0
   20118:	f19ff06f          	j	20030 <__subtf3+0x1268>
   2011c:	04060593          	addi	a1,a2,64
   20120:	002585b3          	add	a1,a1,sp
   20124:	ff05a583          	lw	a1,-16(a1)
   20128:	02000813          	li	a6,32
   2012c:	40f80833          	sub	a6,a6,a5
   20130:	010595b3          	sll	a1,a1,a6
   20134:	00b36333          	or	t1,t1,a1
   20138:	00000e13          	li	t3,0
   2013c:	00c885b3          	add	a1,a7,a2
   20140:	40c00633          	neg	a2,a2
   20144:	06de4863          	blt	t3,a3,201b4 <__subtf3+0x13ec>
   20148:	00400613          	li	a2,4
   2014c:	40e60733          	sub	a4,a2,a4
   20150:	03c12603          	lw	a2,60(sp)
   20154:	00269693          	slli	a3,a3,0x2
   20158:	04068693          	addi	a3,a3,64
   2015c:	002686b3          	add	a3,a3,sp
   20160:	00f657b3          	srl	a5,a2,a5
   20164:	fef6a823          	sw	a5,-16(a3)
   20168:	00400693          	li	a3,4
   2016c:	40e686b3          	sub	a3,a3,a4
   20170:	00271713          	slli	a4,a4,0x2
   20174:	00e887b3          	add	a5,a7,a4
   20178:	00269713          	slli	a4,a3,0x2
   2017c:	00800693          	li	a3,8
   20180:	00d76a63          	bltu	a4,a3,20194 <__subtf3+0x13cc>
   20184:	0007a023          	sw	zero,0(a5)
   20188:	0007a223          	sw	zero,4(a5)
   2018c:	ff870713          	addi	a4,a4,-8
   20190:	00878793          	addi	a5,a5,8
   20194:	00400693          	li	a3,4
   20198:	00d76463          	bltu	a4,a3,201a0 <__subtf3+0x13d8>
   2019c:	0007a023          	sw	zero,0(a5)
   201a0:	03012703          	lw	a4,48(sp)
   201a4:	006037b3          	snez	a5,t1
   201a8:	00f767b3          	or	a5,a4,a5
   201ac:	02f12823          	sw	a5,48(sp)
   201b0:	c60ff06f          	j	1f610 <__subtf3+0x848>
   201b4:	0005a503          	lw	a0,0(a1)
   201b8:	0045ae83          	lw	t4,4(a1)
   201bc:	00c58f33          	add	t5,a1,a2
   201c0:	00f55533          	srl	a0,a0,a5
   201c4:	010e9eb3          	sll	t4,t4,a6
   201c8:	01d56533          	or	a0,a0,t4
   201cc:	00af2023          	sw	a0,0(t5)
   201d0:	001e0e13          	addi	t3,t3,1
   201d4:	00458593          	addi	a1,a1,4
   201d8:	f6dff06f          	j	20144 <__subtf3+0x137c>
   201dc:	03c12783          	lw	a5,60(sp)
   201e0:	fff80737          	lui	a4,0xfff80
   201e4:	fff70713          	addi	a4,a4,-1 # fff7ffff <__BSS_END__+0xfff5d1ff>
   201e8:	00e7f7b3          	and	a5,a5,a4
   201ec:	40a484b3          	sub	s1,s1,a0
   201f0:	02f12e23          	sw	a5,60(sp)
   201f4:	df1fe06f          	j	1efe4 <__subtf3+0x21c>
   201f8:	02012e23          	sw	zero,60(sp)
   201fc:	02012c23          	sw	zero,56(sp)
   20200:	02012a23          	sw	zero,52(sp)
   20204:	02012823          	sw	zero,48(sp)
   20208:	e5dfe06f          	j	1f064 <__subtf3+0x29c>
   2020c:	07400713          	li	a4,116
   20210:	00c74463          	blt	a4,a2,20218 <__subtf3+0x1450>
   20214:	f51fe06f          	j	1f164 <__subtf3+0x39c>
   20218:	02012623          	sw	zero,44(sp)
   2021c:	02012423          	sw	zero,40(sp)
   20220:	02012223          	sw	zero,36(sp)
   20224:	00100713          	li	a4,1
   20228:	82cff06f          	j	1f254 <__subtf3+0x48c>
   2022c:	07400793          	li	a5,116
   20230:	a0b7de63          	bge	a5,a1,1f44c <__subtf3+0x684>
   20234:	00012e23          	sw	zero,28(sp)
   20238:	00012c23          	sw	zero,24(sp)
   2023c:	00012a23          	sw	zero,20(sp)
   20240:	00100793          	li	a5,1
   20244:	b00ff06f          	j	1f544 <__subtf3+0x77c>
   20248:	07400793          	li	a5,116
   2024c:	a717c4e3          	blt	a5,a7,1fcb4 <__subtf3+0xeec>
   20250:	00088313          	mv	t1,a7
   20254:	8d5ff06f          	j	1fb28 <__subtf3+0xd60>

00020258 <__unordtf2>:
   20258:	00052703          	lw	a4,0(a0)
   2025c:	00452e83          	lw	t4,4(a0)
   20260:	00852e03          	lw	t3,8(a0)
   20264:	00c52503          	lw	a0,12(a0)
   20268:	00c5a603          	lw	a2,12(a1)
   2026c:	000086b7          	lui	a3,0x8
   20270:	fff68693          	addi	a3,a3,-1 # 7fff <exit-0x80b5>
   20274:	01055813          	srli	a6,a0,0x10
   20278:	0005a783          	lw	a5,0(a1)
   2027c:	0045a303          	lw	t1,4(a1)
   20280:	0085a883          	lw	a7,8(a1)
   20284:	00d87833          	and	a6,a6,a3
   20288:	01065593          	srli	a1,a2,0x10
   2028c:	ff010113          	addi	sp,sp,-16
   20290:	00d5f5b3          	and	a1,a1,a3
   20294:	02d81063          	bne	a6,a3,202b4 <__unordtf2+0x5c>
   20298:	01d76733          	or	a4,a4,t4
   2029c:	01051513          	slli	a0,a0,0x10
   202a0:	01055513          	srli	a0,a0,0x10
   202a4:	01c76733          	or	a4,a4,t3
   202a8:	00a76733          	or	a4,a4,a0
   202ac:	00100513          	li	a0,1
   202b0:	02071663          	bnez	a4,202dc <__unordtf2+0x84>
   202b4:	00008737          	lui	a4,0x8
   202b8:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   202bc:	00000513          	li	a0,0
   202c0:	00e59e63          	bne	a1,a4,202dc <__unordtf2+0x84>
   202c4:	0067e533          	or	a0,a5,t1
   202c8:	01061613          	slli	a2,a2,0x10
   202cc:	01156533          	or	a0,a0,a7
   202d0:	01065613          	srli	a2,a2,0x10
   202d4:	00c56533          	or	a0,a0,a2
   202d8:	00a03533          	snez	a0,a0
   202dc:	01010113          	addi	sp,sp,16
   202e0:	00008067          	ret

000202e4 <__fixtfsi>:
   202e4:	00052703          	lw	a4,0(a0)
   202e8:	00452683          	lw	a3,4(a0)
   202ec:	00c52783          	lw	a5,12(a0)
   202f0:	00852583          	lw	a1,8(a0)
   202f4:	fe010113          	addi	sp,sp,-32
   202f8:	00e12023          	sw	a4,0(sp)
   202fc:	00d12223          	sw	a3,4(sp)
   20300:	00e12823          	sw	a4,16(sp)
   20304:	00179693          	slli	a3,a5,0x1
   20308:	00004737          	lui	a4,0x4
   2030c:	00b12423          	sw	a1,8(sp)
   20310:	00f12623          	sw	a5,12(sp)
   20314:	00b12c23          	sw	a1,24(sp)
   20318:	0116d693          	srli	a3,a3,0x11
   2031c:	ffe70713          	addi	a4,a4,-2 # 3ffe <exit-0xc0b6>
   20320:	00000513          	li	a0,0
   20324:	02d75063          	bge	a4,a3,20344 <__fixtfsi+0x60>
   20328:	00004737          	lui	a4,0x4
   2032c:	01d70713          	addi	a4,a4,29 # 401d <exit-0xc097>
   20330:	01f7d813          	srli	a6,a5,0x1f
   20334:	00d75c63          	bge	a4,a3,2034c <__fixtfsi+0x68>
   20338:	80000537          	lui	a0,0x80000
   2033c:	fff50513          	addi	a0,a0,-1 # 7fffffff <__BSS_END__+0x7ffdd1ff>
   20340:	00a80533          	add	a0,a6,a0
   20344:	02010113          	addi	sp,sp,32
   20348:	00008067          	ret
   2034c:	01079793          	slli	a5,a5,0x10
   20350:	00010737          	lui	a4,0x10
   20354:	0107d793          	srli	a5,a5,0x10
   20358:	00e7e7b3          	or	a5,a5,a4
   2035c:	00004737          	lui	a4,0x4
   20360:	06f70713          	addi	a4,a4,111 # 406f <exit-0xc045>
   20364:	40d70733          	sub	a4,a4,a3
   20368:	40575613          	srai	a2,a4,0x5
   2036c:	00f12e23          	sw	a5,28(sp)
   20370:	01f77713          	andi	a4,a4,31
   20374:	02071463          	bnez	a4,2039c <__fixtfsi+0xb8>
   20378:	00261613          	slli	a2,a2,0x2
   2037c:	02060793          	addi	a5,a2,32
   20380:	00278633          	add	a2,a5,sp
   20384:	ff062783          	lw	a5,-16(a2)
   20388:	00f12823          	sw	a5,16(sp)
   2038c:	01012503          	lw	a0,16(sp)
   20390:	fa080ae3          	beqz	a6,20344 <__fixtfsi+0x60>
   20394:	40a00533          	neg	a0,a0
   20398:	fadff06f          	j	20344 <__fixtfsi+0x60>
   2039c:	00200513          	li	a0,2
   203a0:	00000693          	li	a3,0
   203a4:	02a61063          	bne	a2,a0,203c4 <__fixtfsi+0xe0>
   203a8:	02000693          	li	a3,32
   203ac:	40e686b3          	sub	a3,a3,a4
   203b0:	00d796b3          	sll	a3,a5,a3
   203b4:	00e5d5b3          	srl	a1,a1,a4
   203b8:	00b6e6b3          	or	a3,a3,a1
   203bc:	00d12823          	sw	a3,16(sp)
   203c0:	00100693          	li	a3,1
   203c4:	00269693          	slli	a3,a3,0x2
   203c8:	02068693          	addi	a3,a3,32
   203cc:	002686b3          	add	a3,a3,sp
   203d0:	00e7d7b3          	srl	a5,a5,a4
   203d4:	fef6a823          	sw	a5,-16(a3)
   203d8:	fb5ff06f          	j	2038c <__fixtfsi+0xa8>

000203dc <__floatsitf>:
   203dc:	fd010113          	addi	sp,sp,-48
   203e0:	02912223          	sw	s1,36(sp)
   203e4:	02112623          	sw	ra,44(sp)
   203e8:	02812423          	sw	s0,40(sp)
   203ec:	03212023          	sw	s2,32(sp)
   203f0:	00050493          	mv	s1,a0
   203f4:	12058263          	beqz	a1,20518 <__floatsitf+0x13c>
   203f8:	41f5d793          	srai	a5,a1,0x1f
   203fc:	00b7c433          	xor	s0,a5,a1
   20400:	40f40433          	sub	s0,s0,a5
   20404:	00040513          	mv	a0,s0
   20408:	01f5d913          	srli	s2,a1,0x1f
   2040c:	330000ef          	jal	2073c <__clzsi2>
   20410:	00004737          	lui	a4,0x4
   20414:	01e70713          	addi	a4,a4,30 # 401e <exit-0xc096>
   20418:	05150793          	addi	a5,a0,81
   2041c:	40a70633          	sub	a2,a4,a0
   20420:	00812823          	sw	s0,16(sp)
   20424:	4057d713          	srai	a4,a5,0x5
   20428:	00012a23          	sw	zero,20(sp)
   2042c:	00012c23          	sw	zero,24(sp)
   20430:	00012e23          	sw	zero,28(sp)
   20434:	01f7f793          	andi	a5,a5,31
   20438:	02078c63          	beqz	a5,20470 <__floatsitf+0x94>
   2043c:	00200693          	li	a3,2
   20440:	0cd71863          	bne	a4,a3,20510 <__floatsitf+0x134>
   20444:	02000693          	li	a3,32
   20448:	40f686b3          	sub	a3,a3,a5
   2044c:	00d456b3          	srl	a3,s0,a3
   20450:	00d12e23          	sw	a3,28(sp)
   20454:	fff70693          	addi	a3,a4,-1
   20458:	00271713          	slli	a4,a4,0x2
   2045c:	02070713          	addi	a4,a4,32
   20460:	00270733          	add	a4,a4,sp
   20464:	00f41433          	sll	s0,s0,a5
   20468:	fe872823          	sw	s0,-16(a4)
   2046c:	0340006f          	j	204a0 <__floatsitf+0xc4>
   20470:	00300793          	li	a5,3
   20474:	40e787b3          	sub	a5,a5,a4
   20478:	00279793          	slli	a5,a5,0x2
   2047c:	02078793          	addi	a5,a5,32
   20480:	002787b3          	add	a5,a5,sp
   20484:	ff07a783          	lw	a5,-16(a5)
   20488:	00200693          	li	a3,2
   2048c:	00f12e23          	sw	a5,28(sp)
   20490:	00200793          	li	a5,2
   20494:	00f71663          	bne	a4,a5,204a0 <__floatsitf+0xc4>
   20498:	00812c23          	sw	s0,24(sp)
   2049c:	00100693          	li	a3,1
   204a0:	00269693          	slli	a3,a3,0x2
   204a4:	00012823          	sw	zero,16(sp)
   204a8:	00012a23          	sw	zero,20(sp)
   204ac:	ffc68693          	addi	a3,a3,-4
   204b0:	00400793          	li	a5,4
   204b4:	00f6e463          	bltu	a3,a5,204bc <__floatsitf+0xe0>
   204b8:	00012c23          	sw	zero,24(sp)
   204bc:	00090593          	mv	a1,s2
   204c0:	01c12783          	lw	a5,28(sp)
   204c4:	00f59413          	slli	s0,a1,0xf
   204c8:	00c46433          	or	s0,s0,a2
   204cc:	00f11623          	sh	a5,12(sp)
   204d0:	01012783          	lw	a5,16(sp)
   204d4:	00811723          	sh	s0,14(sp)
   204d8:	02c12083          	lw	ra,44(sp)
   204dc:	00f4a023          	sw	a5,0(s1)
   204e0:	01412783          	lw	a5,20(sp)
   204e4:	02812403          	lw	s0,40(sp)
   204e8:	02012903          	lw	s2,32(sp)
   204ec:	00f4a223          	sw	a5,4(s1)
   204f0:	01812783          	lw	a5,24(sp)
   204f4:	00048513          	mv	a0,s1
   204f8:	00f4a423          	sw	a5,8(s1)
   204fc:	00c12783          	lw	a5,12(sp)
   20500:	00f4a623          	sw	a5,12(s1)
   20504:	02412483          	lw	s1,36(sp)
   20508:	03010113          	addi	sp,sp,48
   2050c:	00008067          	ret
   20510:	00300713          	li	a4,3
   20514:	f41ff06f          	j	20454 <__floatsitf+0x78>
   20518:	00012e23          	sw	zero,28(sp)
   2051c:	00012c23          	sw	zero,24(sp)
   20520:	00012a23          	sw	zero,20(sp)
   20524:	00012823          	sw	zero,16(sp)
   20528:	00000613          	li	a2,0
   2052c:	f95ff06f          	j	204c0 <__floatsitf+0xe4>

00020530 <__extenddftf2>:
   20530:	01465713          	srli	a4,a2,0x14
   20534:	00c61793          	slli	a5,a2,0xc
   20538:	7ff77713          	andi	a4,a4,2047
   2053c:	fd010113          	addi	sp,sp,-48
   20540:	00c7d793          	srli	a5,a5,0xc
   20544:	00170693          	addi	a3,a4,1
   20548:	02812423          	sw	s0,40(sp)
   2054c:	02912223          	sw	s1,36(sp)
   20550:	03212023          	sw	s2,32(sp)
   20554:	02112623          	sw	ra,44(sp)
   20558:	00b12823          	sw	a1,16(sp)
   2055c:	00f12a23          	sw	a5,20(sp)
   20560:	00012e23          	sw	zero,28(sp)
   20564:	00012c23          	sw	zero,24(sp)
   20568:	7fe6f693          	andi	a3,a3,2046
   2056c:	00050913          	mv	s2,a0
   20570:	00058413          	mv	s0,a1
   20574:	01f65493          	srli	s1,a2,0x1f
   20578:	08068263          	beqz	a3,205fc <__extenddftf2+0xcc>
   2057c:	000046b7          	lui	a3,0x4
   20580:	c0068693          	addi	a3,a3,-1024 # 3c00 <exit-0xc4b4>
   20584:	00d70733          	add	a4,a4,a3
   20588:	0047d693          	srli	a3,a5,0x4
   2058c:	00d12e23          	sw	a3,28(sp)
   20590:	01c79793          	slli	a5,a5,0x1c
   20594:	0045d693          	srli	a3,a1,0x4
   20598:	00d7e7b3          	or	a5,a5,a3
   2059c:	01c59413          	slli	s0,a1,0x1c
   205a0:	00f12c23          	sw	a5,24(sp)
   205a4:	00812a23          	sw	s0,20(sp)
   205a8:	00012823          	sw	zero,16(sp)
   205ac:	01c12783          	lw	a5,28(sp)
   205b0:	00f49493          	slli	s1,s1,0xf
   205b4:	00e4e4b3          	or	s1,s1,a4
   205b8:	00f11623          	sh	a5,12(sp)
   205bc:	01012783          	lw	a5,16(sp)
   205c0:	00911723          	sh	s1,14(sp)
   205c4:	02c12083          	lw	ra,44(sp)
   205c8:	00f92023          	sw	a5,0(s2)
   205cc:	01412783          	lw	a5,20(sp)
   205d0:	02812403          	lw	s0,40(sp)
   205d4:	02412483          	lw	s1,36(sp)
   205d8:	00f92223          	sw	a5,4(s2)
   205dc:	01812783          	lw	a5,24(sp)
   205e0:	00090513          	mv	a0,s2
   205e4:	00f92423          	sw	a5,8(s2)
   205e8:	00c12783          	lw	a5,12(sp)
   205ec:	00f92623          	sw	a5,12(s2)
   205f0:	02012903          	lw	s2,32(sp)
   205f4:	03010113          	addi	sp,sp,48
   205f8:	00008067          	ret
   205fc:	00b7e533          	or	a0,a5,a1
   20600:	10071063          	bnez	a4,20700 <__extenddftf2+0x1d0>
   20604:	fa0504e3          	beqz	a0,205ac <__extenddftf2+0x7c>
   20608:	04078e63          	beqz	a5,20664 <__extenddftf2+0x134>
   2060c:	00078513          	mv	a0,a5
   20610:	12c000ef          	jal	2073c <__clzsi2>
   20614:	03150693          	addi	a3,a0,49
   20618:	4056d793          	srai	a5,a3,0x5
   2061c:	01f6f693          	andi	a3,a3,31
   20620:	04068863          	beqz	a3,20670 <__extenddftf2+0x140>
   20624:	ffc00613          	li	a2,-4
   20628:	02c78633          	mul	a2,a5,a2
   2062c:	02000813          	li	a6,32
   20630:	01010313          	addi	t1,sp,16
   20634:	40d80833          	sub	a6,a6,a3
   20638:	00c60713          	addi	a4,a2,12
   2063c:	00e30733          	add	a4,t1,a4
   20640:	40c00633          	neg	a2,a2
   20644:	08e31c63          	bne	t1,a4,206dc <__extenddftf2+0x1ac>
   20648:	fff78713          	addi	a4,a5,-1
   2064c:	00279793          	slli	a5,a5,0x2
   20650:	02078793          	addi	a5,a5,32
   20654:	002787b3          	add	a5,a5,sp
   20658:	00d416b3          	sll	a3,s0,a3
   2065c:	fed7a823          	sw	a3,-16(a5)
   20660:	03c0006f          	j	2069c <__extenddftf2+0x16c>
   20664:	0d8000ef          	jal	2073c <__clzsi2>
   20668:	02050513          	addi	a0,a0,32
   2066c:	fa9ff06f          	j	20614 <__extenddftf2+0xe4>
   20670:	ffc00613          	li	a2,-4
   20674:	02c78633          	mul	a2,a5,a2
   20678:	01c10713          	addi	a4,sp,28
   2067c:	00300693          	li	a3,3
   20680:	00c705b3          	add	a1,a4,a2
   20684:	0005a583          	lw	a1,0(a1)
   20688:	fff68693          	addi	a3,a3,-1
   2068c:	ffc70713          	addi	a4,a4,-4
   20690:	00b72223          	sw	a1,4(a4)
   20694:	fef6d6e3          	bge	a3,a5,20680 <__extenddftf2+0x150>
   20698:	fff78713          	addi	a4,a5,-1
   2069c:	00170793          	addi	a5,a4,1
   206a0:	00279793          	slli	a5,a5,0x2
   206a4:	00800693          	li	a3,8
   206a8:	01010713          	addi	a4,sp,16
   206ac:	00d7ea63          	bltu	a5,a3,206c0 <__extenddftf2+0x190>
   206b0:	00012823          	sw	zero,16(sp)
   206b4:	00072223          	sw	zero,4(a4)
   206b8:	ff878793          	addi	a5,a5,-8
   206bc:	01810713          	addi	a4,sp,24
   206c0:	00400693          	li	a3,4
   206c4:	00d7e463          	bltu	a5,a3,206cc <__extenddftf2+0x19c>
   206c8:	00072023          	sw	zero,0(a4)
   206cc:	00004737          	lui	a4,0x4
   206d0:	c0c70713          	addi	a4,a4,-1012 # 3c0c <exit-0xc4a8>
   206d4:	40a70733          	sub	a4,a4,a0
   206d8:	ed5ff06f          	j	205ac <__extenddftf2+0x7c>
   206dc:	00072583          	lw	a1,0(a4)
   206e0:	ffc72883          	lw	a7,-4(a4)
   206e4:	00c70e33          	add	t3,a4,a2
   206e8:	00d595b3          	sll	a1,a1,a3
   206ec:	0108d8b3          	srl	a7,a7,a6
   206f0:	0115e5b3          	or	a1,a1,a7
   206f4:	00be2023          	sw	a1,0(t3)
   206f8:	ffc70713          	addi	a4,a4,-4
   206fc:	f49ff06f          	j	20644 <__extenddftf2+0x114>
   20700:	02050863          	beqz	a0,20730 <__extenddftf2+0x200>
   20704:	01c79713          	slli	a4,a5,0x1c
   20708:	0045d693          	srli	a3,a1,0x4
   2070c:	00d76733          	or	a4,a4,a3
   20710:	00e12c23          	sw	a4,24(sp)
   20714:	0047d793          	srli	a5,a5,0x4
   20718:	00008737          	lui	a4,0x8
   2071c:	01c59413          	slli	s0,a1,0x1c
   20720:	00e7e7b3          	or	a5,a5,a4
   20724:	00812a23          	sw	s0,20(sp)
   20728:	00012823          	sw	zero,16(sp)
   2072c:	00f12e23          	sw	a5,28(sp)
   20730:	00008737          	lui	a4,0x8
   20734:	fff70713          	addi	a4,a4,-1 # 7fff <exit-0x80b5>
   20738:	e75ff06f          	j	205ac <__extenddftf2+0x7c>

0002073c <__clzsi2>:
   2073c:	000107b7          	lui	a5,0x10
   20740:	02f57a63          	bgeu	a0,a5,20774 <__clzsi2+0x38>
   20744:	10053793          	sltiu	a5,a0,256
   20748:	0017b793          	seqz	a5,a5
   2074c:	00379793          	slli	a5,a5,0x3
   20750:	02000713          	li	a4,32
   20754:	40f70733          	sub	a4,a4,a5
   20758:	00f55533          	srl	a0,a0,a5
   2075c:	00001797          	auipc	a5,0x1
   20760:	99c78793          	addi	a5,a5,-1636 # 210f8 <__clz_tab>
   20764:	00a787b3          	add	a5,a5,a0
   20768:	0007c503          	lbu	a0,0(a5)
   2076c:	40a70533          	sub	a0,a4,a0
   20770:	00008067          	ret
   20774:	01000737          	lui	a4,0x1000
   20778:	01800793          	li	a5,24
   2077c:	fce57ae3          	bgeu	a0,a4,20750 <__clzsi2+0x14>
   20780:	01000793          	li	a5,16
   20784:	fcdff06f          	j	20750 <__clzsi2+0x14>

00020788 <_close>:
   20788:	05800793          	li	a5,88
   2078c:	f6f1a623          	sw	a5,-148(gp) # 229ec <errno>
   20790:	fff00513          	li	a0,-1
   20794:	00008067          	ret

00020798 <_fstat>:
   20798:	05800793          	li	a5,88
   2079c:	f6f1a623          	sw	a5,-148(gp) # 229ec <errno>
   207a0:	fff00513          	li	a0,-1
   207a4:	00008067          	ret

000207a8 <_getpid>:
   207a8:	05800793          	li	a5,88
   207ac:	f6f1a623          	sw	a5,-148(gp) # 229ec <errno>
   207b0:	fff00513          	li	a0,-1
   207b4:	00008067          	ret

000207b8 <_isatty>:
   207b8:	05800793          	li	a5,88
   207bc:	f6f1a623          	sw	a5,-148(gp) # 229ec <errno>
   207c0:	00000513          	li	a0,0
   207c4:	00008067          	ret

000207c8 <_kill>:
   207c8:	05800793          	li	a5,88
   207cc:	f6f1a623          	sw	a5,-148(gp) # 229ec <errno>
   207d0:	fff00513          	li	a0,-1
   207d4:	00008067          	ret

000207d8 <_lseek>:
   207d8:	05800793          	li	a5,88
   207dc:	f6f1a623          	sw	a5,-148(gp) # 229ec <errno>
   207e0:	fff00513          	li	a0,-1
   207e4:	00008067          	ret

000207e8 <_read>:
   207e8:	05800793          	li	a5,88
   207ec:	f6f1a623          	sw	a5,-148(gp) # 229ec <errno>
   207f0:	fff00513          	li	a0,-1
   207f4:	00008067          	ret

000207f8 <_sbrk>:
   207f8:	f8418713          	addi	a4,gp,-124 # 22a04 <heap_end.0>
   207fc:	00072783          	lw	a5,0(a4) # 1000000 <__BSS_END__+0xfdd200>
   20800:	00078a63          	beqz	a5,20814 <_sbrk+0x1c>
   20804:	00a78533          	add	a0,a5,a0
   20808:	00a72023          	sw	a0,0(a4)
   2080c:	00078513          	mv	a0,a5
   20810:	00008067          	ret
   20814:	38018793          	addi	a5,gp,896 # 22e00 <__BSS_END__>
   20818:	00a78533          	add	a0,a5,a0
   2081c:	00a72023          	sw	a0,0(a4)
   20820:	00078513          	mv	a0,a5
   20824:	00008067          	ret

00020828 <_write>:
   20828:	05800793          	li	a5,88
   2082c:	f6f1a623          	sw	a5,-148(gp) # 229ec <errno>
   20830:	fff00513          	li	a0,-1
   20834:	00008067          	ret

00020838 <_exit>:
   20838:	0000006f          	j	20838 <_exit>

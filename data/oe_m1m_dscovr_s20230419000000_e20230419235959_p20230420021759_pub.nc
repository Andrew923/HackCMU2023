CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230419000000_e20230419235959_p20230420021759_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-04-20T02:17:59.994Z   date_calibration_data_updated         2023-04-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-04-19T00:00:00.000Z   time_coverage_end         2023-04-19T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxyl��  "          @��R@l(�@z=q��G��z�HB:��@l(�@�z�?!G�@���BAff                                    Bxym&  
�          @�{@aG�@�z������BL��@aG�@��H?�
=A���BE{                                    Bxym�  �          @���@J=q@��ÿ:�H��
=BbQ�@J=q@�Q�?޸RA���B\�                                    Bxym%r  �          @�\)@1G�@��/\)��
=Ba��@1G�@�������AG�Bs�H                                    Bxym4  
�          @�@(�@����S33�\)Bv\)@(�@��ͿY���
ffB��                                    BxymB�  �          @�{@?\)@�  �^�R�Q�Bg(�@?\)@��?���A��Bb��                                    BxymQd  "          @�  @+�@�\)��{��G�Br  @+�@���?:�H@���Bw��                                    Bxym`
  "          @�{@)��@�G�������Bn��@)��@�
=>�G�@��Bw�R                                    Bxymn�  "          @��@e@��\>���@��BI(�@e@c�
@\)A�  B2�H                                    Bxym}V  �          @�\)@HQ�@��R�
=q���Bap�@HQ�@��?�\)A�\)BY�                                    Bxym��  
�          @�@L��@�p���=q�[\)BXG�@L��@�  ?��
A(��BZ=q                                    Bxym��  �          @�p�@=q@����=p���Bo\)@=q@�=q����33B�(�                                    Bxym�H  T          @���@
=q@���XQ��=qBz33@
=q@�G��aG����B��=                                    Bxym��  �          @�=q@\)@�Q��E�����Bn�@\)@�\)����(�B���                                    BxymƔ  "          @��@C�
@��
�Q���ffB\  @C�
@�33>���@AG�Bg{                                    Bxym�:  
�          @���@8��@�  �����G�Be33@8��@���>��@!�Bp�                                    Bxym��  T          @��H@3�
@��
�	�����Bj�\@3�
@��\>��@�  Bt
=                                    Bxym�  
Z          @�33@1G�@�G��(���p�Bj=q@1G�@���=�G�?���Bv��                                    Bxyn,  "          @�33@,��@�33�����  Bn��@,��@�{>8Q�?�  Bz33                                    Bxyn�  �          @��
@/\)@����#33���Bk(�@/\)@�{<#�
=���Bx�R                                    Bxynx  T          @��@1�@����{���
Bi�@1�@��=��
?=p�Bv�
                                    Bxyn-  �          @�z�@7�@��\�33���\Bg@7�@��
>�  @��Br�                                    Bxyn;�  "          @�(�@AG�@��H���=qBb�@AG�@���>�G�@��Bk��                                    BxynJj  
�          @�(�@L(�@�G������HB[�@L(�@���?\)@��RBc                                    BxynY  
�          @�(�@U�@�Q��\���\BV\)@U�@���?+�@��B]�                                    Bxyng�  
�          @��@Z�H@���ٙ���  BR��@Z�H@�  ?:�H@�\BY                                      Bxynv\  
�          @��
@a�@�{��  �mBN  @a�@�33?^�RA(�BR{                                    Bxyn�  
�          @��
@b�\@�
=��{�W
=BNff@b�\@��?�G�A{BP�R                                    Bxyn��  	�          @���@R�\@����\)�\)BY��@R�\@��?Y��A��B^�\                                    Bxyn�N  T          @���@:=q@��ÿ�\)��{Bj��@:=q@��\?5@��Bp�
                                    Bxyn��  T          @���@E@�33����L��Bf=q@E@��
?��RAB{Bf��                                    Bxyn��  T          @���@H��@��ÿ�\)�X  Bb�R@H��@��H?��A3\)Bd
=                                    Bxyn�@  T          @���@J=q@���Ǯ�t��Ba33@J=q@�z�?uA��Bd��                                    Bxyn��  �          @���@Mp�@�  �����X��B_��@Mp�@�=q?�\)A/33Baz�                                    Bxyn�  "          @��
@L(�@�������P��B`
=@L(�@���?�z�A6�\B`��                                    Bxyn�2  "          @��@J=q@����{�W33Ba{@J=q@���?���A1G�Bbp�                                    Bxyo�  �          @�z�@S33@��R���
�H��B[�
@S33@��?�
=A9�B\ff                                    Bxyo~  T          @��
@P  @�{��\)�XQ�B]
=@P  @�Q�?��A+\)B^�R                                    Bxyo&$  "          @��
@K�@��R��Q��c\)B_�@K�@�=q?��A#�Bb=q                                    Bxyo4�  �          @���@J�H@��ÿ����S�
Bb  @J�H@��\?�33A4��Bc�                                    BxyoCp  T          @�@K�@��\���\�E��Bb�\@K�@��H?�  AB�\Bb�                                    BxyoR  
�          @�{@Fff@��\��  �j�HBe\)@Fff@�ff?��A!G�Bg�                                    Bxyo`�  
�          @�@H��@�Q�˅�x��Bb33@H��@�?n{A(�Be�                                    Bxyoob  T          @�
=@%�@����(����RBz�@%�@�=q?333@�G�B�B�                                    Bxyo~  �          @�z�@z=q@�Q�z����RB=��@z=q@�Q�?���Ay�B6�                                    Bxyo��  
�          @��H@���@|�ͽL�;�ffB*(�@���@^{?��A�  B��                                    Bxyo�T  T          @�z�@�G�@��þ����  B,@�G�@fff?���A�{B�                                    Bxyo��  
�          @���@��R@�(����R�C33B1�H@��R@p��?�p�A�=qB'=q                                    Bxyo��  
�          @���@��@��H�����7�B/�@��@n{?�(�A�  B%�                                    Bxyo�F  T          @�@�ff@�p���\)�.�RB3(�@�ff@q�?�\A�G�B({                                    Bxyo��  T          @�
=@G�@�ff���H�:�RBg(�@G�@��?��AO33Bfp�                                    Bxyo�  
�          @�G�@\��@��H��ff� ��BZ=q@\��@�  ?�AY��BX{                                    Bxyo�8  T          @��H@h��@��ÿz�H�z�BR@h��@��?���A\��BO�                                    Bxyp�  T          @�ff@fff@��Ϳk��ffBP�@fff@�Q�?�Q�A`  BM�                                    Bxyp�  �          @�z�@i��@����L�����\BM�@i��@��
?��RAk33BH��                                    Bxyp*  
�          @��
@w�@��
�.{��p�BA�H@w�@���?�G�An=qB<(�                                    Bxyp-�  T          @��@p��@��:�H��(�BF�@p��@�\)?�  AmG�BAz�                                    Bxyp<v  �          @��
@o\)@�{�W
=�\)BG��@o\)@�G�?�z�A]�BC��                                    BxypK  T          @��@o\)@�Q�=p���BIz�@o\)@��?\An�RBDff                                    BxypY�  "          @��@c33@��Ϳ^�R��BR�\@c33@��?�(�Af�RBN�
                                    Bxyphh  �          @�(�@\(�@�{�Q�� Q�BW�@\(�@�Q�?��Ar�HBR�R                                    Bxypw  
�          @�@Z�H@�Q�fff�
=BY{@Z�H@�33?�  Aj�\BUz�                                    Bxyp��  "          @��@Q�@�(���\)�-G�B`\)@Q�@�=q?�{AQ��B_{                                    Bxyp�Z  T          @�@K�@��ÿ�\)�V�\BaQ�@K�@�33?��A*�\Bb�
                                    Bxyp�   "          @�@X��@��H�Ǯ�tz�BV(�@X��@���?Tz�Ap�BZz�                                    Bxyp��  �          @�\)@p  @����u��BJ{@p  @�ff?��AO\)BG��                                    Bxyp�L  
Z          @�
=@Q�@�\)�\)��ffBb��@Q�@�{@G�A���BV(�                                    Bxyp��  
�          @�G�@%@���>.{?У�B��3@%@��@.�RA�
=Bs��                                    Bxypݘ  %          @\@   @��ý��Ϳz�HB�Q�@   @���@#33A��HB{��                                    Bxyp�>  T          @��@��@�\)<#�
=uB�{@��@��@'�A�{B��{                                    Bxyp��  �          @�
=?�Q�@�z������B�#�?�Q�@�\)?=p�@�RB��{                                    Bxyq	�  �          @���?�33@�{�Ǯ�p��B�\?�33@��?��AN{B�G�                                    Bxyq0  "          @�
=@S�
@�{�k��\)B`��@S�
@�ff@	��A�\)BUp�                                    Bxyq&�            @�\)@_\)@�녽��Ϳ�G�BXQ�@_\)@���@��A��
BK\)                                    Bxyq5|  W          @���@z=q@�{?��
Ao�B;�@z=q@=p�@UB	(�B\)                                    BxyqD"  �          @�  @�(�@���?W
=A�RB4��@�(�@P  @-p�A�ffB�H                                    BxyqR�  T          @��H@x��@�z�
=q���HBH
=@x��@��H?�p�A�(�B@p�                                    Bxyqan  �          @\@Mp�@���?n{A\)B`ff@Mp�@p��@EB   BF{                                    Bxyqp  �          @�33@X��@��?�  Adz�B[p�@X��@aG�@fffBz�B8��                                    Bxyq~�  �          @Å@k�@���?s33A
=BQ�R@k�@p��@FffA�=qB6�                                    Bxyq�`  
�          @�p�@���@�ff?��AB{B?@���@S�
@P  A�p�B                                      Bxyq�  "          @�@�Q�@�G�@��A�(�B4�H@�Q�@��@�=qB$(�A�G�                                    Bxyq��  
�          @�@�=q@n{@ffA�p�B"��@�=q@
=q@u�B�A�
=                                    Bxyq�R  �          @ƸR@i��@�z�?��HA�p�BO(�@i��@Q�@l��B{B'�R                                    Bxyq��  "          @�{@6ff@��R?333@�Q�B}��@6ff@�ff@S�
A��\Bk33                                    Bxyq֞  "          @�{@�R@�Q�>�z�@*=qB�aH@�R@�{@>{A�(�B}\)                                    Bxyq�D  
�          @�p�@1G�@�{?B�\@ڏ\B�  @1G�@��@VffA���Bl�                                    Bxyq��  �          @�{@z�@�?c�
@��B�W
@z�@��@e�BffB���                                    Bxyr�  �          @�(�@+�@��R?L��@�{B��3@+�@�p�@X��B �
Bpp�                                    Bxyr6  Q          @ʏ\@"�\@�
=?s33A	p�B�  @"�\@�33@aG�B=qBt33                                    Bxyr�  �          @�=q@   @�\)?fffA�RB��f@   @�(�@^�RB�Bv                                    Bxyr.�  
�          @ʏ\@��@�ff>8Q�?���B�W
@��@�p�@<��A�(�B��                                    Bxyr=(  
�          @�
=@�
@�
=?#�
@��RB�  @�
@�  @P  A��HB��                                    BxyrK�  "          @���?�p�@�{>Ǯ@c�
B��?�p�@�=q@HQ�A�33B�G�                                    BxyrZt  T          @�=q@�@�>�(�@z�HB��R@�@�G�@J=qA�{B�p�                                    Bxyri  �          @ʏ\?�  @��>�@��\B��R?�  @�(�@P  A��
B�G�                                    Bxyrw�  "          @�33?У�@�33?#�
@�G�B�#�?У�@��H@Z=qB=qB��{                                    Bxyr�f  T          @˅?Ǯ@�z�?��@���B�ff?Ǯ@���@X��B  B�.                                    Bxyr�  T          @˅?�G�@�?G�@�Q�B�?�G�@��@dz�B�B�\)                                    Bxyr��  �          @��
?�ff@��
?B�\@�(�B��=?�ff@��@a�B�B��H                                    Bxyr�X  
�          @ʏ\?��@���?.{@�(�B��H?��@���@Z=qB=qB��3                                    Bxyr��  
�          @ə�?�Q�@���?333@�p�B��?�Q�@�  @Z�HB��B��3                                    BxyrϤ  
�          @���?�
=@�
=>�G�@z�HB��?�
=@���@Q�A��\B���                                    Bxyr�J  �          @���?�@�\)>Ǯ@a�B�L�?�@��\@O\)A�\)B�(�                                    Bxyr��  
Z          @˅?Ǯ@���>���@;�B���?Ǯ@���@H��A�
=B�G�                                    Bxyr��  
(          @��H?�Q�@���>�{@G
=B�L�?�Q�@�@EA��B��                                    Bxys
<  �          @��
?�  @���>�?���B��?�  @�Q�@AG�A��B�p�                                    Bxys�  T          @��H?J=q@��ý����B�?J=q@��
@333A�{B��                                    Bxys'�            @��?��\@�
=>B�\?�Q�B�p�?��\@�@B�\A�z�B���                                    Bxys6.  
+          @�33?���@Ǯ=�?���B�\?���@�\)@?\)A�
=B�\)                                    BxysD�  �          @��H?�  @�  =�\)?!G�B��)?�  @�Q�@<(�A�B�z�                                    BxysSz  �          @��?��@�ff��33�L��B�  ?��@��@"�\A���B�=q                                    Bxysb   T          @�Q�?��\@���Ǯ�eB�33?��\@���@\)A�p�B��=                                    Bxysp�  T          @ə�?u@��.{��ffB�k�?u@���@\)A���B�G�                                    Bxysl  T          @���?��
@�z�8Q��ҏ\B���?��
@���@�A��B��
                                    Bxys�  T          @�33?�(�@�(�����  B��{?�(�@�{@ffA�(�B��H                                    Bxys��  
�          @�{?�G�@��ÿ(���
=B�L�?�G�@��@�A�(�B��q                                    Bxys�^  T          @�
=?�G�@�G��W
=��B�W
?�G�@�
=@��A�Q�B�.                                    Bxys�  
�          @�{?�=q@�\)�aG����HB��H?�=q@�{@z�A�=qB�                                    BxysȪ  T          @�?�Q�@�녾�  �(�B��3?�Q�@�\)@*�HA�{B��{                                    Bxys�P  �          @�?}p�@�33=�\)?�RB�B�?}p�@��
@=p�A�p�B�                                      Bxys��  "          @�p�>�(�@�33?�\@��B�=q>�(�@�p�@VffA���B��f                                    Bxys��  �          @�?�\@��H?!G�@�z�B��q?�\@��@\��B�\B�\                                    BxytB  
�          @���?}p�@ə�>��R@4z�B�\?}p�@��R@I��A�=qB�ff                                    Bxyt�  �          @�(�?�G�@Ǯ>.{?ǮB��?�G�@�\)@?\)A�(�B�                                    Bxyt �  �          @��H?��\@�ff���Ϳn{B��?��\@�=q@.�RA�  B�ff                                    Bxyt/4  �          @�33?��\@ƸR�����,��B�Q�?��\@��@#33A�G�B��{                                    Bxyt=�  "          @��>L��@�Q�?�\)A�{B��>L��@�Q�@�z�B0��B���                                    BxytL�  �          @�G�?@  @Å?���A0  B�
=?@  @��@uBB���                                    Bxyt[&  �          @�G�?@  @�{?E�@ᙚB�.?@  @�p�@_\)B�HB��                                     Bxyti�  �          @ə�?��@�z�?E�@��B���?��@�(�@]p�B��B���                                    Bxytxr  �          @�G�?�ff@�{>�{@E�B���?�ff@��
@EA�  B��                                    Bxyt�  T          @Ǯ?xQ�@���>.{?���B��?xQ�@�p�@;�A�(�B��R                                    Bxyt��  �          @�?fff@\?(�@�(�B�8R?fff@���@Q�B Q�B�G�                                    Bxyt�d  �          @�  ?��@�?.{@�G�B��?��@��R@X��B�B��f                                    Bxyt�
  
�          @�  ?�33@�(�>8Q�?�
=B���?�33@�z�@:�HA��B�                                      Bxyt��  �          @�\)?���@���?
=q@���B��3?���@���@H��A���B�.                                    Bxyt�V  T          @�{@�\@��R=�Q�?Q�B�W
@�\@�=q@(��A�G�B�ff                                    Bxyt��  �          @�?n{@�G�>���@qG�B�p�?n{@�
=@Dz�A�=qB��q                                    Bxyt��  �          @�{?���@�p��333��G�B�33?���@��@G�A�
=B���                                    Bxyt�H  �          @���@$z�@�{�����)B�aH@$z�@��
?�AW
=B���                                    Bxyu
�  �          @��@ff@��R�5�ӅB���@ff@�?�A�  B��3                                    Bxyu�  �          @�ff?��
@��׿8Q���Q�B��3?��
@�\)?�
=A�=qB�                                    Bxyu(:  �          @���?0��@Ǯ=�G�?xQ�B�k�?0��@�G�@8Q�A�  B��
                                    Bxyu6�  �          @ȣ�?W
=@ƸR�#�
����B��{?W
=@�=q@.�RAΣ�B��)                                    BxyuE�  
�          @�G�?5@�  >��?�=qB�  ?5@���@:=qA�=qB�W
                                    BxyuT,  T          @�=q>�{@�G�>�G�@~�RB��f>�{@�ff@L(�A�Q�B��                                    Bxyub�  �          @��=u@���?(�@���B���=u@��@U�A��RB�Ǯ                                    Bxyuqx  T          @��>�@ə�>�Q�@O\)B��R>�@��@G
=A�=qB�aH                                    Bxyu�  T          @�\)>8Q�@�
=>�\)@$z�B���>8Q�@�ff@@  A�z�B�(�                                    Bxyu��  T          @�>��R@��>���@2�\B�(�>��R@�z�@?\)A��
B�Q�                                    Bxyu�j  �          @�
=�#�
@�{>��H@�\)B���#�
@��H@J�HA�\)B�#�                                    Bxyu�  �          @Ǯ���
@�
=>���@k�B�ff���
@��@FffA��B���                                    Bxyu��  
�          @�  ���
@���?xQ�A�HB�� ���
@��H@dz�B33B���                                    Bxyu�\  �          @�Q�aG�@\?���AVffB���aG�@�=q@~�RB�HB�(�                                    Bxyu�  "          @ȣ׾�@���?�33A)G�B�aH��@���@n�RB�B��                                    Bxyu�  �          @��þu@Ǯ?(�@���B�z�u@�33@Q�A�=qB�8R                                    Bxyu�N  
�          @ȣ׿�@�{>�@�
=B�uÿ�@��@G�A�RB�{                                    Bxyv�  
Z          @Ǯ�.{@���
=q��
=B���.{@���@p�A��\B��                                    Bxyv�  �          @�Q�>L��@�z�?��@�z�B�33>L��@���@K�A�Q�B��{                                    Bxyv!@  
�          @ə�=u@�ff?n{A	G�B��=u@�p�@c33B	�
B��3                                    Bxyv/�  
�          @�=q>�@�\)?O\)@�B���>�@�Q�@\(�B�B��                                    Bxyv>�  T          @�=q?0��@�Q�>�@�ffB�z�?0��@�@H��A�RB���                                    BxyvM2  �          @�=q?#�
@��þB�\�޸RB�aH?#�
@��@#�
A�33B�L�                                    Bxyv[�  
Z          @ȣ�?Q�@ƸR>B�\?�  B�  ?Q�@�Q�@7�A�{B��                                    Bxyvj~  �          @���?Q�@ƸR�8Q�У�B�  ?Q�@�p�@!�A���B���                                    Bxyvy$  �          @���?��
@�p���(��y��B��?��
@�  @�\A�\)B��q                                    Bxyv��  �          @�=q?��H@�(��^�R��
=B��q?��H@��?�{A��B��f                                    Bxyv�p  T          @�(�>�
=@�G�?�@�33B�L�>�
=@�{@L��A��B��                                    Bxyv�  "          @�=q?Y��@�  �8Q��
=B�p�?Y��@��R@!�A��B�                                      Bxyv��  
�          @�p�?��@��׿Q����
B�ff?��@���?�{A�Q�B���                                    Bxyv�b  
�          @��?�z�@��
��33�~�\B���?�z�@�G�?n{A�B��\                                    Bxyv�  "          @�z�?�ff@��H<#�
>�B�L�?�ff@�Q�@!G�AŅB�8R                                    Bxyv߮  
�          @�?��
@�\)>Ǯ@fffB�\)?��
@�\)@:�HA�G�B�z�                                    Bxyv�T  
�          @Å?��
@�=q��  �z�B�
=?��
@��@G�A�\)B�W
                                    Bxyv��  T          @��
?�(�@���>�33@P��B�ff?�(�@�=q@2�\A�33B���                                    Bxyw�  �          @��?�\)@���#�
�\)B��?�\)@���@\)A�\)B�=q                                    BxywF  
Z          @�p�?�@��H��=q� ��B�
=?�@��?\Adz�B��                                    Bxyw(�  T          @�\)?�z�@��\���
�>�\B�.?�z�@�=q?�=qAFffB��                                    Bxyw7�  
Z          @�ff?�{@�G��(���  B���?�{@�{>\@c�
B��q                                    BxywF8  
�          @Ǯ?���@�33�!����B���?���@�p��#�
����B��\                                    BxywT�  T          @�@G�@�=q�S33�33B��=@G�@�Q�p���Q�B�ff                                    Bxywc�  �          @�(�@ ��@��R�k����B��\@ ��@�33��
=�X��B��\                                    Bxywr*  T          @�\)@(�@}p������.(�Bt  @(�@����{��z�B��
                                    Bxyw��  
�          @ƸR?��@���#33��Q�B�B�?��@���<��
>��B���                                    Bxyw�v  
�          @�=q?��@Ǯ�z�����B�  ?��@���@
=A���B�aH                                    Bxyw�  T          @�33?�@�녾����\)B��\?�@�p�@\)A�33B��                                    Bxyw��  �          @�G�>�@�ff�5�љ�B��\>�@�p�?�(�A�z�B�(�                                    Bxyw�h  �          @�\)>8Q�@�ff�k��	��B���>8Q�@�\)@��A��
B�Q�                                    Bxyw�  �          @���ff@�z�    <#�
B�k���ff@�=q@%�A�ffB�G�                                    Bxywش  �          @�{?��@�(������z�B��q?��@�(�@(�A�z�B���                                    Bxyw�Z  �          @�
=?�@�{>��?���B�(�?�@��@.�RA�=qB�
=                                    Bxyw�   
�          @�ff?��@��>�\)@(Q�B���?��@�
=@5A�(�B��                                    Bxyx�  �          @�{?��@���=�G�?z�HB��{?��@��R@(Q�Aʏ\B��                                     BxyxL  T          @ƸR?�=q@��R�:�H��G�B�Q�?�=q@�
=?���A�z�B�#�                                    Bxyx!�  R          @�  ?���@�z�aG���
B�8R?���@�p�@
=A�33B��\                                    Bxyx0�  
�          @ȣ�?�ff@���L�;�
=B��{?�ff@��
@!G�A�z�B��q                                    Bxyx?>  
Z          @�  ?
=q@�{?��@��HB�?
=q@���@E�A�ffB��{                                    BxyxM�  �          @�Q�?#�
@ƸR>k�@�B�L�?#�
@��@1�A�B��                                    Bxyx\�  �          @�G�?��@�ff    �#�
B��?��@�z�@$z�A��HB�\                                    Bxyxk0  
�          @��=���@ə�?���A�\B�(�=���@���@g
=B	p�B���                                    Bxyxy�  "          @��=�\)@ə�?�{A (�B��R=�\)@���@g�B	�B�u�                                    Bxyx�|  T          @��>Ǯ@�33?&ff@��\B���>Ǯ@�Q�@Mp�A�G�B��H                                    Bxyx�"  T          @�p�>���@�(�?
=q@��B�{>���@��H@G�A���B�B�                                   Bxyx��  �          @��?�@ə�?\(�@��B���?�@�z�@XQ�A�p�B��H                                   Bxyx�n  �          @�(�?k�@�Q�?G�@��B�W
?k�@�(�@Q�A�  B��                                    Bxyx�  T          @��?�G�@\?޸RA|  B�(�?�G�@��\@��\BQ�B�k�                                    BxyxѺ  T          @˅?:�H@��H?���Ai�B�Q�?:�H@���@}p�BffB�
=                                    Bxyx�`  
Z          @���>�{@�33?��A��HB���>�{@���@��RB$�
B��)                                    Bxyx�  
�          @���#�
@��@p�A�z�B��)�#�
@�z�@��B:�B�(�                                    Bxyx��  
(          @�{��  @��H@(Q�A��B�녾�  @�Q�@���BA
=B��q                                    BxyyR  	�          @θR����@�\)@��A��HB������@��@�(�B7�\B���                                    Bxyy�  �          @�{�.{@\@33A�
=B�L;.{@�
=@�33B*z�B�=q                                    Bxyy)�  �          @�����@ȣ�?��
A8  B�G�����@��R@mp�B��B�                                    Bxyy8D  �          @θR�.{@ʏ\?��\A5�B�{�.{@���@n{B�B��3                                    BxyyF�  T          @Ϯ�B�\@��
?���A*{B�k��B�\@��H@j�HB

=B�{                                    BxyyU�  "          @�\)��G�@���?���AO�B�  ��G�@��@w�B  B��q                                    Bxyyd6  T          @�p�?�R@�33=u?��B��
?�R@���@'�A��B�                                    Bxyyr�  �          @�p�?xQ�@�=q�\)��G�B��{?xQ�@��H@�HA�B�.                                    Bxyy��  �          @��?:�H@˅��Q�L��B��?:�H@��@�RA�(�B���                                    Bxyy�(  �          @�z�>�@��
    ��B�ff>�@��\@#33A��RB���                                    Bxyy��  T          @�z�?��@�33>�@��\B��?��@��
@?\)A��B��
                                    Bxyy�t  �          @�p�?   @˅?
=@�Q�B��?   @�33@EA�ffB��3                                    Bxyy�  �          @ʏ\?�{@�Q�L����\)B�  ?�{@��\?�A{\)B�\)                                    Bxyy��  T          @˅?�R@��=�Q�?O\)B��
?�R@�  @&ffA��
B�                                    Bxyy�f  
�          @��?Y��@�
=�\)���
B�aH?Y��@�p�?��HA�G�B���                                    Bxyy�  �          @ȣ�?c�
@������
�=G�B�B�?c�
@��?�p�A6=qB�L�                                    Bxyy��  �          @�G�?���@�>aG�@ ��B�Q�?���@��\@*=qA�(�B�(�                                    BxyzX  �          @��?��@ƸR����z�B�\?��@�  @ffA��B��{                                    Bxyz�  	�          @���?�@��;�  ��\B��\?�@�  @p�A�
=B�{                                    Bxyz"�  �          @�=q?h��@�\)�����HB�u�?h��@���@A���B�.                                    Bxyz1J  
�          @���?p��@�z�J=q��  B���?p��@�ff?ٙ�A|  B�#�                                    Bxyz?�  �          @ȣ�?���@�  ��z��O�
B��?���@��?���A��B��f                                    BxyzN�  "          @�33?��\@�z�n{��HB���?��\@�Q�?ǮAdz�B�#�                                    Bxyz]<  	�          @�z�?��H@�{�}p��
=B���?��H@\?\A\��B��\                                    Bxyzk�  �          @�33?��@�ff�(����{B�{?��@��R?�=qA�=qB�B�                                    Bxyzz�  "          @��?��H@�����H���B��
?��H@��?��HA��B��q                                    Bxyz�.  
�          @�G�?�G�@��ͿB�\��p�B�L�?�G�@��R?��HA|��B��3                                    Bxyz��  
(          @��?\(�@ƸR�=p���  B�=q?\(�@�  ?޸RA�(�B��R                                    Bxyz�z  �          @�=q?aG�@�\)�Ǯ�`��B���?aG�@�(�@z�A��B�
=                                    Bxyz�   
�          @�=q?��
@��L�Ϳ��B��{?��
@���@�RA�
=B���                                    Bxyz��  �          @ʏ\?n{@�  �u�
=qB�#�?n{@�33@p�A���B�                                      Bxyz�l  "          @��H?�@�ff����n{B��\?�@�(�@�A���B�aH                                    Bxyz�  �          @˅?�Q�@�\)����=qB�L�?�Q�@�G�@33A�Q�B��3                                    Bxyz�  �          @��
?��@ƸR>\)?��\B��3?��@�@!�A��
B�z�                                    Bxyz�^  �          @�(�?��@���>aG�@   B�=q?��@��H@%�A�G�B�aH                                    Bxy{  "          @�ff?���@ʏ\��Q�J=qB���?���@�(�@
=A�B�                                      Bxy{�  T          @�
=?��@�(��#�
��B��?��@�z�@(�A�\)B��                                    Bxy{*P  �          @�ff?Q�@�z�>L��?��B�k�?Q�@�=q@(��A���B�                                    Bxy{8�  �          @Ϯ?fff@���>��H@��
B�?fff@�
=@:=qAՅB�#�                                    Bxy{G�  "          @�  ?�  @��
?(��@��B��?�  @�(�@C33A�(�B���                                    Bxy{VB  T          @У�?Tz�@�p�?5@ǮB�Q�?Tz�@�p�@G
=A�(�B�aH                                    Bxy{d�  
�          @���>�@Ϯ>u@�B��{>�@��@,��AÙ�B��q                                    Bxy{s�  "          @У�?xQ�@�=�\)?!G�B��?xQ�@�p�@!G�A�(�B�z�                                    Bxy{�4  T          @�Q�>�@�\)=u?   B���>�@�
=@!G�A�ffB��                                    Bxy{��  "          @�G�>�(�@�Q���Ϳ^�RB�� >�(�@��@Q�A�33B���                                    Bxy{��  T          @�=q>�  @У׼#�
��G�B���>�  @�G�@p�A�33B�Q�                                    Bxy{�&  T          @��H?(�@У׽�Q�B�\B�\)?(�@�=q@��A�
=B��\                                    Bxy{��  "          @��
>��H@ҏ\��G��q�B�p�>��H@�Q�@�A��HB�                                      Bxy{�r  
�          @�33?
=@ȣ׿�
=��ffB�=q?
=@�G�?z�@���B���                                    Bxy{�  
�          @�G����@�(���
��Q�B��;��@θR>���@dz�B���                                    Bxy{�  "          @љ���@��Ϳ���6�\B�z��@�p�?��HA*�RB�z�                                    Bxy{�d  	�          @��ÿ�z�@�  ��{�@��B�{��z�@ə�?��A�B��f                                    Bxy|
  �          @�\)�,��@��׿�Q��(��B�\�,��@���?��A
=B��H                                    Bxy|�  
�          @�Q��:=q@�p������*�RB�Q��:=q@��R?�G�A�HB�                                    Bxy|#V  
�          @У��Q�@�\)�^�R���B�k��Q�@�(�?�
=AK�B�(�                                    Bxy|1�  "          @�G���\)@�G��333��z�B��ÿ�\)@�33?ٙ�Ap��B��
                                    Bxy|@�  �          @Ϯ����@��ÿW
=��B�G�����@�z�?ǮA^�RB���                                    Bxy|OH  �          @��ÿ��@�
=�����33Bъ=���@�{?�G�A2=qBѨ�                                    Bxy|]�  �          @�Q����@�\)��  ��B������@���?���AC�B�#�                                    Bxy|l�  �          @����ff@������:�HB�p��ff@�
=?uA��B�                                      Bxy|{:  
�          @љ��xQ�@���=�G�?s33B�=q�xQ�@��@{A���Bƣ�                                    Bxy|��  
�          @�=q�c�
@�
=����G�BÙ��c�
@Å@	��A�G�BĀ                                     Bxy|��  
�          @��=�Q�@�\)�\�X��B�k�=�Q�@��@�A��B�W
                                    Bxy|�,  "          @љ�=��
@љ�=u?\)B���=��
@�=q@p�A��HB��\                                    Bxy|��  �          @�=q=�G�@��>��?���B�
==�G�@���@#33A���B��H                                    Bxy|�x  �          @�=q?��@��ýu�\)B�k�?��@�33@�A��RB��q                                    Bxy|�  �          @�33>�{@�녾�p��L(�B�\>�{@Ǯ@�
A��B�                                    Bxy|��  
�          @�33>\@�녿�����B�k�>\@ə�?�z�A���B�#�                                    Bxy|�j  "          @�=q>Ǯ@У׿����Q�B�{>Ǯ@ə�?���A�ffB���                                    Bxy|�  �          @�33=���@�녿�����\B�L�=���@�=q?�\)A��
B�=q                                    Bxy}�  �          @��=u@�G��L�Ϳ��
B���=u@�p�@�A�  B��f                                    Bxy}\  �          @��aG�@��
    �#�
B���aG�@�p�@��A�  B�.                                    Bxy}+  �          @���{@���.{���RB�녾�{@ȣ�@  A�p�B�L�                                    Bxy}9�  T          @�p���@�z�>#�
?��B�B���@�z�@#33A��
B��                                    Bxy}HN  
Z          @���G�@���>L��?�Q�B��\��G�@�(�@%�A�{B�.                                    Bxy}V�  "          @���\)@��=#�
>ǮB�8R��\)@�{@�A��B�L�                                    Bxy}e�  �          @��;B�\@љ�����z�B�Q�B�\@Ϯ?���A?�
B�\)                                    Bxy}t@  
�          @����
@�z�=�?��B��=���
@��@\)A�=qB���                                    Bxy}��  �          @׮����@��?L��@��
B��)����@�{@HQ�A�=qB��3                                    Bxy}��  �          @ָR>��@�33?��A�B�ff>��@���@W�A�=qB�                                    Bxy}�2  "          @��?�R@��?��Ad��B�  ?�R@���@tz�B=qB���                                    Bxy}��  "          @��?.{@˅?�=qA�B���?.{@�G�@~{B�B�aH                                    Bxy}�~  
�          @���<��
@�{?��Ad  B���<��
@�{@tz�B(�B��\                                    Bxy}�$  �          @�{<�@�ff?�  Ar�RB�z�<�@��@z�HB�B�aH                                    Bxy}��  T          @ָR�8Q�@�G�?��HAH��B�#׾8Q�@�33@j�HB  B���                                    Bxy}�p  �          @�ff�k�@Ӆ?���A
=B���k�@���@U�A���B��                                    Bxy}�  �          @�{��z�@���?&ff@��B�녾�z�@��@<��A�G�B�p�                                    Bxy~�  �          @��ͽu@��
?\)@��
B�{�u@�  @7
=AˮB�33                                    Bxy~b  T          @�p����@��>��R@(��B������@�(�@(Q�A�p�B���                                    Bxy~$  �          @��    @�z�>�@}p�B�      @��@0��AîB�                                      Bxy~2�  �          @��ͼ#�
@�(�>�?�33B�.�#�
@�p�@(�A���B�33                                    Bxy~AT  T          @�����
@��
>�p�@J�HB�����
@\@)��A��\B�                                      Bxy~O�  
�          @�>�  @�z�?�@�G�B��>�  @�G�@3�
A�G�B�B�                                    Bxy~^�  T          @�ff>aG�@�p�?
=@���B�.>aG�@���@7�A���B�Ǯ                                    Bxy~mF  �          @�ff>�  @�>\@Q�B���>�  @�(�@+�A��\B�k�                                    Bxy~{�  �          @�{=�\)@�>�Q�@E�B��R=�\)@�z�@)��A���B���                                    Bxy~��  �          @׮?0��@�33?�ffA�B��?0��@��\@P��A��B��=                                    Bxy~�8  "          @��H���@�G��!G�����B�33���@˅?�AaB�{                                    Bxy~��  �          @ٙ���  @�z�>u@z�B�33��  @���@ ��A�BƊ=                                    Bxy~��  T          @ָR?Tz�@Ϯ?�
=AEG�B�u�?Tz�@�33@dz�B  B�33                                    Bxy~�*  �          @�ff?�Q�@��?�G�AP��B���?�Q�@�  @g�B��B��{                                    Bxy~��  "          @���?p��@�\)?�A!B��=?p��@�{@S�
A�33B�G�                                    Bxy~�v  �          @���?.{@љ�?c�
@�\)B�{?.{@�33@E�AܸRB���                                    Bxy~�  T          @�ff?\)@Ӆ?p��A�B�\)?\)@�z�@H��A�p�B�.                                    Bxy~��  
�          @ָR>�=q@��
?��A33B�L�>�=q@��
@N�RA��B��3                                    Bxyh  �          @�
=?&ff@Ӆ?p��A��B���?&ff@���@HQ�A��HB�u�                                    Bxy  �          @ָR?�{@�=q?=p�@˅B��)?�{@�@:�HA�
=B��
                                    Bxy+�  T          @�
=?fff@��H?aG�@��B��?fff@��@C�
A���B��q                                    Bxy:Z  
�          @׮?�(�@�@(�A�ffB�L�?�(�@�=q@���B��B�Ǯ                                    BxyI   �          @�G�?�ff@�G�@!G�A�ffB�Ǯ?�ff@��H@��B"
=B�#�                                    BxyW�  
�          @�=q?��H@�G�@�A���B���?��H@�{@��B�RB��                                    BxyfL  
�          @�=q?���@�ff?�{A}�B�.?���@�ff@z�HB��B���                                    Bxyt�  �          @�33?\@�{?�\Ao�B�u�?\@�\)@u�B�
B�                                      Bxy��  
�          @�(�?aG�@��
?�=qATz�B��
?aG�@��R@l��B�B�u�                                    Bxy�>  
�          @�p�?���@�p�?���AA�B�=q?���@���@fffA���B��                                     Bxy��  "          @���?���@�p�?�\)A733B�?���@��H@`��A�
=B�33                                    Bxy��  T          @�(�?���@���?�G�A(��B��?���@�33@Y��A�
=B���                                    Bxy�0  �          @��?�=q@Ӆ?��
A*{B�{?�=q@��@Y��A�\)B��=                                    Bxy��  "          @޸R?���@�ff?��\A\)B��\?���@�\)@K�A��B��                                    Bxy�|  T          @�?���@���?���A�RB���?���@��@O\)A�
=B��R                                    Bxy�"  �          @�ff?�ff@�=q?�{A4  B�L�?�ff@�Q�@\��A�  B�33                                    Bxy��  T          @�?�33@љ�?�G�A'\)B���?�33@���@VffA�RB�z�                                    Bxy�n  �          @��?��@�33?333@�=qB�8R?��@�Q�@5�A�B�=q                                    Bxy�  �          @أ�?ٙ�@��@  A�{B���?ٙ�@�G�@���B  B��                                    Bxy�$�  
�          @���?�33@e@��BX{B�aH?�33?�z�@�{B�aHBG{                                    Bxy�3`  
�          @�?�p�@i��@�G�BRz�B�(�?�p�?�(�@�{B��fB3=q                                    Bxy�B  T          @ٙ�?\@�{@q�B�B���?\@z=q@��HBOz�B��=                                    Bxy�P�  T          @�z�?�33@�  @�(�B�B��?�33@g�@��BZ��B�                                      Bxy�_R  	�          @���?�{@�ff@�G�B$G�B���?�{@Mp�@���Bj�B}z�                                    Bxy�m�  "          @�33?��@�
=@n�RB{B��{?��@~{@���BL��B���                                    Bxy�|�  
�          @�  @�R@��@i��B��Bz�@�R@g
=@��HBE{B]��                                    Bxy��D  �          @�  ?�@��H@y��B�B��?�@c�
@��HBSQ�Bv�                                    Bxy���  T          @�p�?p��@�z�@��B>33B��)?p��@%�@��B�Q�B��                                    Bxy���  T          @�p�>B�\@��@���B-�B�z�>B�\@Fff@��Bxp�B���                                    Bxy��6  �          @ָR?.{@�(�@�
=B4�
B�G�?.{@8Q�@��RB~ffB�\)                                    Bxy���  �          @�
=?��@�33@S33A�RB�{?��@�  @�p�B?ffB���                                    Bxy�Ԃ  
Z          @�@�@���@#33A�B���@�@���@�Q�B��B���                                    Bxy��(  
(          @ٙ�?�z�@�33@
�HA��B�B�?�z�@��H@�  BffB��f                                    Bxy���  �          @���@  @��?���AW�B���@  @�
=@a�A�z�B�aH                                    Bxy� t             @�G�?��R@�Q�?��AQ�B�.?��R@��H@Dz�A�(�B�Q�                                    Bxy�  	�          @޸R?�
=@��
?���AT  B��=?�
=@���@g
=A�  B��                                    Bxy��  �          @�\)?�ff@�p�?��\A((�B�Ǯ?�ff@�p�@S�
A�G�B���                                    Bxy�,f  
Z          @�33?�33@�\)?�A(�B�p�?�33@���@O\)A��B��                                    Bxy�;  
Z          @�?�ff@�ff?�  AC\)B�  ?�ff@�z�@a�A���B�{                                    Bxy�I�  �          @�G�?��
@��
?�ArffB���?��
@��R@u�B�RB�                                      Bxy�XX  !          @�\?�{@�\)?�z�AY�B�(�?�{@��
@l(�A��B��                                    Bxy�f�  	`          @��
?�(�@��?�(�A>�HB���?�(�@�Q�@a�A��B�u�                                    Bxy�u�  "          @��?��@�33?���AO
=B�k�?��@�Q�@j=qA�(�B�z�                                    Bxy��J  "          @�R?���@�z�?p��@�\)B���?���@�  @B�\A�B�{                                    Bxy���  
�          @�=q@�@��?�p�A   B��@�@�ff@O\)A�{B�W
                                    Bxy���  �          @��
?�@أ�?���A�
B��\?�@��@P  A�
=B�33                                    Bxy��<  "          @���?�(�@�Q�?�p�A!G�B�u�?�(�@���@P��A�\)B��                                    Bxy���  
(          @�?�p�@�?���A3�B�\?�p�@�{@VffA�33B��                                     Bxy�͈  
(          @�?�z�@�z�?��A2ffB�\?�z�@���@Tz�A�B�.                                    Bxy��.  �          @���?�z�@θR?�(�A%G�B�.?�z�@���@H��A��B��                                    Bxy���  �          @׮?�p�@��
?�=qA6�RB��\?�p�@���@N{A��HB��                                    Bxy��z  T          @�\)?��@˅?�A�B�
=?��@�ff@C�
A�(�B�p�                                    Bxy�   �          @�  ?�
=@��
?�{A�B�L�?�
=@�
=@@��Aә�B��q                                    Bxy��  �          @�  ?�@�33?�A Q�B�W
?�@�{@C33A׮B��                                    Bxy�%l  T          @�{?��R@��?xQ�A{B�?��R@��R@6ffA��
B���                                    Bxy�4  "          @�{@@�G�?^�R@�\)B�B�@@�\)@/\)A���B��                                    Bxy�B�  �          @�ff@ ��@�=q?�  A��B��3@ ��@�
=@7�A�z�B�B�                                    Bxy�Q^  
(          @�z�?��@�G�?n{A ��B��3?��@�
=@2�\A�
=B��=                                    Bxy�`  T          @�33?�
=@ȣ�?8Q�@�Q�B���?�
=@�Q�@%�A�{B���                                    Bxy�n�  
�          @�z�@�\@Ǯ?h��@�z�B�Ǯ@�\@�@0  A��B�p�                                    Bxy�}P  
�          @��@   @�Q�?��
A�RB��=@   @��@7
=A�(�B�
=                                    Bxy���  T          @���?�@�Q�?��A333B���?�@��\@G
=A�\)B�(�                                    Bxy���  
�          @�?���@�
=?Y��@�\B��?���@�p�@0  A¸RB��q                                    Bxy��B  �          @ָR?�G�@�  ?���A\)B�\?�G�@��
@?\)A�  B�Ǯ                                    Bxy���  "          @��
?�@�33?�G�A)p�B��3?�@�p�@J�HA�(�B�                                    Bxy�Ǝ  
�          @׮?��@�?^�R@�
=B�?��@�(�@/\)A�(�B���                                    Bxy��4  �          @�G�?�\)@�  ?�\)A:{B�8R?�\)@���@O\)A�  B�p�                                    Bxy���  T          @�33?���@�Q�?�
=ABffB�{?���@���@S33A�{B�W
                                    Bxy��  T          @��?5@Ǯ@z�A�Q�B�\?5@���@��Bp�B��3                                    Bxy�&  �          @�  >�p�@�  @��HB�B��=>�p�@dz�@�z�BaQ�B�(�                                    Bxy��  
�          @Ϯ?��@�ff@^{B=qB��=?��@�{@�p�BE=qB���                                    Bxy�r  �          @�33?�@��
@q�B�B���?�@���@�{BP
=B�Q�                                    Bxy�-  T          @�33?Tz�@��
@n�RB	�B�u�?Tz�@�G�@�z�BM�B���                                    Bxy�;�  
�          @��?
=@�Q�@j�HB�B��?
=@�{@��
BI�B��)                                    Bxy�Jd  
�          @ҏ\>�G�@�=q@�p�B)�RB�p�>�G�@Tz�@�z�Bm=qB��R                                    Bxy�Y
  �          @�
=��@�(�@z=qB=qB���@qG�@�\)BX��B�                                    Bxy�g�  T          @�����@�G��.{�\B�����@��\?���Ah��B���                                    Bxy�vV  �          @�33@(��@�\)@J=qA�=qBsQ�@(��@g
=@�z�B2�BW=q                                    Bxy���  
�          @�z�?�p�@�z�@�HA�(�B���?�p�@�ff@|��B��B��                                    Bxy���  �          @θR?�\)@���?��A�G�B�ff?�\)@�Q�@]p�B��B�Ǯ                                    Bxy��H  T          @�>\@�ff?�{Ahz�B��>\@�
=@U�A�  B���                                    Bxy���  T          @����@�z�>W
=?���B�G���@��@�\A�z�B�ff                                    Bxy���  
�          @�G�>�=q@�
=@ ��A�ffB���>�=q@��@c�
B�HB��R                                    Bxy��:  "          @ƸR?\(�@\)@�Q�B}��B�  ?\(�?aG�@�  B��fB6�
                                    Bxy���  
�          @�?���>���@�=qB�u�AG�?��ÿ��
@�{B�L�C�xR                                    Bxy��  �          @�(�?�=q�u@�p�B��qC��=?�=q�޸R@��B��3C��                                     Bxy��,  "          @�?:�H?^�R@�{B�ffBG�?:�H�\(�@�{B�u�C�.                                    Bxy��  �          @ʏ\?�Q�@=q@��\Bx��B[�
?�Q�?:�H@�G�B�\A�Q�                                    Bxy�x  
�          @˅?�{@@  @�Q�B`�\Bhz�?�{?���@�z�B�BQ�                                    Bxy�&  �          @Ǯ@G�@&ff@�(�B`Q�BC��@G�?��@���B�aHAř�                                    Bxy�4�  T          @�z�@   @J=q@���BJp�BN�\@   ?��@��HBwz�Bff                                    Bxy�Cj  T          @�  ?���@(�@�=qB�
B�\?���?G�@���B�aHB�R                                    Bxy�R  T          @�
=>.{?+�@ÅB�{B�>.{����@���B���C��{                                    Bxy�`�  T          @�\)>�?5@�z�B�ffBc�
>����@��HB�ǮC�ٚ                                    Bxy�o\  "          @�(�?�33@\)@��
B|�Brff?�33?Q�@�33B�#�A�{                                    Bxy�~  
�          @Ϯ@{@��@�=qB+  Bv�@{@/\)@�G�BaQ�BL                                      Bxy���  
�          @�=q@+�@��R@mp�B	��Bqp�@+�@_\)@��
B?BQ�                                    Bxy��N  
�          @�p�@8Q�@�
=@X��A���Bop�@8Q�@u�@�z�B0Q�BT                                      Bxy���  "          @�ff@U�@�z�@@  A�G�B^�
@U�@w
=@��B�BD�
                                    Bxy���  T          @�ff@B�\@���@fffBBe��@B�\@e@���B5BF�H                                    Bxy��@  "          @��H@L��@��@P��A��Be\)@L��@x��@�Q�B'ffBJff                                    Bxy���  T          @��@AG�@�\)@0��A��RBx�@AG�@�  @�\)B�
Be��                                    Bxy��  "          @�=q@*=q@�
=?�Q�A�B�Ǯ@*=q@�ff@e�A�=qB�                                    Bxy��2  
(          @�Q�@J=q@\?�A;
=Bz(�@J=q@��R@A�AͅBo�R                                    Bxy��  �          @޸R@>{@�?xQ�A z�B�@>{@�@(Q�A�(�By                                    Bxy�~  T          @�  @Dz�@�{?B�\@�  B~p�@Dz�@�  @�HA�
=Bw�                                    Bxy�$  
�          @�G�@Mp�@\>�
=@`��Bx�@Mp�@�\)@�
A��Bs{                                    Bxy�-�  �          @߮@L��@��H@(Q�A�p�Bp�@L��@��@���BffB^=q                                    Bxy�<p  
�          @�G�@e@�=q?�\)AUBiff@e@��@I��Aԏ\B\                                    Bxy�K  "          @��@H��@�  >�G�@e�B}(�@H��@���@�A�z�Bw�R                                    Bxy�Y�  "          @�\)@J=q@ƸR=�Q�?E�B|  @J=q@�ff?�\Ak\)Bx
=                                    Bxy�hb  T          @�  @Tz�@�z���ͿTz�Bv�@Tz�@�{?ǮAN�RBsG�                                    Bxy�w  �          @�\@/\)@�  ����z�B�W
@/\)@�p�?���AQ�B���                                    Bxy���  T          @�33@'
=@�=q�z����RB���@'
=@�\)?��HAz�B�#�                                    Bxy��T  �          @�=q@�@Ӆ�������B���@�@��?=p�@��HB�                                    Bxy���  �          @߮@p�@У׾��p  B���@p�@���?��A,Q�B���                                    Bxy���  �          @��@*�H@�
=��{�3�
B�.@*�H@ʏ\?�33A8  B�B�                                    Bxy��F  "          @���@5�@�p�����
=qB�� @5�@�Q�?��HA?�B�ff                                    Bxy���  �          @ᙚ@/\)@Ϯ��\)�
=qB�B�@/\)@���?�z�AZffB��)                                    Bxy�ݒ  �          @�=q@.{@У׾W
=�޸RB��@.{@��H?\AG33B��=                                    Bxy��8  �          @�G�@3�
@�{��\)�\)B��
@3�
@���?�Q�A=�B���                                    Bxy���  
�          @��@=q@�논��
�.{B��{@=q@ʏ\?��HAc
=B�=q                                    Bxy�	�  �          @߮@!�@�Q�>.{?�33B���@!�@Ǯ?��Az{B��f                                    Bxy�*  �          @߮@E�@�Q�=L��>ǮB=q@E�@���?�Q�A`��B{                                    Bxy�&�  �          @�Q�@.�R@�{=��
?0��B�{@.�R@�{?�\Aj�HB�p�                                    Bxy�5v  �          @�G�@S33@�{���H�}p�Bw�H@S33@�33?�33A33Bvz�                                    Bxy�D  �          @��H@g�@����8Q���G�Blff@g�@���?c�
@�\)Bl                                      Bxy�R�  �          @�=q@c33@��H��ff�h��Bo{@c33@�  ?�z�A�RBm�                                    Bxy�ah  T          @�Q�@o\)@��H�k���\)Be��@o\)@�ff?�ffA-G�Bc
=                                    Bxy�p  "          @�  @QG�@��#�
��Q�Bx��@QG�@�
=?���AS
=BuQ�                                    Bxy�~�  "          @޸R@3�
@�33�u��p�B�W
@3�
@�ff?�A<  B�G�                                    Bxy��Z  �          @޸R@@��þ.{��B��=@@�33?\AK
=B��                                    Bxy��   �          @�
=?���@�33>��H@��\B�.?���@Ϯ@  A��HB�33                                    Bxy���  
�          @��@A�@˅>L��?���B�
=@A�@��H?�=qAp��B~=q                                    Bxy��L  �          @߮@Q�@�z�?   @�33Bw��@Q�@��@�
A�p�Br33                                    Bxy���  �          @�G�@Mp�@�ff?8Q�@���Bzp�@Mp�@��@�\A�=qBt(�                                    Bxy�֘  �          @�G�@P  @�
=>���@,��By��@P  @�?�z�A|��Bu33                                    Bxy��>  �          @�\)@P��@��>.{?��BxG�@P��@��?޸RAf�HBt\)                                    Bxy���  
(          @߮@e�@�\)�u��Bl\)@e�@���?�(�AB�RBi33                                    Bxy��  �          @�
=@Fff@�\)>W
=?޸RB~(�@Fff@�
=?��AnffBz33                                    Bxy�0  �          @�@Tz�@���>�ff@n{Bt��@Tz�@��?�(�A�z�Bo                                    Bxy��  T          @�@Fff@��>��@y��B}
=@Fff@��H@G�A��Bx
=                                    Bxy�.|  �          @߮@Fff@�ff?�@���B}��@Fff@��@�A���Bx��                                    Bxy�="  T          @ᙚ@u�@��R?�ffAJ�RB`�\@u�@�(�@<(�A�
=BU{                                    Bxy�K�  
�          @�\)@q�@��R?��A-p�Bb{@q�@�ff@-p�A��HBW��                                    Bxy�Zn  �          @�@J�H@�=q?L��@�p�By��@J�H@�p�@33A�  Bs(�                                    Bxy�i  T          @�@<��@��R?�{A�B~z�@<��@��@$z�A�33Bv��                                    Bxy�w�  �          @�  @P��@���@_\)A�Bd{@P��@�  @�z�B'�
BK
=                                    Bxy��`  �          @�ff@Dz�@\?�G�A'�
B|�@Dz�@�=q@/\)A�ffBtp�                                    Bxy��  T          @�  @6ff@ə�?.{@��
B�\)@6ff@�@�RA��
B��R                                    Bxy���  T          @��H@A�@�  >\@P��B}
=@A�@�
=?�\)A���Bxz�                                    Bxy��R  �          @�33@
=q@�(�?��RA�Q�B�8R@
=q@�{@\��A�(�B�                                    Bxy���  �          @��@P  @���?�\)A��Bn�@P  @��@L(�A�G�Bb                                      Bxy�Ϟ  "          @ۅ@h��@�p�?   @���Be(�@h��@��
?��A�  B_p�                                    Bxy��D  �          @ڏ\@B�\@�@�\A�33Br��@B�\@�  @Tz�A�p�Bep�                                    Bxy���  �          @���@]p�@�(�>L��?��HBi��@]p�@��?�=qA\��Bez�                                    Bxy���  �          @У�@\(�@�(�@>�RA�G�BOp�@\(�@`��@~{B��B6ff                                    Bxy�
6  �          @��@:�H@K�?�(�A�\)B=�@:�H@*=q@(�A�B)�                                    Bxy��  �          @��
@(�@2�\@�z�BR�\BOp�@(�?Ǯ@�Bz=qBG�                                    Bxy�'�  �          @���?���@6ff@�p�B]z�Bc�R?���?Ǯ@�
=B�33B =q                                    Bxy�6(  �          @�
=@Q�@x��@|(�B&�Bu\)@Q�@3�
@���BV�BS��                                    Bxy�D�  �          @�(�@�R@��@z�A��ByG�@�R@���@Y��B�HBi\)                                    Bxy�St  �          @�33@z�@��?�33A��B���@z�@�{@HQ�A�33B~�                                    Bxy�b  �          @�?�\)@��?У�Aw\)B�#�?�\)@�G�@<��A�B�W
                                    Bxy�p�  �          @��
?�33@�ff?�\A�z�B���?�33@��@B�\A�B�aH                                    Bxy�f  �          @�@��@�p�?B�\@��B��\@��@�=q@
=A��B�{                                    Bxy��  �          @�
=@j=q@�ff�xQ��(�BJ\)@j=q@��>\)?�=qBL�H                                    Bxy���  �          @��\@�
=@u������/
=B){@�
=@\)�������B-�                                    Bxy��X  "          @��H@�{@S33����33Bp�@�{@h�ÿxQ���BQ�                                    Bxy���  �          @�p�@��H@s�
����3
=B%{@��H@~{�L�Ϳ�z�B)�
                                    Bxy�Ȥ  �          @ʏ\@��R@��\����&�\B0��@��R@�\)��\)�+�B4p�                                    Bxy��J  �          @ʏ\@�33@�p���\)�#�BB�@�33@���<�>�  BF�                                    Bxy���  
�          @ə�@vff@�33���\���BNQ�@vff@��R>#�
?�G�BP�                                    Bxy���  �          @��@��@�z�>�?��B-��@��@\)?�{A*�HB)�R                                    Bxy�<  T          @��
@}p�@��R?��\A?�
BA@}p�@���@�\A�z�B6=q                                    Bxy��  �          @��@z�@��?���A��B��@z�@�=q@��A��B��)                                    Bxy� �  �          @��@��@��?�\A�\)B��\@��@��@>{A�(�B}
=                                    Bxy�/.  T          @�{@0��@�G�>�  @�By�
@0��@��\?��RAf�RBv                                      Bxy�=�  �          @���?��R@�p���z��)��B��
?��R@�=q?�{A$(�B�B�                                    Bxy�Lz  
�          @ƸR?�
=@��>�@�{B��f?�
=@���?��A�(�B��)                                    Bxy�[   T          @��
?}p�@�Q�����
=B�33?}p�@�(�?�G�A?33B���                                    Bxy�i�  �          @�
=?�{@�Q��Q�����B�B�?�{@����
=q����B�L�                                    Bxy�xl  �          @ə�?˅@�G��h���B��\?˅@��>�33@fffB���                                    Bxy��  T          @�33@Vff@�ff@#33A���BY�@Vff@~{@dz�B

=BG\)                                    Bxy���  T          @ə�@N�R@��
@G�A�Ba@N�R@��R@UB ��BQ��                                    Bxy��^  �          @��@C33@����(���
=BW�@C33@�ff�333����B^G�                                    Bxy��  T          @���@l��@�H�U����B�R@l��@HQ��+���B                                     Bxy���  �          @�\)@	��?������H�}p�A�{@	��@{��{�\��B7                                    Bxy��P  �          @���!녿Y������o�CF���!�>u���v33C.��                                    Bxy���  T          @��R�)���(������h��CA���)��>\��{�k�\C+��                                    Bxy��  �          @�Q��  >.{��  � C/���  ?�\)��=q�z(�C��                                    Bxy��B  T          @��H�ٙ�?+����(�CaH�ٙ�?���{�~�HC�\                                    Bxy�
�  �          @�G��33?n{��(��C���33@�
��Q��w{C�\                                    Bxy��  �          @��
��?z���{� C"����?�\)���8RC�)                                    Bxy�(4  T          @ə����>����\)�{C0aH���?��H��G��{
=C8R                                    Bxy�6�  �          @�\)�l(��333��(��e��C>���l(�?(����(��e�
C)��                                    Bxy�E�  T          @߮�\)��z����
�V  CD&f�\)>����\)�\�C1�=                                    Bxy�T&  T          @ٙ���33�����{�Gp�CH�H��33���
����S
=C8�                                     Bxy�b�  T          @أ����R�\��=q�BG�CGǮ���R���
�����MffC8h�                                    Bxy�qr  �          @��������H��(��CffCG����������\�M�
C7��                                    Bxy��  �          @�\)��=q��
=���R�=��CFY���=q��\)�����GC7��                                    Bxy���  �          @�����p��������;\)CD����p��L������D\)C6��                                    Bxy��d  �          @������R��\)��33�2CD(����R�u��G��;p�C6�H                                    Bxy��
  �          @�G������z���33�6z�CB+�����#�
����=33C4��                                    Bxy���  �          @�\)�S33���
��\)�o�RC9�{�S33?xQ�����j�RC#��                                    Bxy��V  T          @����>����Hu�C0�{��?�  ���p�C\)                                    Bxy���  �          @�  ����\)����\C:���?�������p�C�3                                    Bxy��  �          @��H�N�R�k���ff�r(�CC��N�R>�(���Q��v  C,k�                                    Bxy��H  �          @˅�&ff�5�����fCCY��&ff?
=����� C'33                                    Bxy��  �          @�����\?����Hu�C&���\?��H���H�r�RCO\                                    Bxy��  
�          @�=q���?�����(��Cٚ���@���Q��g�RCc�                                    Bxy�!:  T          @�����\)?����H33C$c׿�\)?��w��k(�C�)                                    Bxy�/�  �          @�����H?��vff\C !H���H?�\)�h���tC��                                    Bxy�>�  �          @��H���R?#�
��G���CǮ���R?�����G��wG�CxR                                    Bxy�M,  T          @�ff�˅>����z��C%� �˅?�\)��{�}z�CG�                                    Bxy�[�  �          @��\��p�?!G�������C�ÿ�p�?�\)�����{C��                                    Bxy�jx  �          @�G��˅>�p�����G�C&�׿˅?�{���\u�C��                                    Bxy�y  T          @�����
>����(�ǮC'����
?�{���w\)C��                                    Bxy���  �          @�Q��33>����p��qC.���33?������8RC0�                                    Bxy��j  �          @��Ϳ�Q�=��
���\G�C18R��Q�?��
��ff��C��                                    Bxy��  S          @��H��33=������H  C0Ϳ�33?}p��~{33C޸                                    Bxy���  �          @�ff���������C@�q��?�\��k�C&@                                     Bxy��\  �          @�=q�1녿�G����H�a\)CLc��1녾k���  �n=qC8                                    Bxy��  T          @�Q��>�R���������o��CLs3�>�R��Q�����{p�C5�                                    Bxy�ߨ  �          @����E�   ��z��|{C=0��E?Y����33�yQ�C$��                                    Bxy��N  T          @����B�\��z����H�w�CH�{�B�\>B�\��{�=qC0k�                                    Bxy���  �          @�(��L�Ϳ�����z��k\)CN
�L�;�  ���H�x�
C8�=                                    Bxy��  �          @��
�>�R���������t�
CM���>�R��G���{��C6�                                    Bxy�@  �          @��<�Ϳ����=q�u  COxR�<�;B�\��Q�aHC7�                                    Bxy�(�  
�          @�ff�?\)��\)����r�CPn�?\)��\)��Q��qC9J=                                    Bxy�7�  �          @�(��3�
�ٙ���G��uCS��3�
��Q���Q�B�C;G�                                    Bxy�F2  �          @�33�&ff������RCP���&ff�u�ʏ\\)C5^�                                    Bxy�T�  �          @Ϯ��R��Q�����CT�3��R�\)��33�)C7��                                    Bxy�c~  �          @���
=���R��p�G�CY���
=�B�\�ÅǮC9��                                    Bxy�r$  �          @ʏ\��
=�!���\)�|(�CuͿ�
=��ff��z��Cc��                                    Bxy���  T          @�=q���QG���ff�M33Cvuÿ��\)�����x��Cm�
                                    Bxy��p  �          @ȣ��(�� �������gQ�Cd���(���=q��
=�qCS5�                                    Bxy��  �          @�G��	���:�H��(��[  Ci���	����G���z��
=C[c�                                    Bxy���  �          @�=q�ff�)�����H�f�Cg���ff��������Q�CV��                                    Bxy��b  �          @�=q��\)�G�����R  Co!H��\)�33�����y��Cc��                                    Bxy��  �          @��ÿ�
=�{������7(�Cz  ��
=�;���G��dQ�Cs��                                    Bxy�خ  �          @�=q��{��  ���H�6��C{8R��{�?\)���H�dz�Cu��                                    Bxy��T  
�          @����
=�����|���   Cuÿ�
=�g���=q�N�HC{��                                    Bxy���  �          @���������
�c33�p�C~�쿨����Q�����;(�C{�                                    Bxy��  T          @��H����{�W���Cw�q���w������2�Cs�f                                    Bxy�F  �          @�\)��{�����N{���C|W
��{��Q����R�*�CyQ�                                    Bxy�!�  �          @��
��ff��Q��J�H��{Cff��ff��\)����,C|�)                                    Bxy�0�  �          @�G���G���p��ff���RC��)��G�����\(���C�{                                    Bxy�?8  T          @�=q�=q��\)�����y
=CY�=�=q�(���=qffCB:�                                    Bxy�M�  �          @�ff����\)���{C\������\���\)CA�                                    Bxy�\�  �          @�����
�H��(��33C_����׿J=q��{�\CG@                                     Bxy�k*  �          @�33��ff��
��{(�Ch
��ff�h�������fCN��                                    Bxy�y�  "          @�=q���>{��(��qG�CnE���У����
L�C]�                                    Bxy��v  
(          @�=q�{�G
=��ff�e(�Cj���{��ff��
=Q�C[(�                                    Bxy��  T          @��H�*�H�P  ��
=�V�Cf�f�*�H���R�ȣ��x33CX�                                     Bxy���  
�          @��H�8���c33����G
=Cf�=�8��������i(�C[�                                    Bxy��h  �          @�=q�33�u������O�Cq�)�33�&ff��
=�w��Cg��                                    Bxy��  "          @��H�(���p�����k�
C_�(�ÿ�z����
W
CK�
                                    Bxy�Ѵ  �          @�p�����x������\{C{������%����33Cs�                                     Bxy��Z  "          @�R�޸R�g�����`�CtQ�޸R��
�ӅCh�                                    Bxy��   T          @�ff��ff�l����\)�`�Cw0���ff�Q����
z�Cl�)                                    Bxy���  T          @�
=�����q���
=�_p�Cy
=�����{��(�B�Co��                                    Bxy�L  T          @������c33����i�Cx�3���������Q��Cm޸                                    Bxy��  T          @���
=�W���{�j��Cs����
=�G���  u�CfW
                                    Bxy�)�  T          @�\)��G��aG���z��h(�Cv�ÿ�G�����\)�Ck:�                                    Bxy�8>  
(          @�׿���b�\��{�j�CyG�����������u�Cn��                                    Bxy�F�  "          @陚���R�I������u�RCt�׿��R��\��p��Ce�)                                    Bxy�U�  �          @�=q���Tz����
�q�Cvٚ����������=Ci�f                                    Bxy�d0  �          @�\�����<����=q�~�Cuuÿ��Ϳ�ff����k�Cd�3                                    Bxy�r�  �          @�녿����Mp���p��u�RCv�ÿ��׿�=q��{k�Ch�3                                    Bxy��|  "          @���33�[������r33C{k���33�33��
=�qCp��                                    Bxy��"  �          @���ff�0  ��  �)Cy���ff��������\Cgk�                                    Bxy���  �          @��
�.{�H����(�\)C��Ϳ.{��(���(��Cx�\                                    Bxy��n  T          @�z��R�XQ���G��y33C�Ǯ��R��(����Hp�C|�                                    Bxy��  "          @�z�
=�W���G��y�C��
=��(����H��C}p�                                    Bxy�ʺ  "          @��Tz��E��(�p�C  �Tz��
=���
z�Cs�                                    Bxy��`  T          @�z�(��J=q�Ӆ\C���(���  ���qCz�                                     Bxy��  T          @�p��ٙ��9�����H�|  Co�H�ٙ���  ��G�Q�C]}q                                    Bxy���  T          @�zῧ��W
=��{�s
=Cx�������p��߮ffClxR                                    Bxy�R  
�          @�{���H�0����{Q�Cn33���H�����CZ33                                    Bxy��  �          @�R�޸R�0  ��{(�Cm���޸R������3CY�)                                    Bxy�"�  T          @�Q쾙����G���G��d�C��R�����,(���\)��C��                                    Bxy�1D  
�          @�G���R�n�R�θR�n��C�C׿�R�ff��\{C.                                    Bxy�?�  "          @�Q�    ��
=��p��^\)C���    �8����z�p�C���                                    Bxy�N�  �          @�\���w
=��ff�k�\C�uþ��\)��33�qC���                                    Bxy�]6  "          @�\�B�\�R�\��G��~{C�|)�B�\��{���=qCw��                                    Bxy�k�  �          @�\�}p��8����{� C{Ϳ}p���
=��(�k�CkW
                                    Bxy�z�  �          @�=q��\)�7
=���
k�CtxR��\)��
=����qCb0�                                    Bxy��(  T          @�33��
=�p���Q��Cd����
=�B�\���L�CI��                                    Bxy���  �          @���G��z�����{Ch�
��G��\(���338RCN�                                    Bxy��t  �          @�=q���\�'
=�߮W
Cx�)���\��33���
aHCd}q                                    Bxy��  �          @��H�k��$z���=qCzJ=�k�������{�Cf(�                                    Bxy���  
Z          @���(�������CpO\��(��(����
=�CP}q                                    Bxy��f  �          @�=q����"�\��k�C�����������
=£.C��                                    Bxy��  
�          @�=q�����\)���
��C�uþ��Ϳ��\��\)£W
Cx��                                    Bxy��  �          @�녿E��Q���R\Cz33�E��&ff��¤��C\\)                                    Bxy��X  �          @��H�n{��33�����Cs�f�n{������¤�=CK�                                    Bxy��  "          @�\�0�׿���Q��=Cz8R�0�׾�G���  §�=CTaH                                    Bxy��  �          @��H?(��.{�ᙚ�HC�H�?(���G���ff�qC��R                                    Bxy�*J  
Z          @���=u������u�C��=u�333��33©aHC�e                                    Bxy�8�  �          @�ff=�G�� ����p���C���=�G����H��p�¬p�C��=                                    Bxy�G�  �          @�ff>�=q��陚8RC�G�>�=q�W
=���
¦�qC��R                                    Bxy�V<  �          @�{�Q녿˅��R
=Cr�{�Q녽�\)��(�§��C90�                                    Bxy�d�  "          @�����H�u��G��CUxR���H?����\ǮC:�                                    Bxy�s�  
�          @��ÿ�������=qB�CY�{��>�
=��z�
=C#�=                                    Bxy��.  �          @��ÿp�׿����Q�Ci��p��>�  ��
=¥��C%L�                                    Bxy���  �          @����E����������Cmh��E�>�{��  §��C�H                                    Bxy��z  
�          @�G���\��
=��¡\Cv����\>�p���Q�ªC#�                                    Bxy��   "          @������������
�=C{�Ϳ�������¬ffC7Y�                                    Bxy���  "          @��
�8Q��z���{u�C����8Q��G����°�CSB�                                    Bxy��l  T          @�ff���
�s33���\CP����
?
=���R�RC!�                                    Bxy��  T          AG��HQ�s33��(�8RCD���HQ�?���p�C*�                                    Bxy��  T          Az��J=q��\)��=q.CGs3�J=q>�����z��3C,�\                                    Bxy��^  	�          A{��>���ffC%�3��@�
��=qL�C��                                    Bxy�  
�          A���33?�����Q�k�C��33@(����(�G�C&f                                    Bxy��  T          A���?�����p�\)Cc���@;����|�HB�8R                                    Bxy�#P  �          @�{�W�?����p��C)�H�W�@   ����q�CL�                                    Bxy�1�  "          @�(�����?����  �k�C,8R����?�\)��Q��^�RC�                                    Bxy�@�  �          @��R���R?L����ff�e�C):����R@ff����V��Cn                                    Bxy�OB  �          A   ��?0�����
�_�C+&f��?�p��˅�R��C��                                    Bxy�]�  "          @�\)���\>�(����c��C.G����\?޸R�θR�X��C�                                    Bxy�l�  "          @�����H>�ff��33�b  C.����H?޸R��(��V��C+�                                    Bxy�{4  T          @�����  >�\)��z��eQ�C0=q��  ?�=q�θR�[�C�)                                    Bxy���  �          @����\)?
=�ٙ��m=qC+�\�\)?��љ��_CY�                                    Bxy���  �          @����w
=?8Q���33�p�C)ff�w
=@33��=q�az�C��                                    Bxy��&  
�          @�33�s�
?=p���=q�q�C)��s�
@�
��G��b
=C�{                                    Bxy���  "          @���y��?�
=�ָR�j�HC#(��y��@�R��33�WC�)                                    Bxy��r  �          @��\�q�?�
=��(��h�C\�q�@<����p��P��C\                                    Bxy��  �          @����mp�?k���Q��r�C&��mp�@{��{�`��C�                                    Bxy��  �          @�G��e�?�������u  C#c��e�@Q���
=�a�CY�                                    Bxy��d  �          @�G��Q�?�����\)�~33C �3�Q�@�R���
�h�\C�                                    Bxy��
  �          @��H�L��?�z��ᙚ� C {�L��@!G���{�j�C�                                    Bxy��  �          @��\�4z�?�=q��\C� �4z�@-p������p{C�                                    Bxy�V  �          @����:=q?������
B�C�H�:=q@$z��׮�p=qC
�\                                    Bxy�*�  �          @�\)�L��?����{��RC ^��L��@�R�ҏ\�i�C:�                                    Bxy�9�  �          @�
=�^�R?�(������u�RC ���^�R@!G�����`z�C\                                    Bxy�HH  �          @����c33?k���33�wQ�C%c��c33@  ��G��e
=C��                                    Bxy�V�  �          @���tz�?(���{�o�
C*���tz�?�z���{�a��Ck�                                    Bxy�e�  �          @�ff�e�?@  �أ��vz�C(=q�e�@�
�Ϯ�f(�C�                                    Bxy�t:  �          @��_\)?p�������w�
C$��_\)@  ��
=�e=qC0�                                    Bxy���  �          @��R�Z�H?333��(��|p�C(\)�Z�H@�\�Ӆ�k��C33                                    Bxy���  �          @���?\)?����aHC)(��?\)?�Q�����yz�C��                                    Bxy��,  �          @�\)��p�>����\ǮC.W
��p�?У������HCL�                                    Bxy���  �          @�
=����=�Q����
£u�C/&f����?�=q��R��B��R                                    Bxy��x  �          @�ff����>.{���
£��C+
=����?�z���{L�B��R                                    Bxy��  �          @��0��=�Q���z�©��C,��0��?�=q��
=
=B���                                    Bxy���  T          @�{���>.{��°��C�\���?���  �
B�8R                                    Bxy��j  T          @�p��#�
>����z�°�qC	.�#�
?����R�B�{                                    Bxy��  
�          @�z�>�=�Q���±��B\)>�?�=q��ff��B�                                    Bxy��  T          @����>����33\C,�)�?�Q�������C�                                    Bxy�\  �          @�(��Q�>�����=qC-��Q�?�Q����
33C��                                    Bxy�$  T          @�{�33>�ff��z�=qC'���33?�����p�C	s3                                    Bxy�2�  �          @��
�.{=u��33�C2�=�.{?����޸R=qC��                                    Bxy�AN  T          @�\�XQ�.{�أ��}��C6ٚ�XQ�?�z���p��w\)C!�                                    Bxy�O�  �          @�=q�,(�>�p�����fC,B��,(�?޸R�ۅ�
C�                                    Bxy�^�  �          @��H�-p�?�\��=qW
C)Y��-p�?���ڏ\aHC33                                    Bxy�m@  "          @��HQ�?!G����p�C(���HQ�?�(������s��C�{                                    Bxy�{�  �          @�p��a�>�(���G��y�\C-
=�a�?�  �ҏ\�l\)C�\                                    Bxy���  �          @���vff>L���ָR�pz�C1\�vff?�G���G��f��C�
                                    Bxy��2  �          @�\)�p  <#�
��  �s�
C3��p  ?�=q���
�l
=C z�                                    Bxy���  �          @�\)�`  ���
�����|\)C4G��`  ?����Q��t�C)                                    Bxy��~  �          @�{�N�R>B�\��
=#�C0�
�N�R?Ǯ�ٙ��x��CL�                                    Bxy��$  
�          @���H��>8Q���\)� C0Ǯ�H��?������{=qC�{                                    Bxy���  �          @�z��C�
>L����  ��C0@ �C�
?�����=q�|�HC޸                                    Bxy��p  �          @�(��?\)>�=q��Q���C.޸�?\)?���ڏ\�}CE                                    Bxy��  T          @��H�)��=�Q���{C2��)��?�p���ff��C��                                    Bxy���  �          @�p��G��aG���\ffC9���G�?�(���\)�3C��                                    Bxy�b  �          @�  ��z�?W
=��  (�C+���z�@33��{��B��                                    Bxy�  �          @�
=���H?aG���\)(�C�Ϳ��H@����B��                                    Bxy�+�  T          @�{��>�����R�3C'���?�{��\)8RC�{                                    Bxy�:T  �          @�\)��ff>�
=��
=C$�H��ff?�33��=q��C
                                    Bxy�H�  �          @�{��  ?333��{�=CO\��  @	�������C:�                                    Bxy�W�  �          @�{���H?0����RG�CͿ��H@����33C ��                                    Bxy�fF  �          @�33�B�\?����\) aHB�aH�B�\@   ��(�ǮB�                                      Bxy�t�  �          @�(���\)?}p���¤z�B�#׾�\)@p���ff�B�=q                                    Bxy���  �          @�\���R?c�
��  ¥��BڸR���R@ff��p�B�#�                                    Bxy��8  �          @�G���ff?z�H��{£� B�녾�ff@���33��B��
                                    Bxy���  �          @�<#�
?Q���  §�B���<#�
@���{B���                                    Bxy���  �          @��H�#�
?aG���G�¦��B�논#�
@ff��
=�)B�W
                                    Bxy��*  �          @�=L��?fff���¦aHB�z�=L��@Q���\)��B��                                    Bxy���  �          @�(�>\)?z�H���¥�B��
>\)@����
=k�B��                                    Bxy��v  �          @��
=q?h����33¤{B�\)�
=q@�������B�z�                                    Bxy��  �          @�>aG�?z�H��¤�HB�\)>aG�@{��Q�L�B���                                    Bxy���  �          @�z�?�?������ (�B�.?�@*�H��(�B�B�ff                                    Bxy�h  �          @�z�?z�?�����  ��B�L�?z�@3�
��\�HB��=                                    Bxy�  �          @��
?\)?�
=��   33B���?\)@)�����
aHB�8R                                    Bxy�$�  �          @���?
=q?��\���¢��By  ?
=q@   ��ff�HB��3                                    Bxy�3Z  T          @�p�?Q�?fff��=q¡�B?ff?Q�@�����B�#�                                    Bxy�B   T          @��
>�(�?�������¢(�B��>�(�@$z�����B�.                                    Bxy�P�  �          @�(�?��?�������¡k�B~z�?��@%�����{B�                                    Bxy�_L  �          @���?J=q?�����Q�aHBbQ�?J=q@*�H���
W
B�                                      Bxy�m�  �          @�(�?   ?�
=��\).B���?   @8���ᙚ�HB��                                     Bxy�|�  �          @�p�>��?�=q��G��HB�#�>��@3�
��(���B��H                                    Bxy��>  �          @��>8Q�?����=q¢��B��=>8Q�@(Q���{ǮB�\)                                    Bxy���  �          @��>B�\?xQ����H¥�B�
=>B�\@{��B�B�=q                                    Bxy���  �          @��R>u?�Q���¡�B��f>u@+���
=�B���                                    Bxy��0  �          @�
==���?��H���
¡�B��q=���@-p���
=�B��)                                    Bxy���  �          @�
=��?��\����¤�3B����@!���G��RB�                                      Bxy��|  �          @�\)=�G�?�����¡��B�\)=�G�@,����
=�B�=q                                    Bxy��"  �          @��?�Q�?�p���z�#�BH�H?�Q�@J�H��z��|B�33                                    Bxy���  �          @�
=?�=q?��H��(�G�Btz�?�=q@Y�����H�y�
B��R                                    Bxy� n  #          @��R?�=q@���33W
Bw�
?�=q@]p��ٙ��w��B�\)                                    Bxy�  �          @�ff?p��@���(�B�L�?p��@^{�ٙ��x��B���                                    Bxy��  �          @�ff?333@�
��z�8RB�(�?333@`  ��=q�y��B�8R                                    Bxy�,`  �          @�ff?Tz�@
=��33��B�.?Tz�@b�\�����w(�B��q                                    Bxy�;  �          @�{?333@	����33B��f?333@e�أ��vz�B��)                                    Bxy�I�  �          @�  ?E�@
=���=qB��f?E�@c�
�ڏ\�w�RB��\                                    Bxy�XR  �          @���?E�@Q��陚�{B�B�?E�@c�
��
=�v\)B���                                    Bxy�f�  �          @��
?�G�@(���
=
=B�Ǯ?�G�@g
=��(��r33B��)                                    Bxy�u�  T          @�\?�G�@{��p�ffB��?�G�@g��ҏ\�p�HB�                                    Bxy��D  �          @�?�z�@
=����=qBt��?�z�@^�R�θR�q�B�(�                                    Bxy���  �          @�  ?�(�?����z��fBd��?�(�@O\)���
�u��B��                                    Bxy���  �          @�ff?�  ?�����#�BH�\?�  @HQ��ə��t��B���                                    Bxy��6  �          @��
?\?�\)��{\)BK33?\@L(����p�HB��H                                    Bxy���  �          @�?��H?�Q���Q�z�BTQ�?��H@QG���\)�pG�B��                                    Bxy�͂  �          @�?�ff@33��  aHBfp�?�ff@XQ���{�n��B��H                                    Bxy��(  �          @�?�Q�@	����Q�.Ba
=?�Q�@^�R��{�j�RB�.                                    Bxy���  �          @��?fff@�
����B�ff?fff@j�H��=q�lG�B�p�                                    Bxy��t  �          @���?c�
@��33�B�aH?c�
@o\)��
=�m�B�8R                                    Bxy�  �          @�
=?�G�@\)�޸R��B���?�G�@vff�ə��f��B��                                    Bxy��  �          @��H?W
=@���ffB�?W
=@l�������kp�B��{                                    Bxy�%f  �          @��
?��
@�����ByQ�?��
@n�R��p��a�\B�(�                                    Bxy�4  �          @��
?��@ff�ҏ\W
Bq�?��@i�����R�c�B���                                    Bxy�B�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxy�QX   1          @ᙚ?�Q�@Q���\)u�Bkp�?�Q�@j=q��33�`G�B�\                                    Bxy�_�  �          @�ff?�G�@p����

=B{\)?�G�@n{��\)�^G�B�z�                                    Bxy�n�  !          @�?�ff@)����=qk�B���?�ff@x����z��Y�RB��R                                    Bxy�}J  T          @�33?O\)@G
=�����w�
B��H?O\)@�G���Q��Ip�B���                                    Bxy���  �          @�(�?��@L����\)�q��B��?��@��
��{�D
=B���                                    Bxy���  �          @�\)?���@2�\��  �~��B�W
?���@�����G��R\)B��H                                    Bxy��<  �          @�  ?��H@3�
��G�#�B�� ?��H@�����=q�SQ�B���                                    Bxy���  �          @߮?���@2�\�ȣ��33B�k�?���@�������R�RB�                                    Bxy�ƈ  T          @׮@#�
@Z=q��p��I��BTQ�@#�
@����33�!��BoG�                                    Bxy��.  �          @��
@(��@N�R���
�K(�BJ�H@(��@�
=���H�$�Bg��                                    Bxy���  T          @��
?�Q�@<(�����gQ�Bbff?�Q�@�G����
�=z�B��R                                    Bxy��z  �          @��@�@Mp������X(�BbQ�@�@�������.p�B}�H                                    Bxy�   �          @У�@@Vff���\�L��B\Q�@@��H�����#�\Bv��                                    Bxy��  �          @�(�@z�@c33��{�Nz�Bo\)@z�@�����\�#�\B�=q                                    Bxy�l  �          @��H?�(�@X�������U��Bo�?�(�@�p���
=�*��B��)                                    Bxy�-  �          @���@C33@_\)���H�>�BC33@C33@�
=����ffB^��                                    Bxy�;�  �          @��@\��@|����p��&��BC\)@\��@��H�o\)� ��BY��                                    Bxy�J^  �          @��@�Q�@�{��p��  B9{@�Q�@�
=�L����
=BL�                                    Bxy�Y  �          @�\)@��@�Q��}p���B2�@��@���>�R��Q�BD�H                                    Bxy�g�  �          @�ff@�Q�@��H�vff�=qB6(�@�Q�@���6ff��z�BGp�                                    Bxy�vP  �          @��@w
=@�z���(����B;��@w
=@���J�H��33BOp�                                    Bxy���  �          @�(�@s33@�G���  �Q�BA�
@s33@��H�P  ���HBU
=                                    Bxy���  �          @�p�@�G�@�G����\�
B;{@�G�@���E��ˮBM��                                    Bxy��B  �          @�
=@�G�@�\)�|(���\B?�H@�G�@�
=�9����{BP�                                    Bxy���  �          @�@�Q�@�Q��xQ���BA�\@�Q�@�\)�5��=qBR(�                                    Bxy���  �          @�@���@���~{��HB.@���@�33�@����
=BA�                                    Bxy��4  �          @�R@��@~�R�z�H��RB$ff@��@�
=�?\)�îB7�                                    Bxy���  �          @�\)@��R@z=q�u�� �B�@��R@�z��:=q��{B2G�                                    Bxy��  �          @�G�@�G�@��H�W���ffB(�
@�G�@��R����B8{                                    Bxy��&  �          @�z�@��@�=q�A���33B'��@��@��
��\����B5p�                                    Bxy��  �          @�
=@��@�G������>�HBG�@��@�����H��{B                                    Bxy�r  �          @��H@��\@�G���p��!��B�@��\@��R�����/\)B                                      Bxy�&  �          @���@��\@������o
=B�H@��\@�33�n{��\)B                                    Bxy�4�  �          @�@��@����p��_\)B�
@��@�z�O\)�θRB                                      Bxy�Cd  �          @��H@��@��׿�\)�3
=B��@��@�\)���w
=B��                                    Bxy�R
  �          @�\@��@z=q���
�H  B
�\@��@��Ϳ&ff����B�                                    Bxy�`�  �          @�z�@�Q�@p�׿333���B =q@�Q�@u�>�?��B�                                    Bxy�oV  �          @�Q�@��@`  >L��?�\)A��@��@W�?s33@�G�A�R                                    Bxy�}�  �          @�G�@���@Z=q>��?�Q�A�=q@���@S33?aG�@�ffA�                                    Bxy���  �          @��H@ȣ�@R�\�\)��z�Aݮ@ȣ�@P  ?�@���A���                                    Bxy��H  �          @�33@���@[��^�R��Q�A�
=@���@a녽��ͿQ�A�
=                                    Bxy���  �          @ᙚ@��@Tz��33�zffA�R@��@j=q������
A��                                    Bxy���  �          @ٙ�@�p�@a��33���RBff@�p�@x�ÿ�p��'\)B�
                                    Bxy��:  �          @�
=@�
=@w
=��(��0  B�@�
=@��þ�33�J=qB
=                                    Bxy���  �          @�\)@G�@�H����Q\)B(�@G�@^{���R�.B@p�                                    Bxy��  �          @�{?�ff@G������Bq{?�ff@c�
��Q��]�
B�                                      Bxy��,  �          @���?�Q�@p�����=qBn�?�Q�@qG����H�X��B�L�                                    Bxy��  �          @�(�?���@33��(�B�By��?���@l����
=�c�B�                                      Bxy�x  �          @�(�?k�@�R��ffffB�ff?k�@i�������h\)B��)                                    Bxy�  �          @��
?=p�?��H�ٙ��RB��{?=p�@Z=q��ff�rz�B�u�                                    Bxy�-�  �          @���?(��@33��=q�B�W
?(��@`����ff�pQ�B��                                    Bxy�<j  T          @�ff?B�\@���=q��B���?B�\@hQ���p��k�B�ff                                    Bxy�K  �          @�\)>��@������B��\>��@c�
�ȣ��p��B���                                    Bxy�Y�  �          @�  >�p�?�z���\)�fB��>�p�@Z=q��z��w
=B���                                    Bxy�h\  �          @��>��H@���ff�fB��>��H@dz���=q�q=qB�k�                                    Bxy�w  �          @陚?�@   ��    B�Q�?�@`  ��(��s��B���                                    Bxy���  �          @陚>�{?�����{B�.>�{@Tz��Ϯ�{�B�#�                                    Bxy��N  �          @��=��
?ٙ���33  B��f=��
@P  ��G��~\)B�Q�                                    Bxy���  �          @���?#�
@�
��{z�B�z�?#�
@c�
�ə��p��B���                                    Bxy���  �          @陚>��R?������B�aH>��R@Vff��
=�z(�B�L�                                    Bxy��@  �          @�\?#�
@33��  �fB�33?#�
@dz��˅�qG�B���                                    Bxy���  �          @��?^�R@
=��ff�3B�W
?^�R@g���G��m�RB�{                                    Bxy�݌  �          @��?�  @p�����B�B�p�?�  @mp���
=�i=qB��f                                    Bxy��2  �          @�?��
@�R�ٙ�G�Bz�R?��
@}p�����^33B�\                                    Bxy���  �          @�p�?���@ff��ff=qB�L�?���@w��Ǯ�e{B�33                                    Bxy�	~  �          @�R?���@33��  �qB
=?���@u���G��fz�B���                                    Bxy�$  �          @�ff?�Q�@{�����Bwff?�Q�@p  �ʏ\�h�\B��                                    Bxy�&�  �          @�{?�
=@
=q���W
Bu�
?�
=@l�����H�j
=B�                                    Bxy�5p  �          @�\)?�z�?���(�\Bk33?�z�@_\)��  �q��B�33                                    Bxy�D  �          @�  ?��?�p����W
Bx��?��@c�
��Q��qp�B�\)                                    Bxy�R�  �          @�Q�?:�H?�����
=�)B���?:�H@c33�ҏ\�t�B���                                    Bxy�ab  �          @�R?�  ?�������
Bx{?�  @^{�����t=qB��H                                    Bxy�p  �          @�\)?8Q�?�33��ff�\B�p�?8Q�@`  �ҏ\�u�B�Ǯ                                    Bxy�~�  �          @�p�?z�@33���H�=B�ff?z�@hQ���p��q  B���                                    Bxy��T  �          @�>��R?�Q���p��HB��>��R@W
=�ڏ\�~�HB���                                    Bxy���  �          @�z�>���?������(�B��>���@b�\�أ��y=qB�p�                                    Bxy���  T          @�p�=�Q�?��
��ff#�B�Q�=�Q�@\�����H�|B�{                                    Bxy��F  T          @�ff>\)?�����{�\B�� >\)@g���G��w�\B�k�                                    Bxy���  T          @�>8Q�@�����.B�{>8Q�@l����
=�t�RB��=                                    Bxy�֒  T          @��?�@*=q����=B���?�@�  ���H�_�HB��f                                    Bxy��8  �          @�{?8Q�@5��(�B��?8Q�@�p���  �YffB��{                                    Bxy���  �          @�?Q�@5��\�qB��q?Q�@����
=�X��B��                                    Bxy��  �          @�p�?p��@<(�����B�?p��@�Q���z��T�\B��\                                    Bxy�*  �          @�p�>�33@����  B�B���>�33@��H��\)�f�\B�=q                                    Bxy��  �          @�G�?�\@p���33B�k�?�\@����=q�dQ�B��{                                    Bxy�.v  �          @���?��
@<������aHB��?��
@������MQ�B���                                    Bxy�=  �          @��
?�@>{��z���B�(�?�@�ff��  �M(�B���                                    Bxy�K�  �          @�\)?�ff@O\)���
�x�RB�p�?�ff@�
=����D�RB�G�                                    Bxy�Zh  �          @��?���@`����Q��o��B�#�?���@�ff����;�B��                                    Bxy�i  �          @�G�?���@e���{�j��B��
?���@�Q������7(�B��=                                    Bxy�w�  �          @�Q�?�\)@\����
=�m�B���?�\)@�z����R�:��B�B�                                    Bxy��Z  �          @��H?�  @j�H����fz�B��?�  @��H��33�3Q�B�                                      Bxy��   T          @�p�@�@�Q�����I�HBs�@�@�����
=��B��                                    Bxy���  �          @�z�@G�@�����Q��OQ�B�k�@G�@��H����
=B�G�                                    Bxy��L  �          @�z�?�G�@�ff��ff�Y{B�p�?�G�@����Q��$
=B���                                    Bxy���  �          @�{?˅@�(����
�R�HB�  ?˅@�
=��z����B��H                                    Bxy�Ϙ  �          @�ff?�33@�����  �L��B�?�33@��H��\)���B��                                     Bxy��>  �          @�
=?�@�  �����L\)B�� ?�@����  ��RB��f                                    Bxy���  �          @�{@�@{���p��U\)Bk��@�@�������#�\B��                                    Bxy���  �          @�{@\)@�33���I�Bz�@\)@����ff��RB�                                    Bxy�
0  �          @�
=@
=q@�Q������N�
B|��@
=q@�����H�\)B��f                                    Bxy��  �          @�{@�@tz��˅�^��Bx{@�@�Q���\)�+�\B��q                                    Bxy�'|  �          @��R?�@q���
=�d\)B��H?�@������0  B���                                    Bxy�6"  T          @��@p�@r�\�ȣ��X�
Bd�@p�@��R�����'=qB�p�                                    Bxy�D�  �          @��@�
@����Ǯ�W��B|ff@�
@��R����#��B��R                                    Bxy�Sn  �          @��@�\@���ȣ��Xz�B}ff@�\@�
=���H�$=qB�.                                    Bxy�b  �          @�Q�@Q�@����\)�U��Bz�@Q�@�Q������!��B���                                    Bxy�p�  �          @��@p�@���\�OG�By�\@p�@�33��33�=qB��)                                    Bxy�`  �          @�
=@@�
=��33�D�
By��@@���������B��                                    Bxy��  �          @�
=@{@��
�����Az�B���@{@����{���B��H                                    Bxy���  �          @�\)@{@�����{�IQ�Bo�H@{@����ff���B��                                    Bxy��R  �          @�\)@=q@������
�E�Buff@=q@�\)���H��B�k�                                    Bxy���  �          @�ff@�\@����(��G33Bz(�@�\@����33���B�p�                                    Bxy�Ȟ  �          @�ff@��@�����
=�@33B�B�@��@���(��
�
B�8R                                    Bxy��D  �          @�
=@�R@�����
�:�B�W
@�R@������
=B��                                    Bxy���  �          @��R@@�  ��33�:Q�B�@@�  ��\)�{B�k�                                    Bxy���  �          @�
=@G�@�
=���
�F
=B|ff@G�@���������B�Q�                                    Bxy�6  �          @�  @\)@��������M33Bz  @\)@��������B�                                    Bxy��  �          @���@��@�(���
=�HffBuff@��@����p���B�Ǯ                                    Bxy� �  �          @�Q�@
=@�ff��=q�NQ�Br�\@
=@�33�����\B�=q                                    Bxy�/(  �          @�Q�@   @�G���(��P�HBi33@   @�
=�����B��f                                    Bxy�=�  �          @��@   @|(����S��Bf�@   @�(���
=���B�=q                                    Bxy�Lt  �          @�Q�@ff@�G���ff�S�HBo\)@ff@�����R�  B���                                    Bxy�[  �          @�\)@G�@i����p��`��Bh33@G�@�p������,Q�B�u�                                    Bxy�i�  �          @��@%@w�����S�Ba
=@%@�=q��
=� �B�#�                                    Bxy�xf  �          @�
=@ ��@�������@(�Bs��@ ��@��H�����
�RB��                                     Bxy��  �          @��\@9��@�\)����� ��Bt
=@9��@�33�a��ָRB��                                    Bxy���  �          @���@9��@�33����$(�Bqz�@9��@Ǯ�g���  B�(�                                    Bxy��X  �          @���@Mp�@���������Be�
@Mp�@����aG���p�By�R                                    Bxy���  �          @���@Vff@����  ���Bc��@Vff@ƸR�P����Q�Bvz�                                    Bxy���  �          @��\@j=q@�(��������BY��@j=q@���J=q��Q�Bm{                                    Bxy��J  T          @���@dz�@��R�����B^=q@dz�@�Q��N{����Bq(�                                    Bxy���  �          @���@S�
@���������Be{@S�
@ȣ��^{�иRBx�                                    Bxy��  �          @�33@&ff@�(����\�+(�B|ff@&ff@�33�s�
��(�B�aH                                    Bxy��<  �          @�33@4z�@�z���\)�&�
Bu(�@4z�@ʏ\�l�����HB�                                    Bxy�
�  �          @��@@  @�{��Q��ffBo�H@@  @�=q�^{����B�33                                    Bxy��  �          @�ff@K�@�(���(��\)Br�@K�@����?\)��z�B��
                                    Bxy�(.  �          @�\)@G
=@�ff��(���HBuz�@G
=@ָR�=p�����B�G�                                    Bxy�6�  �          @�ff@J�H@�33���
=Bq�
@J�H@�z��B�\��p�B��H                                    Bxy�Ez  �          A   @0��@������=qB��@0��@���G���=qB��f                                    Bxy�T   �          A ��@5@�Q�������Bz�H@5@��`����z�B��f                                    Bxy�b�  �          A�R@Vff@��������RBk  @Vff@����Vff���B}                                      Bxy�ql  �          A�@e�@�����  �=qBdQ�@e�@��H�G
=��{Bv
=                                    Bxy��  �          A{@j�H@�������Bb33@j�H@�33�@�����
Bs��                                    Bxy���  �          Aff@L��@��H��  � z�BlG�@L��@љ��hQ��ӮB�
                                    Bxy��^  �          A�\@H��@����{�z�Bs�R@H��@أ��P  ��  B�33                                    Bxy��  �          Aff@L(�@�Q�������\Bt(�@L(�@��H�E����B�                                    Bxy���  �          A�@Fff@�����Q��p�Bw�@Fff@��
�AG�����B�ff                                    Bxy��P  �          A�\@�
=@�Q��z�H��(�BW  @�
=@�33�{�}p�Be��                                    Bxy���  �          A
=@~{@�{�\)��B`��@~{@�G��\)�~ffBn�
                                    Bxy��  �          A��@��@�\)�~{��
=BYQ�@��@��H�G���Q�Bh(�                                    Bxy��B  �          A@w�@�ff�������RB_Q�@w�@����$z����
Bo33                                    Bxy��  �          Ap�@��@�Q��z�H��  BY��@��@Ӆ�p��}p�Bh�                                    Bxy��  �          A�@�z�@���mp��ٮB[�@�z�@�ff���H�_�Bi{                                    Bxy�!4  �          AG�@���@�p��[���33BX�
@���@�(���
=�@Q�Bd�                                    Bxy�/�  �          A�@�{@���Z�H�ȏ\B\
=@�{@�ff��33�<z�Bg��                                    Bxy�>�  �          Ap�@��@��
�]p���z�BU�\@��@��H���H�C�
Bb{                                    Bxy�M&  �          Ap�@�z�@���h����BRff@�z�@У׿��[\)B`33                                    Bxy�[�  �          Ap�@�
=@���tz���BV�@�
=@�=q��p  Be(�                                    Bxy�jr  �          Ap�@}p�@�ff������
=B\�
@}p�@��
�=q���HBl�                                    Bxy�y  �          A@mp�@�  ��z��  B_@mp�@�=q�<(���ffBr                                      Bxy���  �          A(�@]p�@��R��(���Bfff@]p�@�p��Z=q��\)Bz33                                    Bxy��d  �          A�
@[�@�ff��(��G�Bg\)@[�@���Z=q��  B{(�                                    Bxy��
  �          AQ�@Z=q@�z������\Bkp�@Z=q@���N{��p�B}��                                    Bxy���  �          Aff@Q�@��
��z��Bn��@Q�@أ��HQ����RB�G�                                    Bxy��V  T          A�\@_\)@����33��HBg��@_\)@�ff�Fff���\By��                                    Bxy���  �          A�@U�@��\���
�\)Bl��@U�@�
=�Fff���B~�\                                    Bxy�ߢ  �          Aff@W�@�G�����HBj��@W�@ָR�J�H��33B}G�                                    Bxy��H  �          A�\@U@�p����H�ffBm��@U@ٙ��C33��33Bff                                    Bxy���  �          Aff@R�\@����p��Bq�@R�\@����5�����B���                                    Bxy��  �          A{@P��@�33����
=qBsff@P��@�p��1G���33B��                                     Bxy�:  �          A�\@Q�@��\�����\Brp�@Q�@���4z���B�33                                    Bxy�(�  �          A�@S33@��H���\�	p�Br{@S33@����.�R��\)B��H                                    Bxy�7�  �          A�
@`  @�����z�� ��Bo\)@`  @���\)����B~{                                    Bxy�F,  T          A�
@e�@�ff���\��p�Bp(�@e�@�����p��B}=q                                    Bxy�T�  �          A(�@_\)@Ǯ��z���RBs(�@_\)@�����t��B�{                                    Bxy�cx  �          A  @p��@��
�hQ���
=Bm��@p��@�z��33�8��Bx                                    Bxy�r  �          A�@�{@�ff�XQ��£�B_�H@�{@�����H�#\)Bk(�                                    Bxy���  
�          A��@~�R@����L�����Bj�@~�R@���
=��HBt{                                    Bxy��j  �          A��@��\@У��G���ffBg�R@��\@�zῌ����Bp��                                    Bxy��  �          A�@��R@�ff�c33�ϙ�B[  @��R@�
=���>=qBg�H                                    Bxy���  �          A=q@�  @���i�����HBX��@�  @ָR��\�Ip�Bf�                                    Bxy��\  �          A�@�ff@��H�
=��\)B[�
@�ff@�  ����9��Bb�\                                    Bxy��  �          A@�\)@�p��+���33BU
=@�\)@ҏ\?�ffA�BSz�                                    Bxy�ب  �          Ap�@�ff@����eBU@�ff@�G�?���A ��BSp�                                    Bxy��N  �          A  @��@��ÿ�p��%�BV�@��@�>��H@\(�BY                                      Bxy���  �          A�@�Q�@Ϯ��Q��=BV�@�Q�@�ff>�z�@G�BZff                                    Bxy��  �          A�@��@θR�����LQ�BV�H@��@ָR>#�
?�33B[                                      Bxy�@  �          Az�@�=q@�{���H�[�BT��@�=q@�\)<��
>��BY�                                    Bxy�!�  �          Az�@��\@�{���V�HBT��@��\@�
==�\)>�BYG�                                    Bxy�0�  T          A��@��H@��H��33��BV��@��H@ָR?(�@�  BX                                    Bxy�?2  �          A�@���@Ӆ��
=�W33B[z�@���@�z�=�?O\)B_�
                                    Bxy�M�  �          A�@�G�@�z�Y����{BX�
@�G�@��H?�z�A{BX
=                                    Bxy�\~  T          A�@��\@�\)��{�4z�BU�@��\@��>Ǯ@.�RBX33                                    Bxy�k$  �          A�@�
=@��Ϳ����\)BP�R@�
=@�Q�?�R@��BR��                                    Bxy�y�  �          A�H@��
@�p�������
BS\)@��
@���?#�
@�\)BU33                                    Bxy��p  �          A�H@�33@�G����H�]G�BQ\)@�33@��H<��
>#�
BVp�                                    Bxy��  �          A�R@���@�33��
=�Z�HBT=q@���@�z�=�\)>�BY{                                    Bxy���  �          A{@���@ȣ׿��R�b�RBR�@���@ҏ\�#�
�L��BW��                                    Bxy��b  �          A�\@��@�  ��
�j=qBQ��@��@ҏ\��\)��BWG�                                    Bxy��  �          A=q@���@�����l��BZz�@���@׮�#�
��=qB_�
                                    Bxy�Ѯ  �          A�@���@��
��(��`(�BWQ�@���@��=u>��B\G�                                    Bxy��T  �          Ap�@���@ƸR���\Q�BQ�@���@�  =u>���BV(�                                    Bxy���  �          A��@��
@Ǯ��(��D��BP�@��
@θR>��?�BT�                                    Bxy���  �          A ��@��R@�G���p��B@{@��R@��;k�����BG                                      Bxy�F  �          A�@�G�@�=q��  �G�BI33@�G�@��>L��?��BM��                                    Bxy��  �          A��@�  @��
��ff�L��BJ�@�  @��
>.{?�Q�BOz�                                    Bxy�)�  �          A��@��@�����=q�Qp�BH�@��@ʏ\=�?Tz�BMp�                                    Bxy�88  �          A=q@�=q@ƸR�
=�p��BP�@�=q@�녽��Ϳ.{BV�R                                    Bxy�F�  �          A=q@��\@�ff�
=�p  BP=q@��\@љ���Q�(�BV=q                                    Bxy�U�  �          Ap�@�@���ff�q�BKff@�@�����aG�BQ�R                                    Bxy�d*  �          A�@�  @������g�BI�@�  @�(��L�;�33BO��                                    Bxy�r�  �          A@�@����G����BJ�
@�@���\)���RBR�                                    Bxy��v  �          A�@���@���������BK�\@���@θR��{��BS(�                                    Bxy��  �          A{@�
=@����G���p�BI�
@�
=@���=q��z�BQ(�                                    Bxy���  �          A�@���@�
=�
=��  BJz�@���@��;�p��)��BRff                                    Bxy��h  �          A�@�33@�
=�(����RBK��@�33@���G��H��BS��                                    Bxy��  �          Ap�@��@�G���H���BM��@��@Ϯ�����3�
BU�
                                    Bxy�ʴ  �          A Q�@�ff@����ff���\BP�\@�ff@�\)�����BX(�                                    Bxy��Z  �          A z�@�ff@��H�33��G�BQG�@�ff@�  �����\)BX�                                    Bxy��   �          A ��@���@���   �f=qBQ  @���@�
==L��>ǮBV�                                    Bxy���  �          A ��@�ff@�  ��\)�W\)BT33@�ff@У�>W
=?�p�BX�
                                    Bxy�L  �          A ��@�=q@�  ��{�8(�BQz�@�=q@�>�@S33BT                                    Bxy��  �          A (�@�@Ǯ���
�N=qBT\)@�@�\)>���@�BX�\                                    Bxy�"�  �          A   @��@�G�����=��BU�@��@Ϯ>�ff@O\)BY=q                                    Bxy�1>  �          A ��@�z�@��
�Ǯ�3
=BW�R@�z�@�G�?\)@�Q�BZz�                                    Bxy�?�  �          A Q�@�(�@�33�����5G�BW@�(�@У�?��@y��BZ��                                    Bxy�N�  �          A   @�z�@��H�\�/
=BW33@�z�@Ϯ?��@���BY                                    Bxy�]0  �          A{@�z�@ȣ��(��{\)BV{@�z�@�z�u����B\\)                                    Bxy�k�  �          A{@��@�녿!G����RBM�@��@�?�Q�A#�BJ�R                                    Bxy�z|  �          A�@�@�p��u�ٙ�BG��@�@��?�\AJ�HBC
=                                    Bxy��"  �          @�@��@�33�h���ҏ\BJ�@��@��?�\)A��BJG�                                    Bxy���  �          A (�@��@�(��:�H��
=BH�@��@�G�?�ffAp�BF�                                    Bxy��n  T          @�ff@�  @�(��333��G�BK\)@�  @���?��A
=BI\)                                    Bxy��  �          @��@��\@�G���G��J�HBG@��\@�33?ǮA5BD(�                                    Bxy�ú  �          @��@�=q@����\�2�\BHG�@�=q@�33?�\)A<��BDQ�                                    Bxy��`  �          @�{@�33@�����Tz�B>=q@�33@�=q?�AR�HB8�                                    Bxy��  �          @�z�@���@����B�\����BH��@���@���?�AT(�BCff                                    Bxy��  �          @�{@�{@�Q��G��G�BD�
@�{@��R?��AZ�RB?                                      Bxy��R  T          @�ff@��@��>8Q�?��B@=q@��@���@�Aw�B8p�                                    Bxy��  �          @�\)@��@���>8Q�?���B?
=@��@�Q�@�Aw
=B7(�                                    Bxy��  �          @�
=@��H@�p�>�  ?�G�B?@��H@�  @(�A33B7z�                                    Bxy�*D  �          A (�@�(�@�p�>aG�?���B?  @�(�@���@
�HA|  B6�
                                    Bxy�8�  �          @��R@���@�>W
=?�ffB@��@���@���@
�HA}B8��                                    Bxy�G�  �          @�@�ff@�\)=���?:�HBC�H@�ff@�33@AtQ�B<z�                                    Bxy�V6  �          @��H@�ff@��H>�(�@J=qBA=q@�ff@��@
=A�{B7p�                                    Bxy�d�  �          @��H@���@��?@  @�  B=�@���@��@(Q�A��HB1p�                                    Bxy�s�  �          @��@�ff@�Q�?\(�@��B?@�ff@�z�@/\)A�ffB2�                                    Bxy��(  �          @��H@���@�?�33A�HB<�\@���@�
=@?\)A��B-�                                    Bxy���  �          @�p�@�33@��?��\@���B<  @�33@��@8��A�p�B-�\                                    Bxy��t  T          @��H@�\)@��?   @hQ�B@  @�\)@���@�A�ffB5�                                    Bxy��  �          @�\)@�  @��
��=q���RBF=q@�  @��?�p�AO�BA33                                    Bxy���  �          @��R@�Q�@�=q�!G���p�BE{@�Q�@�{?���A%G�BBz�                                    Bxy��f  �          @�ff@��H@�������33BI�R@��H@�p�?\(�@�{BJ�
                                    Bxy��  �          @�
=@�{@\�ff�t��BQ\)@�{@�=�Q�?.{BW�\                                    Bxy��  �          A Q�@��R@�  ��(��F�RBT(�@��R@�
=?�\@j=qBW�
                                    Bxy��X  T          @�\)@�G�@ʏ\�����R�HBYff@�G�@�=q>�ff@Mp�B]p�                                    Bxy��  T          @��@���@�\)��p��IBWG�@���@�{?�\@l��BZ��                                    Bxy��  �          A=q@��@ȣ׿����N=qBP�
@��@�Q�>�G�@HQ�BU                                      Bxy�#J  �          Aff@���@�녿�33�X(�BS
=@���@ҏ\>\@+�BW��                                    Bxy�1�  �          A�@���@Å�,����33BO@���@����(��>{BYQ�                                    Bxy�@�  �          A�@�33@����2�\���RB_G�@�33@�\)����7
=Bh33                                    Bxy�O<  �          A�@���@�=q�:=q���B\�\@���@��
=q�r�\Bf\)                                    Bxy�]�  �          A\)@�{@�
=�@  ��G�BZ  @�{@��
�(�����\Bd��                                    Bxy�l�  �          A�
@�{@Ǯ�B�\��p�BZ=q@�{@���0����G�Be
=                                    Bxy�{.  �          A(�@��@��H�)�����
BYG�@��@��
��=q��z�Ba��                                    Bxy���  T          A�@�(�@ȣ��?\)���RB\G�@�(�@�p���R����Bf�                                    Bxy��z  T          A(�@���@ʏ\�<����\)B\�@���@޸R����tz�Bf�                                    Bxy��   
�          A��@�Q�@��H�7���  BZQ�@�Q�@�{����P  Bd
=                                    Bxy���  �          A�@�z�@�z��'
=��Q�BX�@�z�@��;L�Ϳ��B`ff                                    Bxy��l  �          A�R@�
=@�=q�ff����BYG�@�
=@�\)=�G�?=p�B_                                    Bxy��  �          A=q@��@��
�
=����BQ�\@��@ٙ�<�>uBX��                                    Bxy��  �          A�@�  @�
=�Q���=qBV@�  @�z�=L��>�Q�B]��                                    Bxy��^  �          A{@�  @��
�	���nffBOz�@�  @�
=>u?�\)BUff                                    Bxy��  �          A��@�33@��
��\�C�
BMff@�33@��H?��@�z�BQ{                                    Bxy��  �          A�@��H@�(���{�3
=BM�@��H@�G�?B�\@��RBPff                                    Bxy�P  �          Aff@�  @�����o�
BP{@�  @�Q�>u?�z�BV
=                                    Bxy�*�  �          A�@�G�@�  �'
=����BR=q@�G�@أ׾W
=����B[                                      Bxy�9�  �          AQ�@��@�������BG��@��@��?��RA{BGG�                                    Bxy�HB  �          AQ�@�
=@�G��Ǯ�%B>�
@�
=@�Q�?��AMG�B9�                                    Bxy�V�  �          A	p�@�p�@ƸR�����p�B9Q�@�p�@��?�z�AN{B3                                    Bxy�e�  T          A��@�\)@��H�W
=��33B5��@�\)@�Q�?��RAW�
B/�\                                    Bxy�t4  T          A��@�33@�
=�#�
���
B;{@�33@��@�RAr{B3=q                                    Bxy���  �          A��@�z�@�p���Q�
=B9=q@�z�@�G�@��Ah��B1�H                                    Bxy���  �          A��@���@��þ#�
����B=  @���@��@
=Ad��B6(�                                    Bxy��&  �          A��@��@��H�k�����B?G�@��@��@�A`��B8�H                                    Bxy���  �          A	�@��@�33��33�B?��@��@���?�p�AU��B:
=                                    Bxy��r  �          A��@��@�G������  B>�@��@\?�p�A;33B:�\                                    Bxy��  �          A��@��@�Q쾀  ��33B<�\@��@�p�@33A^{B6(�                                    Bxy�ھ  T          Az�@��@�z����n{BB@��@���?�AD��B>p�                                    Bxy��d  �          A�
@�z�@�Q쿀  ��Q�B?��@�z�@�{?�{A\)B>�R                                    Bxy��
  �          A  @��@�  ������B?Q�@��@�ff?�ffA��B>z�                                    Bxy��  �          Az�@��@ə���  �׮B@33@��@�\)?���A��B>�H                                    Bxy�V  �          A	�@�@��\(���\)BBG�@�@ə�?ǮA'33B?�H                                    Bxy�#�  �          A	@�p�@�p��c�
��ffBB{@�p�@�G�?��
A$  B?�H                                    Bxy�2�  �          A	@���@�������33BBp�@���@�33?�\)A
=BA\)                                    Bxy�AH  �          A
{@��\@�
=���
���BD�H@��\@�\)?���@��RBE33                                    Bxy�O�  �          A
ff@���@Ϯ��\)�G�BE�
@���@���?���@�G�BF��                                    Bxy�^�  �          A
=q@�z�@ʏ\��\)�-G�BA{@�z�@�\)?W
=@��HBC                                    Bxy�m:  �          A��@��\@�
=���F�\B@z�@��\@θR?(�@��BD�R                                    Bxy�{�  �          A�
@�ff@�Q��Q��7�
BC��@�ff@�{?E�@�{BG�                                    Bxy���  �          A
ff@�ff@�\)��p��8��BG��@�ff@��?O\)@�(�BJ�H                                    Bxy��,  �          A	@�Q�@��Ϳ�
=�4z�BE
=@�Q�@�=q?Tz�@���BG�                                    Bxy���  �          A	p�@�Q�@�(����3
=BD��@�Q�@�G�?W
=@��
BGp�                                    Bxy��x  �          A
ff@���@�z�����B=qBDz�@���@Ӆ?5@��RBH=q                                    Bxy��  �          A
�R@���@�33��Q��4z�BAp�@���@У�?Q�@���BDz�                                    Bxy���  �          A�@��
@љ�����BE�@��
@�=q?�(�A Q�BE�                                    Bxy��j  T          A�@�33@љ���ff�	p�BE�H@�33@��?��\A��BF                                      Bxy��  �          A
=@��@�G�����BF�@��@��?�p�AffBF�H                                    Bxy���  �          A
�\@�  @�  �����*�RBF�@�  @��
?xQ�@�ffBI
=                                    Bxy�\  �          A�@��
@�{���0��BCz�@��
@��H?c�
@���BF�                                    Bxy�  �          A�@�(�@�(���33�IG�BB=q@�(�@��
?+�@�(�BFz�                                    Bxy�+�  �          A
=@���@�=q�   �T��B7=q@���@��
>�@B�\B<�
                                    Bxy�:N  �          A
�R@���@�(��z��\��B1G�@���@�
=>���@�B7��                                    Bxy�H�  �          A
�H@�33@��
��33�J�\B9�@�33@�(�?z�@u�B=�                                    Bxy�W�  T          A33@��@��H����H��B7�@��@�33?
=@w�B<G�                                    Bxy�f@  �          Az�@�Q�@�������$��B@=q@�Q�@У�?�G�@��
BB33                                    Bxy�t�  �          A��@�33@�z῰���(�BGff@�33@��?��
A��BG�R                                    Bxy���  �          AG�@��@�Q쿊=q��  B@  @��@�p�?�G�AB>p�                                    Bxy��2  �          A�@��@�  �Q����\B>�
@��@��?�  A6�RB;�                                    Bxy���  T          A��@�
=@Ӆ������z�BD�@�
=@�G�?�G�Ap�BC\)                                    Bxy��~  �          Az�@���@��H���I��BC  @���@�Q�@ffA]B==q                                    Bxy��$  �          A��@��@�=q���^�RBB  @��@ȣ�@33AX(�B<�                                    Bxy���  �          Ap�@�
=@ָR�(������BFQ�@�
=@�{@   AP��BA��                                    Bxy��p  �          A�@��R@�  �0�����BG(�@��R@Ϯ?�p�AN�HBB�                                    Bxy��  �          A��@���@ָR�&ff��Q�BG�@���@�{@ ��AS�BB��                                    Bxy���  �          AQ�@���@���5���BF�R@���@��?�Q�AL��BBff                                    Bxy�b  �          A(�@�{@��;�z���BE�
@�{@Ǯ@�Aw\)B>z�                                    Bxy�  �          A  @�\)@�=q����|(�BC@�\)@���@G�AU�B>��                                    Bxy�$�  �          A  @��\@�Q쾽p����B@��@��\@�(�@p�Aj�RB9��                                    Bxy�3T  �          A�
@�=q@�  ��\�VffB@��@�=q@�p�@�A\z�B:�
                                    Bxy�A�  �          A�@��@Ϯ�Ǯ�%B@�@��@��
@(�Ah��B:                                      Bxy�P�  �          A�
@���@�녽#�
��\)BB@���@���@#33A���B9p�                                    Bxy�_F  �          A�@�ff@��H��Q��BD��@�ff@��H@!G�A��\B;��                                    Bxy�m�  �          A�@�  @љ���=q��  BB��@�  @��
@Ay��B;=q                                    Bxy�|�  �          A�
@�\)@��H�������BC��@�\)@�@�
Au�B<�                                    Bxy��8  �          A�
@���@љ��\�   BBG�@���@��@  AnffB;\)                                    Bxy���  �          A��@���@�33���J=qBB��@���@�  @�Ae�B<�\                                    Bxy���  �          A��@��@�(�����eBDff@��@ə�@Q�A`��B>��                                    Bxy��*  �          A��@��R@�������(Q�BE�@��R@ȣ�@�\AqG�B>��                                    Bxy���  �          AQ�@�33@�\)���XQ�BH�
@�33@��
@p�Ai��BB                                    Bxy��v  �          Az�@�\)@�z��
=�/\)BD�H@�\)@�  @G�Ao�
B>{                                    Bxy��  �          A��@�z�@�  ��  ����BHp�@�z�@���@�RA���B@Q�                                    Bxy���  �          A��@�z�@׮�k���(�BHQ�@�z�@�Q�@   A�  B@
=                                    Bxy� h  �          AQ�@�  @��
<�>aG�BD33@�  @���@,(�A�
=B9�H                                    Bxy�  �          AQ�@�z�@�ff��p����BG�@�z�@���@ffAyG�B@ff                                    Bxy��  �          A��@�{@�ff��ff�=p�BF��@�{@�=q@�\Ap��B?�                                    Bxy�,Z  �          A��@���@�  ��33���BHff@���@�=q@=qA}�B@�
                                    Bxy�;   �          A�@���@�G����L��B?�H@���@�  @'�A�Q�B5�                                    Bxy�I�  �          A��@�(�@��
>L��?��\B8=q@�(�@��@0��A�\)B,ff                                    Bxy�XL  �          A�@��
@\?(�@|��B.��@��
@��@AG�A�(�B33                                    Bxy�f�  �          A��@�p�@���?0��@�\)B-(�@�p�@�  @EA�p�B
=                                    Bxy�u�  �          A@ҏ\@��?c�
@�Q�B&�R@ҏ\@��@L��A�G�B�R                                    Bxy��>  �          A��@�@���?=p�@�G�B,@�@��R@HQ�A�  B{                                    Bxy���  �          AG�@�=q@\?z�H@���B/�@�=q@���@XQ�A�=qB                                    Bxy���  T          A��@ƸR@�33?�
=@��RB1�H@ƸR@��\@dz�A�B(�                                    Bxy��0  �          A�@�33@\?�Q�@�{B.�@�33@��@dz�A��B�                                    Bxy���  �          A�\@љ�@��?��RA ��B(G�@љ�@�(�@c�
A�ffB                                    Bxy��|  �          AG�@�(�@���?�\)A33B!�R@�(�@��H@dz�A��B
�R                                    Bxy��"  �          A��@��@���?k�@�\)B%�
@��@��@N{A��B(�                                    Bxy���  �          AG�@�=q@�G�?h��@�Bp�@�=q@�p�@G
=A�33B	                                    Bxy��n  �          AG�@���@��H?J=q@���B��@���@���@A�A�z�B\)                                    Bxy�  �          A@�=q@��H?c�
@�Q�B�@�=q@�\)@G�A�
=B
��                                    Bxy��  �          A@�{@\>#�
?��B-z�@�{@�
=@*�HA�(�B!ff                                    Bxy�%`  �          AG�@�(�@�33�W
=���B7�@�(�@��@(�A��\B.��                                    Bxy�4  �          A{@Å@�{������B:
=@Å@�\)@��A{
=B1��                                    Bxy�B�  �          Aff@\@�\)�����%�B;p�@\@��@�As
=B3                                    Bxy�QR  T          Aff@�\)@�=q��ff�8Q�B>�H@�\)@��@�Ar�HB7�                                    Bxy�_�  �          A�R@�33@Ϯ��33�\)B;(�@�33@���@��Ax��B3                                      Bxy�n�  T          Aff@�p�@����=q�޸RB8ff@�p�@�{@�A}��B/��                                    Bxy�}D  T          Aff@Ǯ@ʏ\���>{B5��@Ǯ@�{@�RAh  B.\)                                    Bxy���  �          A{@���@�(�����@��B/{@���@�Q�@	��A_�B(                                      Bxy���  �          A�@��@ƸR�
=q�aG�B2(�@��@��@
=A\(�B+��                                    Bxy��6  �          A=q@��H@�p��Tz���33B0�H@��H@�ff?�A?
=B,�                                    Bxy���  �          A�@�=q@�p��(����Q�B1G�@�=q@�(�@   AP��B+�                                    Bxy�Ƃ  �          A{@˅@��   �P��B0��@˅@��@��A_�B)�R                                    Bxy��(  �          Aff@�p�@�z����vffB.��@�p�@�=q@33AT��B(�
                                    Bxy���  �          A�@θR@����z��n{B,\)@θR@�
=@�AT  B&�                                    Bxy��t  �          AG�@ʏ\@�(��!G���33B0Q�@ʏ\@�=q@G�AS�B*p�                                    Bxy�  �          AG�@�  @�>�  ?�33BE(�@�  @�ff@C�
A�ffB7�
                                    Bxy��  �          AG�@��@ڏ\?�\)@�Q�BL�R@��@��R@z=qA�  B8ff                                    Bxy�f  �          Ap�@�z�@�Q�?8Q�@���BH�\@�z�@��@aG�A��
B7p�                                    Bxy�-  �          A��@�p�@�  ���
��B>��@�p�@�p�@,(�A��\B3�H                                    Bxy�;�  
�          A�@Ǯ@��Ϳs33��B2=q@Ǯ@�
=?�G�A8z�B.��                                    Bxy�JX  �          A��@��H@�G��B�\��\)B7�R@��H@���?��RAQ��B2�\                                    Bxy�X�  �          AG�@�ff@Ϯ����z=qB=�H@�ff@��
@�RAi�B7=q                                    Bxy�g�  �          A@�@�녿\)�g
=B?p�@�@�p�@33Ap��B8ff                                    Bxy�vJ  �          A=q@��R@ҏ\��ff�8Q�B?G�@��R@�(�@=qA|(�B7G�                                    Bxy���  �          A=q@�  @љ��\��B>
=@�  @\@{A�
=B5z�                                    Bxy���  �          A@�{@�=q��Q��z�B?�\@�{@��H@   A���B6��                                    Bxy��<  �          AG�@�G�@�z���n{BC�\@�G�@Ǯ@AuB<�                                    Bxy���  �          A�@�\)@�Q켣�
�#�
BF��@�\)@��
@8��A�Q�B;z�                                    Bxy���  �          A@��R@أ�<��
>��BGz�@��R@�33@<(�A��B;��                                    Bxy��.  �          A{@�{@ٙ����
�z�BHQ�@�{@ȣ�@)��A��HB?�                                    Bxy���  �          A
=@�\)@�>��@(Q�BX��@�\)@�33@`  A�ffBJ�                                    Bxy��z  �          A�R@�@�\)?�R@�Q�BYz�@�@�  @l(�AîBIG�                                    Bxy��   �          A�\@���@��?#�
@�(�BV��@���@�p�@k�A�33BF{                                    Bxy��  �          A�\@�z�@�=q?
=@s�
BR�@�z�@Å@fffA��RBB\)                                    Bxy�l  �          A�@�@�Q�>\@�RBP�@�@���@X��A�BB�                                    Bxy�&  �          A��@�{@��þ����z�BG�
@�{@ƸR@.{A�
=B=��                                    Bxy�4�  �          A��@�33@�33=�G�?8Q�BJ��@�33@Å@E�A�33B=��                                    Bxy�C^  �          Ap�@�Q�@��>8Q�?�
=BM�R@�Q�@�z�@K�A���B@Q�                                    Bxy�R  �          A�\@�\)@�z�?�  @�
=BW  @�\)@�\)@���A֣�BCG�                                    Bxy�`�  �          A=q@�\)@��?.{@�z�BW�@�\)@�z�@o\)A�G�BFff                                    Bxy�oP  �          A�@�(�@�
=?��@x��BZp�@�(�@�
=@l��A�G�BJ�                                    Bxy�}�  �          A{@�Q�@�?+�@��BN�@�Q�@�{@h��A���B<��                                    Bxy���  �          A�@�{@߮>��@C33BPff@�{@�=q@_\)A��B@�                                    Bxy��B  �          A=q@��H@�>�\)?��
BT\)@��H@�Q�@W�A�Q�BF33                                    Bxy���  �          A�@��
@�      ���
BQ��@��
@���@E�A�\)BE�                                    Bxy���            Az�@��\@�
=�k���G�BRG�@��\@˅@7
=A�Q�BH                                      Bxy��4  �          A  @��
@�(�����E�BP(�@��
@�(�@'�A�G�BG�R                                    Bxy���  �          A�
@��@��H�(���  BN��@��@���@{A���BG=q                                    Bxy��  �          A(�@�ff@�녿O\)��=qBM\)@�ff@θR@�AqBG�\                                    Bxy��&  �          A��@�  @ָR��33��HBJ��@�  @�z�?��HA4  BIp�                                    Bxy��  �          A��@��@�\)������BK=q@��@�?�Q�A0��BJQ�                                    Bxy�r  �          A��@���@�\)��\)��RBJ��@���@�z�?�G�A8��BI{                                    Bxy�  �          AG�@���@ָR��  ���BI��@���@�p�?��A*�RBI�                                    Bxy�-�  �          A��@�=q@����(��2�HBH=q@�=q@�
=?�
=AQ�BIff                                    Bxy�<d  �          A@���@��Ϳ����@z�BH�@���@أ�?��A  BJ��                                    Bxy�K
  �          A@��R@�=q��33�G�BM\)@��R@�\)?��A;
=BK�
                                    Bxy�Y�  �          A��@��@�ff��ff�!��BIG�@��@�{?�{A'\)BI{                                    Bxy�hV  �          A��@���@�ff���
���BI\)@���@��?���AB�\BG
=                                    Bxy�v�  T          A  @�z�@�(��p������BFp�@�z�@�33@Q�Aa�BA�                                    Bxy���  �          Az�@��\@љ��W
=����BA33@��\@�ff@0  A��B6=q                                    Bxy��H  �          Az�@��H@љ�        BA{@��H@�33@<(�A�z�B4=q                                    Bxy���  �          AQ�@�=q@�G�>�=q?޸RBA\)@�=q@�\)@K�A�{B2�                                    Bxy���  �          A��@��@���=�G�?:�HB@ff@��@�G�@B�\A��B2z�                                    Bxy��:  �          AQ�@�z�@�ff��
=�0��BG��@�z�@�@(��A�=qB>z�                                    Bxy���  �          A�@�\)@�z�z��qG�BN
=@�\)@��@%A�z�BE�                                    Bxy�݆  �          AG�@�p�@��ü��W
=BHQ�@�p�@��@A�A�ffB;��                                    Bxy��,  �          A�@���@�
=�\)�fffBEff@���@��@:=qA�p�B9�\                                    Bxy���  �          A@�  @׮���B�\BFG�@�  @�=q@<(�A�G�B:=q                                    Bxy�	x  �          A�@�=q@�zᾔz��33BL{@�=q@ȣ�@7
=A�Q�BA�\                                    Bxy�  �          A�@��\@�z�\)�c�
BL
=@��\@ƸR@@  A�(�B@33                                    Bxy�&�  �          A��@���@�z�=�Q�?(�BL�@���@�33@Mp�A�=qB>�
                                    Bxy�5j  �          A�@�p�@�׾.{����BQ33@�p�@ʏ\@A�A��
BE�                                    Bxy�D  �          A{@��@ڏ\��33�  BIff@��@Ǯ@2�\A��RB?(�                                    Bxy�R�  �          A{@�33@�(���=q��\BKQ�@�33@�  @8��A��
B@z�                                    Bxy�a\  �          A@���@���\)��BM@���@�G�@9��A���BC                                      Bxy�p  �          Ap�@�{@޸R�����BP
=@�{@��@<(�A�BE
=                                    Bxy�~�  �          A{@��@����\)��BL�R@��@�@EA�33B@{                                    Bxy��N  �          Aff@���@�ff>�33@G�BM�H@���@���@_\)A��HB=��                                    Bxy���  �          Az�@�@�R����'
=BYG�@�@��?�A<��BX�                                    Bxy���  �          A��@�Q�@�ff��p���RBW\)@�Q�@�\?�p�AJ�RBUz�                                    Bxy��@  �          A�
@��@�ff���H�/�BZ�\@��@�?��
A6�\BZQ�                                    Bxy���  �          A33@�
=@��ÿ�ff�\)B^�\@�
=@�?��HAJ�RB]{                                    Bxy�֌  �          A
=@��@�\��(���
B`�@��@�{@�
AT��B^��                                    Bxy��2  �          A�@�\)@�ff��\)�&�\Bfz�@�\)@�?��RAL��Be=q                                    Bxy���  �          A�@�33@����R���Bk�@�33@�\@=qAx��Bg��                                    Bxy�~  �          A  @��\@�{�u��33Bl�@��\@�Q�@,(�A�\)Bg(�                                    Bxy�$  �          A�
@�z�@�R��{��33Bc{@�z�@�(�@��A}��B^G�                                    Bxy��  �          A  @��@�{�O\)��ffB`�@��@�
=@-p�A�ffBY                                    Bxy�.p  �          A  @��@�p��Ǯ�\)B^�H@��@�Q�@E�A�z�BT��                                    Bxy�=  �          Az�@�ff@�(��#�
�}p�B[G�@�ff@��
@Q�A�{BOG�                                    Bxy�K�  �          AG�@�  @��ͽ��=p�BZ��@�  @��
@Tz�A�BN=q                                    Bxy�Zb  �          AG�@��
@�  �.{����B^��@��
@�
=@Tz�A��BR                                    Bxy�i  �          A��@�33@�
=�u�ǮB^��@�33@�z�@Z=qA��BQ�H                                    Bxy�w�  �          AG�@��R@�33>\)?h��Bc�@��R@��@j=qA�{BUff                                    Bxy��T  �          A{@�p�@�{>��R?���Bez�@�p�@��@vffA�\)BV(�                                    Bxy���  �          A{@��H@�=q=L��>�z�B`Q�@��H@�@c�
A��BR�R                                    Bxy���  
�          A=q@���@�>u?�  Ba�
@���@�(�@p��A�(�BR��                                    Bxy��F  �          A�@�@��>�{@Q�Be{@�@��
@w�A�
=BUp�                                    Bxy���  �          A�\@��@�  >�\)?�G�Bg�@��@�\)@w�A��
BX��                                    Bxy�ϒ  �          A@�z�@�p�>Ǯ@�Bf  @�z�@Ӆ@{�A�=qBU��                                    Bxy��8  �          Aff@���@�G�?
=q@VffBj(�@���@�z�@��A�BYG�                                    Bxy���  �          A{@�ff@���?&ff@��Bk@�ff@��H@�
=A�=qBZ{                                    Bxy���  �          A�@���@�G�?��@�33Boz�@���@���@�=qA�  B[                                      Bxy�
*  �          AG�@�33@�G�?G�@�Bm@�33@У�@��HA�Q�B[(�                                    Bxy��  �          A�@��@�{?
=@n{Bf�
@��@У�@�(�A�p�BU{                                    Bxy�'v  �          Ap�@��@�Q�?J=q@�Q�Bk��@��@�\)@��HA�Q�BY                                      Bxy�6  �          A��@�Q�@�33?\(�@�Bp�\@�Q�@���@�ffA�Q�B]z�                                    Bxy�D�  �          A{@�G�@�z�?(�@uBp=q@�G�@�p�@�  A�  B_                                      Bxy�Sh  T          A�\@��@�z�?(�@s33Bn@��@�p�@�  A�
=B]p�                                    Bxy�b  �          A
=@��R@�\)?^�R@��Bs�@��R@�(�@���A�  B`=q                                    Bxy�p�  �          A�R@�=q@��?B�\@���Bo�H@�=q@Ӆ@���A噚B]ff                                    Bxy�Z  �          A�\@���@�z�>���@Bb��@���@ҏ\@y��A�BRp�                                    Bxy��   �          A��@���@�  ��\)��(�B^
=@���@�z�@^�RA��BP��                                    Bxy���  �          A��@�G�@�=q���Ϳ�RBXp�@�G�@Ϯ@W�A�\)BK=q                                    Bxy��L  �          A��@�\)@�zΐ33���HBH�@�\)@��
@\)AeBD\)                                    Bxy���  �          AQ�@��@�Q쿗
=��Q�BDG�@��@�Q�@
=qA]��B@
=                                    Bxy�Ș  �          AQ�@�G�@������hQ�B9
=@�G�@��
?�  @�33B>��                                    Bxy��>  �          A�@�=q@�ff�\)�h(�B6\)@�=q@У�?u@��B<(�                                    Bxy���  �          A@��H@�Q��6ff��z�B-�@��H@�p�>aG�?�B:33                                    Bxy��  �          A{@�
=@�G��C�
���B0�@�
=@љ�=u>�Q�B>��                                    Bxy�0  T          A\)@�ff@\�333��  B6ff@�ff@�>�
=@,(�BA33                                    Bxy��  �          A��@��@���<(���  B8�R@��@�=q>�{@�BD=q                                    Bxy� |  �          A��@�z�@����Q����B7
=@�z�@�(��#�
�L��BE                                    Bxy�/"  �          A�@���@�Q��Dz���z�B3\)@���@�  >#�
?��
B@��                                    Bxy�=�  �          A��@���@Ǯ�L(���ffB<G�@���@�Q�>.{?��BIz�                                    Bxy�Ln  �          A�@�  @�z��^�R��  B1�
@�  @��H��  ��=qBB�                                    Bxy�[  �          Aff@�ff@�(��l(���(�B)  @�ff@ָR����]p�B=
=                                    Bxy�i�  �          A
=@�ff@�p��P����(�B%Q�@�ff@�G�����s33B5�\                                    Bxy�x`  �          A�
@��@���C�
����B.33@��@�\)>8Q�?�\)B;ff                                    Bxy��  �          A��@˅@Ǯ�*=q��33B1�H@˅@�Q�?&ff@\)B:�                                    Bxy���  �          Az�@�=q@���\)�x��B*�R@�=q@�Q�?8Q�@�B2��                                    Bxy��R  �          A�@��H@����%���RB0�@��H@�z�?.{@�ffB9G�                                    Bxy���  �          A�@�(�@��H���H�E�B3(�@�(�@�  ?���A\)B6�                                    Bxy���  �          A33@�\)@�녿�
=�(��B1
=@�\)@�33?�ffA\)B1��                                    Bxy��D  �          A�\@���@�=q�ٙ��*�HB2��@���@˅?�ffA�B3G�                                    Bxy���  �          A{@�ff@Ϯ��z��'�
B9\)@�ff@Ϯ?�z�A(  B9\)                                    Bxy��  �          Aff@ƸR@���Q��C�B8�@ƸR@�=q?�33Az�B:�\                                    Bxy��6  �          A{@�(�@�\)���H�F�RB:ff@�(�@��
?�33A��B<�                                    Bxy�
�  �          Ap�@Å@Ϯ���7\)B;  @Å@��?��A(�B<33                                    Bxy��  �          Ap�@�G�@�=q��=q�9p�B4�\@�G�@�p�?�Q�A�B6\)                                    Bxy�((  �          A�@�p�@�
=���733B033@�p�@�=q?�z�A=qB2{                                    Bxy�6�  �          A�\@�\)@��ÿ�ff��B0Q�@�\)@�  ?�A(z�B/                                    Bxy�Et  �          A�\@�(�@�z���
�B4G�@�(�@��H?�  A0  B3=q                                    Bxy�T  �          A�H@��
@�{��(���B5=q@��
@�33?���A7\)B3��                                    Bxy�b�  �          A(�@�33@��W
=��B1�@�33@���@Q�AnffB)��                                    Bxy�qf  �          A  @��@�(���ff���B9@��@�@z�AN�HB6G�                                    Bxy��  T          A33@�Q�@�{�aG�����B2�R@�Q�@��@Al(�B+�
                                    Bxy���  �          A�
@У�@�G��Ǯ��B4G�@У�@��@4z�A�Q�B(�R                                    Bxy��X  �          AQ�@θR@�z���O\)B7  @θR@��
@FffA�z�B(��                                    Bxy���  �          AQ�@ȣ�@ڏ\�L�;��
B=��@ȣ�@�  @QG�A�
=B.�                                    Bxy���  �          A�@˅@��>��?s33B;�@˅@�(�@\(�A�G�B+{                                    Bxy��J  �          A��@�@ָR>�\)?޸RB8�@�@�\)@`  A�33B&��                                    Bxy���  �          A��@��H@�G�?�@a�B333@��H@�{@k�A�ffB(�                                    Bxy��  �          AQ�@�\)@ʏ\?�R@u�B,��@�\)@�\)@g�A�  B\)                                    Bxy��<  �          A(�@��H@��?k�@�B'��@��H@�{@s33A��
B33                                    Bxy��  �          A��@߮@��?h��@��B#��@߮@�33@o\)A��B
��                                    Bxy��  �          A  @��@���?p��@�33B��@��@��H@h��A�G�B��                                    Bxy�!.  �          A\)@�z�@�z�?�z�A�B33@�z�@�\)@~�RA��HA�p�                                    Bxy�/�  �          A33@��@��?�  AffB{@��@���@��A�ffA���                                    Bxy�>z  T          A�\@��H@��?���@�\)B$��@��H@�ff@w
=A�p�B

=                                    Bxy�M   T          A
=@��H@˅?
=@mp�B/�
@��H@�Q�@g�A��BQ�                                    Bxy�[�  �          A�\@�@�  ?��@\��B5=q@�@���@j=qA�(�B �                                    Bxy�jl  �          A33@�(�@��
>�G�@0��B8(�@�(�@��@g�A�G�B$33                                    Bxy�y  �          A33@�ff@�\)?��\@˅B4z�@�ff@���@��A�G�Bz�                                    Bxy���  �          A�R@���@�ff?�Q�@�{B4��@���@���@�{AٮB{                                    Bxy��^  �          A=q@�ff@��
?���@�B2�@�ff@�  @�33A�p�B�                                    Bxy��  �          A=q@�ff@�p�?fff@��B3ff@�ff@���@{�A˅B�                                    Bxy���  �          A�R@��
@�\)?�p�@�\)B5�@��
@�G�@�  A��HBz�                                    Bxy��P  �          A�H@Ӆ@ȣ�?�\)@�G�B-��@Ӆ@��@���Aљ�B�                                    Bxy���  �          A(�@�\)@�Q�?�33@�z�B4�@�\)@��@�ffAי�B                                      Bxy�ߜ  �          A��@�z�@�z�8Q���\)BA�@�z�@˅@0  A�=qB7��                                    Bxy��B  �          A��@��@�Q��=q�5��BIz�@��@�Q�?���A7�
BI\)                                    Bxy���  T          A��@�  @�{�˅���BD��@�  @�=q@�AI�BB�R                                    Bxy��  �          A�@��H@�33�ٙ��((�BAp�@��H@ٙ�?��A:�\B@��                                    Bxy�4              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxy�(�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxy�7�   �          AG�@ƸR@޸R������\B@�@ƸR@�
=@HQ�A���B4�                                    Bxy�F&  �          Ap�@�ff@�\)���:�HBA�@�ff@�z�@U�A�G�B2�R                                    Bxy�T�  �          AG�@ʏ\@ڏ\���R���B<��@ʏ\@��H@FffA�G�B/��                                    Bxy�cr  �          AG�@�(�@��ÿ\)�\(�B:�
@�(�@�p�@6ffA�z�B0(�                                    Bxy�r  �          A��@��
@�\)�8Q���p�B1��@��
@��@$z�A��\B(��                                    Bxy���  �          A��@�\)@�z�Ǯ�Q�B.{@�\)@�  @3�
A��B"
=                                    Bxy��d  �          A��@���@�33>aG�?��B,�@���@�p�@Tz�A���B�                                    Bxy��
  �          Ap�@��@�
=�8Q쿎{BJ�
@��@��
@Z=qA�\)B<��                                    Bxy���  �          Ap�@��@��<#�
=L��BHQ�@��@�
=@b�\A�ffB8Q�                                    Bxy��V  �          A��@���@�{?G�@���B9�@���@�{@~�RA��HB!\)                                    Bxy���  �          A�@��@��H>�\)?��HB=33@��@��@g�A��B*�                                    Bxy�آ  �          A��@��@�ff���
��BA�R@��@\@W�A�{B2p�                                    Bxy��H  �          AQ�@�(�@�
=<�>aG�B9�@�(�@�=q@VffA�B)�                                    Bxy���  �          A(�@�(�@�ff>8Q�?�\)B9z�@�(�@�\)@^{A�=qB'��                                    Bxy��  �          A  @�G�@���=�Q�?
=qB<�@�G�@�33@[�A�(�B+��                                    Bxy�:  �          A�H@�  @�p�?.{@�  B)��@�  @���@i��A�p�B                                    Bxy�!�  �          A
=@�\)@�?8Q�@���B!Q�@�\)@�G�@dz�A���B	��                                    Bxy�0�  �          A�H@�{@�
=?=p�@�33B"��@�{@�=q@fffA��\B�                                    Bxy�?,  �          A33@��
@�p�?���@��
B33@��
@�(�@qG�A��A���                                    Bxy�M�  �          A�R@�R@�{?�
=@��B\)@�R@z�H@fffA�
=A��                                    Bxy�\x  �          A�@��@���?k�@��RB (�@��@�p�@n{A��\Bz�                                    Bxy�k  T          A(�@ۅ@�{?�\@K�B(Q�@ۅ@��
@aG�A���B
=                                    Bxy�y�  T          A�\@�z�@�?(��@�(�B��@�z�@�33@X��A�{B33                                    Bxy��j  �          A{@ᙚ@���>�@B�\B�@ᙚ@���@R�\A��B��                                    Bxy��  �          A�H@���@�  >L��?��\B{@���@�z�@A�A�=qB	Q�                                    Bxy���  �          A�@�(�@�Q�>aG�?��B�@�(�@�p�@;�A�ffB=q                                    Bxy��\  �          A  @�(�@�=q>��?��B�@�(�@�ff@@  A�G�B��                                    Bxy��  �          A
=@�@��>�Q�@�RB�@�@��
@L��A�
=B	�\                                    Bxy�Ѩ  �          Aff@�{@�{>�(�@,��B"Q�@�{@�{@U�A��RB�
                                    Bxy��N  �          A33@�z�@�G�>��
?�(�B=q@�z�@��
@J=qA��\B	(�                                    Bxy���  �          A�\@�z�@�p�?c�
@��HB�H@�z�@�
=@fffA���B Q�                                    Bxy���  �          Aff@�R@�=q?=p�@���B@�R@��R@Z�HA��A�                                      Bxy�@  �          A�H@�  @��?�@UB �R@�  @��@Z=qA�Q�B{                                    Bxy��  �          A
=@�z�@�  ?��@n�RBp�@�z�@�{@XQ�A��HB33                                    Bxy�)�  �          A�H@�\@�ff?fff@�33Bz�@�\@���@_\)A��HA�(�                                    Bxy�82  �          A
=@�Q�@��
>�@6ffB{@�Q�@���@L��A���B�\                                    Bxy�F�  �          A33@���@�33?
=q@VffB(�@���@��\@P��A��
B                                     Bxy�U~  �          A�
@�\)@�ff?J=q@�B��@�\)@���@a�A���B �\                                    Bxy�d$  �          A
=@��
@�  ?G�@�z�B��@��
@�33@c33A�Bff                                    Bxy�r�  �          A�@��@�ff?(��@��HB!�@��@��\@b�\A�ffB
{                                    Bxy��p  T          A��@��@�p�?=p�@�=qB'{@��@�\)@n{A���B{                                    Bxy��  �          Az�@�33@Ϯ?
=q@W�B2  @�33@�33@mp�A��RB{                                    Bxy���  �          A�@ڏ\@ə�?:�H@�Q�B*�
@ڏ\@�33@q�A��B�                                    Bxy��b  �          A��@��@Ϯ?�@K�B0�H@��@��@l(�A�z�BG�                                    Bxy��  �          A��@�=q@��?��@XQ�B3@�=q@��@p  A�Q�B��                                    Bxy�ʮ  T          A��@�@θR?(�@q�B0=q@�@�G�@p��A�z�B�\                                    Bxy��T  �          A@�\@\?:�H@��B"��@�\@���@j�HA���B
�R                                    Bxy���  �          A��@�\)@�G�?Tz�@��
B{@�\)@�(�@_\)A�p�A��H                                    Bxy���  |          AG�A (�@�ff?��@ƸRB��A (�@~�R@`  A��HAә�                                    Bxy�F  �          A�
A ��@��
?�33@�Q�B��A ��@��\@k�A��A�33                                    Bxy��  �          AffA z�@��R?��R@�z�B��A z�@x��@k�A�G�AΣ�                                    Bxy�"�  �          A33A��@�ff?��R@��HB��A��@xQ�@j�HA�  A��H                                    Bxy�18  �          A�H@�ff@���?�Q�@��B��@�ff@��\@n�RA�\)Aم                                    Bxy�?�  �          A�@�\)@�(�?��
@��HB��@�\)@���@s33A�ffA�                                    Bxy�N�  �          Aff@��@�ff?�  A\)BG�@��@p  @y��A��HA�33                                    Bxy�]*  �          A�\A�\@�p�?���A��A�z�A�\@^{@tz�A�ffA�=q                                    Bxy�k�  �          A{A��@��?�  A'\)A��
A��@XQ�@~{A��HA�
=                                    Bxy�zv  �          A�\A ��@���?�\A)�A�\)A ��@]p�@���A�(�A��
                                    Bxy��  T          A�A�\@���?ٙ�A"�\A��
A�\@S�
@w�A�  A�z�                                    Bxy���  �          A�A{@�ff?�=qA\)A��
A{@C�
@fffA�A�z�                                    Bxy��h  �          Ap�A@�?�ffA��A�33A@C33@c�
A�ffA��\                                    Bxy��  �          A��A33@�p�?˅AG�A�Q�A33@3�
@^{A�(�A�p�                                    Bxy�ô  �          AG�A
�\@l��?�\)A  A���A
�\@��@P  A�33Av�\                                    Bxy��Z  �          A
=@�
=@�  >�33@��B033@�
=@�ff@c�
A��HB33                                    Bxy��   �          A��@��@�?B�\@��B��@��@�  @hQ�A�
=B��                                    Bxy��  �          A��@��
@�p�?n{@���B
Q�@��
@�\)@aG�A�(�A�                                      Bxy��L  �          A=q@���@��
?u@�\)B�\@���@�(�@i��A�{A�p�                                    Bxy��  �          A�\@�
=@�  ?L��@��B��@�
=@��\@dz�A�A�\)                                    Bxy��  �          A�H@���@��?�  @�{B�
@���@�=q@s33A�
=A���                                    Bxy�*>  �          A
=@�@�
=?�{@�33B(�@�@�33@|��A��RA���                                    Bxy�8�  �          A\)@�{@�(�?�33@���B��@�{@�\)@��A�  B��                                    Bxy�G�  �          A�
@��@��?�{@У�B��@��@�
=@w�A�G�A�=q                                    Bxy�V0  �          A�
@���@���?z�H@�Q�B��@���@���@p  A�G�A�                                    Bxy�d�  �          A33@���@�G�?z�H@���B�@���@���@g�A�p�A�\                                    Bxy�s|  �          A�
@�ff@��R?�=q@��B
  @�ff@��@x��A�z�A�ff                                    Bxy��"  �          A\)@�@�
=?���A��B��@�@��@�
=A�z�A�\)                                    Bxy���  �          A33@�(�@�ff?�33@ٙ�B �@�(�@���@�33A�{B�H                                    Bxy��n  �          A�@���@�G�?�ff@�
=B��@���@�ff@|(�A���A��                                    Bxy��  �          A�@�p�@�
=?c�
@���B�@�p�@�@x��A���B�                                    Bxy���  �          A�
@�33@�  ?���@�=qB!z�@�33@��@�p�A�33B�                                    Bxy��`  �          A(�@�@���?ٙ�A ��BQ�@�@��@�  AۅA�                                    Bxy��  �          A�@�=q@�?�{A��B @�=q@�G�@��A�z�A�z�                                    Bxy��  �          A33@�Q�@���?�33Az�B#p�@�Q�@�\)@��AծB\)                                    Bxy��R  �          A�\@�G�@У�?��@�G�B+{@�G�@��@�Q�A��B�
                                    Bxy��  
�          A
=@�Q�@�=q?���@�33B,��@�Q�@��\@��\Aԣ�B��                                    Bxy��  �          A
=@���@ҏ\?�{@ҏ\B,Q�@���@�(�@�Q�AЏ\Bz�                                    Bxy�#D  �          A�H@�{@�p�?z�H@��\B/�@�{@���@�ffAͮB
=                                    Bxy�1�  �          A=q@�ff@��
?Q�@��
B?=q@�ff@��@���A���B&                                    Bxy�@�  �          A�R@�(�@�R?aG�@�ffBB  @�(�@���@�(�A�\)B)
=                                    Bxy�O6  �          A�\@�  @��?s33@��BE@�  @�33@�\)A��B,G�                                    Bxy�]�  �          Ap�@ƸR@�?h��@��RBE�@ƸR@��@�p�A��
B,G�                                    Bxy�l�  �          A=q@ȣ�@�R?��@�Q�BC��@ȣ�@�{@��A�z�B(�                                    Bxy�{(  �          A
=@�  @��?���A
=BE33@�  @�33@��HA�B'Q�                                    Bxy���  �          A\)@ə�@�?�z�A{BC@ə�@���@�33A�(�B%z�                                    Bxy��t  �          A\)@˅@�p�?��HA
{BA��@˅@�
=@��A�ffB"��                                    Bxy��  �          AQ�@�  @�33?��A�RB>  @�  @�=q@�\)A�33B{                                    Bxy���  �          A(�@�(�@�ff?У�A{BA��@�(�@��@���A��
B!33                                    Bxy��f  
(          A�@�p�@��
?У�A{B?�H@�p�@��H@�\)A�ffB(�                                    Bxy��  �          A  @�G�@���?ٙ�A z�B<G�@�G�@�\)@�  A��\B��                                    Bxy��  �          AG�@љ�@��@�AC�B;�H@љ�@���@�=qB{B
=                                    Bxy��X  �          A��@�{@��@AE�B>�@�{@��@��HBz�B�                                    Bxy���  �          A��@�\)@�33?�
=A5p�B>Q�@�\)@�p�@�\)B �
B\)                                    Bxy��  �          A�@�Q�@�p�?�A��B6�R@�Q�@���@�p�A�ffB=q                                    Bxy�J  �          AQ�@�@��?ٙ�A ��B/p�@�@��@��A�=qBG�                                    Bxy�*�  �          Az�@߮@�33?�A*�\B-G�@߮@���@��
A���B	�
                                    Bxy�9�  �          AG�@���@�33?��A1�B5=q@���@��@�=qA�ffBQ�                                    Bxy�H<  �          AG�@�@�Q�?�z�A�B9z�@�@��@�ffA��
BQ�                                    Bxy�V�  �          A��@�p�@�G�?�
=A��B:33@�p�@�  @�\)A��B�                                    Bxy�e�  �          A=q@��@�{?��
A&{B6
=@��@��@���A�  Bz�                                    Bxy�t.  �          A{@�(�@��?ǮA�B4�@�(�@�ff@�=qA�BQ�                                    Bxy���  �          A=q@��@���?���A�\B3��@��@�@�=qA�33Bp�                                    Bxy��z  �          Aff@��
@أ�?�@�G�B.Q�@��
@���@��A�p�B33                                    Bxy��   �          A�H@�p�@�\?�z�A1�B:@�p�@�p�@�ffA�
=B
=                                    Bxy���  �          A�@�\)@���@333A�  BR�H@�\)@��@�{B�\B'                                      Bxy��l  �          A��@�@�z�@A�A�p�BZz�@�@�33@�ffB#��B,�
                                    Bxy��  �          A  @,��A(�@z=qA�33B�@,��@��@���BN��B}�
                                    Bxy�ڸ  �          A�@,(�A��@r�\A���B�@,(�@�  @��BK��B��                                    Bxy��^  �          A33@B�\A(�@aG�A�=qB���@B�\@�33@陚BCz�Bv
=                                    Bxy��  �          A=q@UA�@K�A��
B��@U@�  @߮B9�RBoG�                                    Bxy��  �          A��@b�\A�\@>�RA���B�W
@b�\@�G�@�G�B4
=Bj=q                                    Bxy�P  �          A�R@H��Aff@j=qA��
B�  @H��@�{@��
BFG�Bp                                      Bxy�#�  �          A=q@o\)A�R@Z�HA��HB��H@o\)@��
@ᙚB<��B\��                                    Bxy�2�  �          A(�@�z�@�p�?�Q�A<��BJz�@�z�@��@�Q�B��B&�R                                    Bxy�AB  �          A��@�@�33@$z�A{�
BV{@�@��\@���BB,=q                                    Bxy�O�  �          A@�\)@��\@<(�A��\Bkff@�\)@��@θRB(�BA=q                                    Bxy�^�  �          A�R@�
=A Q�@VffA�z�Bx��@�
=@���@��B6��BMG�                                    Bxy�m4  �          A\)@y��AG�@l��A�\)B��=@y��@���@�BA�BS��                                    Bxy�{�  �          A
=@l��AG�@x��A�33B���@l��@��@���BG
=BW=q                                    Bxy���  �          A�@tz�Aff@l��A��HB�Ǯ@tz�@��R@��BA�\BW�                                    Bxy��&  �          A(�@mp�Aff@w
=A�=qB�  @mp�@�z�@��BEBX��                                    Bxy���  �          A�H@s33A
=@]p�A���B�8R@s33@��
@�\B<ffBZ��                                    Bxy��r  �          A=q@��
@�G�@QG�A���Bmff@��
@��
@�
=B1  B?�H                                    Bxy��  �          A�@�{@���@=p�A�  Bd�\@�{@��@�(�B'�B8�
                                    Bxy�Ӿ  �          A�@�=q@��
@\)Aq�BS�@�=q@���@��\B33B+                                      Bxy��d  �          A�@��
@���@+�A�B[�
@��
@���@�z�B�B2�\                                    Bxy��
  �          A�@�  @�ff@0  A�B^�
@�  @���@�
=B\)B5�                                    Bxy���  �          A�@�@��@1G�A��RB`�H@�@��\@�Q�B z�B7Q�                                    Bxy�V  �          A(�@�(�@��H@.�RA�=qBc33@�(�@�@ȣ�B �B:��                                    Bxy��  	�          A(�@�p�@��H@*=qA}p�Bb�@�p�@�
=@�
=B33B:��                                    Bxy�+�  
�          A  @��\@�  @#�
As\)B]�
@��\@�ff@�=qB33B6ff                                    Bxy�:H  T          A(�@��@�G�@�A\��B\�
@��@�33@���B�\B7�
                                    Bxy�H�  T          A
=@��
@�\@�HAf�HBM33@��
@���@��BG�B%�                                    Bxy�W�  
�          A
=@���@�@�AYG�BPz�@���@�=q@�p�BG�B*ff                                    Bxy�f:  "          A�R@�ff@�Q�@A_�BJ��@�ff@�z�@�z�B�\B#=q                                    Bxy�t�  "          Aff@�{@�  @�A`  BJ�R@�{@�(�@�(�B��B#G�                                    Bxy���  "          A�R@��H@�G�@{Am�BM=q@��H@�33@���B��B$�                                    Bxy��,  �          A\)@�(�@���@%�Av=qBL=q@�(�@�G�@�33BQ�B"Q�                                    Bxy���  �          A�R@���@�
=@#33Atz�BK  @���@�  @���BffB!33                                    Bxy��x  �          A�@��@��@'�A}�BA33@��@�ff@��RB�B\)                                    Bxy��  �          A{@�\)@ڏ\@)��A�
B>�@�\)@��
@�{B{B�                                    Bxy���  �          A�H@�33@��@*�HA�Q�BD  @�33@���@���BffB
=                                    Bxy��j  
�          Aff@��
@�\)@$z�Ax  BC  @��
@���@�ffB  B
=                                    Bxy��  
Z          AG�@�33@߮@
=Ac�
BC�\@�33@��@�Q�B��B33                                    Bxy���  
�          AG�@���@�R@   Ar=qBM�@���@���@�  Bz�B$                                      Bxy�\  �          A�@���@�{@
=AM�BLp�@���@�ff@�z�B
�
B'�                                    Bxy�  
�          A
=@��@��
@)��A�Q�BE33@��@��@�ffB��B��                                    Bxy�$�  &          A\)@�z�@�{@#33Az�\BF��@�z�@���@��B�B
=                                    Bxy�3N  
�          A�\@���@�{@'
=A��\BHp�@���@��@�ffB
=B
=                                    Bxy�A�  
�          A�@��@��H@%�A�
BE�@��@�p�@�(�B��B33                                    Bxy�P�  "          A��@�Q�@��@!�A{�BH�@�Q�@�  @��
B��B
=                                    
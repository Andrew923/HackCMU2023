CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230425000000_e20230425235959_p20230426021843_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-04-26T02:18:43.913Z   date_calibration_data_updated         2023-04-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-04-25T00:00:00.000Z   time_coverage_end         2023-04-25T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx{[\   
(          A=q@�p�@��@��BD33B*Q�@�p�>�  A (�Bt\)@L(�                                    Bx{[j�  �          Az�@��@p��@��HBO=qB)�R@������A Q�B{\)C�S3                                    Bx{[yL  T          A@k�@dz�@�\)BZ��B0ff@k�����A ��B��C�n                                    Bx{[��  T          A�@dz�@j=q@�  B[33B6�R@dzᾅ�AB�aHC��=                                    Bx{[��  �          A�
@fff@W�@�RB^z�B,ff@fff���H@�{B���C�)                                    Bx{[�>  �          A	�@{�@<(�@�=qB\��B�@{��O\)@��
BxC�33                                    Bx{[��  T          Az�@c�
@;�@�p�Be
=B��@c�
�\(�@�{B��C�:�                                    Bx{[  "          A=q@\)@/\)@��B33B>��@\)��Q�@��B��\C�9�                                    Bx{[�0  T          A�@Tz�@E@�(�Bf{B+�\@Tz�0��@��B���C�)                                    Bx{[��  �          A�\@hQ�@j�H@��HBO�RB5(�@hQ�=��
@�B��{?�
=                                    Bx{[�|  	�          A�H@��\@b�\@�G�BAffBG�@��\=�G�@�
=Bl33?�p�                                    Bx{[�"  -          Az�@�z�@aG�@��BB�
B
=@�z�=#�
@��Bl�?                                       Bx{\�  �          A�@�33@_\)@У�BIB!�@�33�#�
@�z�Bt  C��
                                    Bx{\n  �          A	@��
@xQ�@�=qB=
=B&G�@��
>Ǯ@���Bmff@���                                    Bx{\)  "          A
{@��
@g
=@�33BI=qB$�@��
<�@��Bu{>�Q�                                    Bx{\7�            A
=@��\@�  @�G�B033B+z�@��\?\(�@�\Bf\)A*{                                    Bx{\F`  
Z          A
=q@��H@�G�@�=qB3(�B%\)@��H?(��@�Q�Be�RA\)                                    Bx{\U  -          A	��@���@�z�@�B��B0�R@���?��R@߮BY��A�
=                                    Bx{\c�            A	�@��@�@�G�Bz�B/@��?���@�(�BU�A��                                    Bx{\rR  -          A  @���@��\@��RB)
=B!�
@���?^�R@޸RB\Q�A%                                    Bx{\��  �          A(�@��
@�G�@�B'�RB��@��
?\(�@�p�BY�
A�
                                    Bx{\��  �          AQ�@��@�@�33B$ffB"�H@��?��
@��BY(�A>�R                                    Bx{\�D  �          A  @���@��R@�z�B&
=B%p�@���?��
@޸RB[��ABff                                    Bx{\��  
Z          A�\@�(�@��@���B(��B'@�(�?}p�@�ffB^�
A@Q�                                    Bx{\��  
�          AQ�@�(�@�33@��B&�B3{@�(�?�(�@���Bb\)Ay�                                    Bx{\�6  �          A�R@�(�@��@�ffB  B>��@�(�?�33@�z�BY(�A�33                                    Bx{\��  �          @�{@�@��\@�\)B\)B7�H@�?�ff@ʏ\BS�HA��\                                    Bx{\�  
�          A (�@���@�{@��B�B>{@���?�=q@�  BY=qA���                                    Bx{\�(  
�          A�@�G�@���@��HBB@�@�G�?�Q�@���BX�
A�Q�                                    Bx{]�  �          AG�@���@�Q�@�=qB	�BB�H@���@��@�z�BQ(�A�G�                                    Bx{]t  I          A ��@�Q�@��R@�B�RB>�@�Q�@�@�  BL
=A���                                    Bx{]"  _          @�@���@��R?�
=A�BJ��@���@k�@��B�B#
=                                    Bx{]0�  "          @���@aG�@�=q�7����
B]  @aG�@�=q�\)����Bk�                                    Bx{]?f  �          @���@@��@�Q��	����{By�\@@��@��H?Q�@�p�B~�R                                    Bx{]N  
�          @�G�@(��@�p��J=q��G�B�\@(��@ȣ׾k����HB�W
                                    Bx{]\�  �          @�@K�@�z��#�
����Bv�\@K�@���?�\@�33B~\)                                    Bx{]kX            @�33@n�R@��\�(����\B`��@n�R@��R?0��@�z�Bg�R                                    Bx{]y�  "          @�(�@o\)@�p���z���p�B]p�@o\)@�?\(�@�
=Bbz�                                    Bx{]��  
�          @�\)@o\)@�zῺ�H�Ip�B]  @o\)@�{?��RA*{B^�                                    Bx{]�J  I          @�
=@g
=@�p���(��R=qB\{@g
=@�  ?���A z�B]�
                                    Bx{]��  _          @�z�@c33@��ͿW
=���
BX(�@c33@�
=?ǮAl��BT
=                                    Bx{]��  
�          @�ff@�  @�33�k���Q�BT�@�  @��\@�
A�BI�\                                    Bx{]�<  "          @ٙ�@��@��>.{?�
=BS(�@��@�ff@,(�A�Q�BC\)                                    Bx{]��  T          @��@��@��>��?��\BS33@��@��R@*�HA���BC�                                    Bx{]��  
�          @�{@}p�@�(���33�AG�BV�\@}p�@�p�@p�A�{BL��                                    Bx{]�.  �          @�z�@z=q@�33���R�,(�BW\)@z=q@��
@�RA�
=BM
=                                    Bx{]��  I          @�(�@u�@�(���(��p  BZ�@u�@��R@Q�A�Q�BQ(�                                    Bx{^z  
�          @Ϯ@S�
@��������(��Bl@S�
@�G�@�A��HBb��                                    Bx{^   
�          @�{@X��@�ff��p��U�Bh�\@X��@�  @{A�33B_ff                                    Bx{^)�  
�          @�  @���@���?E�@ҏ\BR��@���@��\@J=qA�=qB;��                                    Bx{^8l  "          @�Q�@�=q@�
=?@  @�z�BN��@�=q@��@Mp�A�=qB8p�                                    Bx{^G  �          @�\@�G�@��H?&ff@�  BQ��@�G�@�z�@K�A�p�B={                                    Bx{^U�  �          @���@�
=@�Q�>�@j=qBW
=@�
=@�(�@E�A�ffBD��                                    Bx{^d^  T          @�G�@�{@��?B�\@�\)BS�@�{@�=q@P��A��
B=��                                    Bx{^s  �          @�z�@�  @�?�An�HBO��@�  @|��@���BQ�B+�R                                    Bx{^��  T          @�p�@�p�@���@
=qA�Q�BQ(�@�p�@q�@�p�Bz�B(��                                    Bx{^�P  "          @�z�@���@�ff@�RA��
BV  @���@r�\@�Q�B��B,�H                                    Bx{^��  T          @��
@\)@�Q�@A���BX�@\)@z=q@��B(�B1�                                    Bx{^��  �          @��@���@��R@��A��BV@���@r�\@�G�B�\B-�                                    Bx{^�B  �          @��
@�G�@�=q@(�A�Q�BS
=@�G�@e@�(�B"=qB&G�                                    Bx{^��  
�          @�  @}p�@�  @L(�A�Q�BH�R@}p�@.�R@���B9  B
p�                                    Bx{^َ  �          @޸R@�Q�@�p�@N{A��HBE=q@�Q�@)��@�Q�B8B�R                                    Bx{^�4  �          @��
@\)@��
@K�A��BD�@\)@(Q�@�ffB8{Bp�                                    Bx{^��  �          @�z�@��R@�=q@#�
A�Q�BCff@��R@E@��B"��B33                                    Bx{_�  
�          @�@��\@�33@333A��RBG�R@��\@@��@�
=B+�
B�                                    Bx{_&  "          @�@�\)@�Q�@  A�\)BG(�@�\)@Z=q@���B�BQ�                                    Bx{_"�  �          @��@���@��\@-p�A���B8��@���@4z�@�Q�B#G�B��                                    Bx{_1r  
�          @�(�@�33@�=q@Dz�AԸRB@p�@�33@(��@��\B2��B=q                                    Bx{_@  
�          @�p�@��@��H@I��A���B@��@��@'�@��B4�
BG�                                    Bx{_N�  T          @�ff@��@z�H@\)B\)B.�@��?�33@�33BIz�A���                                    Bx{_]d  �          @�\)@�=q@\)@qG�B�RB*�@�=q?�@�B?��A�{                                    Bx{_l
  �          @޸R@�(�@�33@s�
B�B3
=@�(�?��@���BD�HAĸR                                    Bx{_z�  �          @���@�33@r�\@q�B�
B$p�@�33?��@�33B>��A�G�                                    Bx{_�V  �          @�z�@���@l��@hQ�A��HB�H@���?�\)@�p�B6��A��                                    Bx{_��  
�          @ٙ�@�@y��@XQ�A�=qB%=q@�?�@�G�B2��A�33                                    Bx{_��  
Z          @�Q�@�@l��@J=qA߮B\)@�?�@�Q�B'�A��                                    Bx{_�H  "          @�ff@���@�z�@:=qA�(�B=�
@���@%�@�33B/p�B\)                                    Bx{_��  �          @��@hQ�@��H@^�RA���BHff@hQ�@G�@��\BGffB =q                                    Bx{_Ҕ  �          @�z�@c�
@�{@C�
A��
BS�@c�
@1G�@��B<�B�R                                    Bx{_�:  
�          @�z�@n{@�z�@R�\A�Q�BF�@n{@��@�B@33B\)                                    Bx{_��  	�          @�z�@z=q@�@A�A�
=BBQ�@z=q@#�
@��RB4�B{                                    Bx{_��  T          @Ӆ@s33@�@FffA�\)BEz�@s33@"�\@���B8�B�H                                    Bx{`,  
�          @���@w
=@���@S�
A�p�B?p�@w
=@�\@�z�B=A�                                    Bx{`�  T          @�(�@x��@��H@_\)A�{B9�H@x��@�
@�
=BA��A�33                                    Bx{`*x  
�          @�z�@xQ�@}p�@j=qB�B6ff@xQ�?�{@��BF��A�p�                                    Bx{`9  �          @��
@{�@j=q@w
=B�B,  @{�?�G�@�33BI��A�z�                                    Bx{`G�  
�          @θR@�@���@X��B��B~��@�@.{@�{B\�RBD{                                    Bx{`Vj  �          @��@W�@�G�@i��B{BH�@W�?���@��\BR33A�                                      Bx{`e  
Z          @�  @]p�@�
=@c�
Bz�BJ��@]p�@	��@��\BM�A�G�                                    Bx{`s�  "          @Ϯ@^�R@}p�@q�B�RBB�R@^�R?���@��BR��A�
=                                    Bx{`�\  
�          @�p�@^{@�33@`  BffBF��@^{@�@�
=BK�\A�33                                    Bx{`�  
�          @�@g�@~{@aG�B��B>�\@g�?��H@�BH�A���                                    Bx{`��  
Z          @�z�@j�H@|(�@[�B�
B<33@j�H?�p�@��HBD�A�ff                                    Bx{`�N  
Z          @�=q@X��@xQ�@h��B
=BC\)@X��?���@�  BQ�\A�(�                                    Bx{`��  T          @���@=p�@y��@���B��BSG�@=p�?�@��HBf33A�{                                    Bx{`˚  
�          @�p�@7
=@p  @|��B�BR��@7
=?���@��RBg�A��H                                    Bx{`�@  �          @\@E�@0��@�z�B:��B'�@E�?   @��BkG�A�                                    Bx{`��  
�          @Å@:�H@0��@�=qBB�\B-��@:�H>�
=@�=qBs�A
=                                    Bx{`��  T          @��@8Q�@�R@��BQ��B��@8Q��@��Bt��C��=                                    Bx{a2  �          @�Q�@=q@(Q�@��\BVG�B=��@=q>B�\@�  B��=@�
=                                    Bx{a�  
�          @��R@(�@1�@��BN
=BC  @(�>��@��B���A�                                    Bx{a#~  �          @�ff@�@?\)@���BH33BN�\@�?#�
@��B�=qAr�\                                    Bx{a2$  �          @�@�@:�H@�p�BP
=BUG�@�?�@�\)B��A\                                      Bx{a@�  T          @�?�(�@:=q@��B\�
Bm�?�(�>�(�@���B��Ab�R                                    Bx{aOp  "          @��R?��@8��@���B\�Bh�
?��>��@�B��ALz�                                    Bx{a^  �          @���?�z�@Mp�@�
=BN�Bl��?�z�?E�@���B��)A��                                    Bx{al�  
Z          @���?�{@\��@��BE�Bv��?�{?���@��
B�33A�
=                                    Bx{a{b  T          @�Q�?\@HQ�@�z�BZffB�=q?\?!G�@���B�8RA�z�                                    Bx{a�  T          @�z�?�(�@)��@���Bl=qBs�
?�(�>�@�ffB���@��                                    Bx{a��  T          @�Q�?�G�@Z=q@��
BI
=B{�?�G�?�G�@���B��A�(�                                    Bx{a�T  �          @�Q�?�@U@�{BM��B}��?�?k�@�{B��)A�z�                                    Bx{a��  T          @�  ?�=q@Mp�@��BUz�B
=?�=q?@  @��B�\A��H                                    Bx{aĠ  �          @�ff?��@E@�
=BR(�Bj��?��?0��@�33B���A�=q                                    Bx{a�F  T          @�?޸R@E@��BT�
Br�H?޸R?.{@��
B�L�A��                                    Bx{a��  �          @���?ٙ�@B�\@�Q�BW�Bsz�?ٙ�?!G�@��B�=qA�                                      Bx{a�  "          @��H?�(�@I��@��BPQ�Bu�?�(�?L��@���B��Aǅ                                    Bx{a�8  "          @��\?�z�@8Q�@�G�B\�HBp\)?�z�>�@��B��=A���                                    Bx{b�  T          @��\?�@<��@�{BVffBj
=?�?
=@�Q�B���A�=q                                    Bx{b�  T          @�Q�@z�@#33@�33BT��B?(�@z�>�\)@�  B��@���                                    Bx{b+*  
�          @���@@@��@��HBB�RBP��@?L��@�
=B�G�A��
                                    Bx{b9�  �          @���@{@K�@�  B>Q�B\�\@{?�  @��RB��A\                                    Bx{bHv  �          @�=q?�=q@N�R@�
=BX��B�\?�=q?W
=@���B��
Bp�                                    Bx{bW  �          @�G�?�  @N�R@�z�BU�B��3?�  ?^�R@��\B�u�B�                                    Bx{be�  �          @���?�p�@P��@�Q�BNG�B�{?�p�?u@�\)B�G�B�H                                    Bx{bth  T          @�  ?��
@QG�@���BQ\)B�G�?��
?u@���B��3Bff                                    Bx{b�  �          @�\)?�@QG�@�G�BR\)B��\?�?xQ�@���B�8RB�
                                    Bx{b��  T          @�?xQ�@N{@�=qBVB�W
?xQ�?h��@���B��fB+�
                                    Bx{b�Z  T          @��
?�33@K�@�\)BS�
B��?�33?h��@�p�B��{B��                                    Bx{b�   T          @�z�?�  @U�@�(�BKz�B���?�  ?���@�z�B���B&                                      Bx{b��  �          @��?��\@XQ�@�G�BG�B�Ǯ?��\?�
=@��\B��fB+��                                    Bx{b�L  �          @�?��@_\)@�G�BD33B��)?��?��
@�(�B�aHB1                                    Bx{b��  "          @��R?�ff@Tz�@��
BG��B��?�ff?�{@��
B�
=Bp�                                    Bx{b�  T          @�
=?��@g
=@c�
B$��By�?��?�(�@���Bw�RB)p�                                    Bx{b�>  �          @�\)?���@��@z�A���B��R?���@S�
@xQ�B6�Bs33                                    Bx{c�  �          @�  ?˅?�\)@��\B�B"�?˅���@���B���C�q                                    Bx{c�  �          @���?�z�>�ff@�33B�L�A�z�?�z��   @�  B��C��q                                    Bx{c$0  
�          @�G�?��?z�H@��B�.B��?����  @�{B���C���                                    Bx{c2�  �          @�  ?��?�=q@�\)B��B��?������@��B��fC��                                    Bx{cA|  "          @��R?@  ?��@�\)B��3B�8R?@  ���@���B��
C��R                                    Bx{cP"  �          @�Q�?���?�{@��B��BJp�?��ÿW
=@�G�B�\)C���                                    Bx{c^�  �          @���?�=q?�  @���B�ǮB@
=?�=q�+�@�Q�B�p�C�u�                                    Bx{cmn  �          @�  ?�  ?�G�@�B��
B5(�?�  ��R@�{B�\)C�/\                                    Bx{c|  T          @�  ?�ff?�@�
=B���B_\)?�ff���H@�G�B��qC���                                    Bx{c��  "          @�Q�?
=@z�@��B��
B��R?
=��@�\)B�
=C��R                                    Bx{c�`  �          @���?�  @�@�ffB���B���?�  ���@�p�B��
C��R                                    Bx{c�  "          @�=q?��@�@�ffB}�\B���?����\)@��RB�ffC�"�                                    Bx{c��  �          @���?��
@Q�@��BxBw�?��
���
@�z�B�� C��                                     Bx{c�R  �          @���?��H@�@�33Bw�HB~33?��H=#�
@���B���?ٙ�                                    Bx{c��  �          @���?�
=@%@���Bq�\B��?�
=>aG�@���B��fA*{                                    Bx{c�  T          @�  ?�p�@.{@��BjB�z�?�p�>Ǯ@�33B��A�=q                                    Bx{c�D  �          @�?�p�@#�
@�p�Bp{B��R?�p�>u@�G�B��\A/
=                                    Bx{c��  �          @�p�?��H@  @���B|�
Bv�?��H����@���B�B�C��                                    Bx{d�  �          @�ff?�  @\)@�p�Bq�HB}�?�  >8Q�@���B�33A                                    Bx{d6  "          @���?�(�@
=q@��B��Bq�
?�(��8Q�@�  B��C�޸                                    Bx{d+�  �          @��
?�  @1�@���BiG�B�� ?�  ?�\@�Q�B�
=A׮                                    Bx{d:�  �          @��?n{@;�@�{Bb��B��?n{?0��@��B�8RB��                                    Bx{dI(  �          @��
?�
=@�@�
=By(�B|��?�
=<�@��B�?�{                                    Bx{dW�  
�          @��
?fff@p�@�
=By
=B��
?fff>\)@�G�B�\)A�R                                    Bx{dft  T          @���?��@0��@��BiB���?��>��H@���B��Aƣ�                                    Bx{du  "          @�p�?��@!�@�{Bs  B��=?��>k�@���B�ffA4(�                                    Bx{d��  T          @�{?h��@!�@�Q�Bw(�B���?h��>W
=@��B�8RANff                                    Bx{d�f  "          @�{?s33@.�R@���Bm�RB��R?s33>�G�@��HB���A��
                                    Bx{d�  �          @�?s33@:�H@���BdB��
?s33?+�@���B�G�Bff                                    Bx{d��  �          @�p�?^�R@B�\@��RB`\)B���?^�R?O\)@�G�B�ǮB+
=                                    Bx{d�X  �          @�?�=q@Mp�@���BUz�B��?�=q?��
@�
=B��{B.                                    Bx{d��  _          @�?��@N{@���BU�\B�L�?��?��@�\)B���B4z�                                    Bx{dۤ  T          @�z�?��
@N�R@�  BT(�B���?��
?���@�B�ffB8��                                    Bx{d�J  �          @�(�?��\@^{@��BH�B�8R?��\?�\)@��B���BU�\                                    Bx{d��  �          @���?xQ�@S�
@�
=BQz�B�k�?xQ�?�@�{B��HBIQ�                                    Bx{e�  
�          @��
?Q�@7
=@��Be��B��H?Q�?.{@�B�ǮB�                                    Bx{e<  T          @��
?Y��@�
@���B�B��\?Y��<�@�G�B�� @p�                                    Bx{e$�  
�          @��?��\@S�
@���BO(�B�?��\?��H@��
B�8RBG(�                                    Bx{e3�  
�          @�z�?�33@�p�@K�BQ�B�=q?�33@\)@��Bl��B���                                    Bx{eB.  
�          @�=q?��@j�H@l(�B0(�B��q?��?�=q@��
B�{Bo�\                                    Bx{eP�  �          @�\)?��@p  @_\)B'ffB�z�?��?��R@�
=B���Bx                                      Bx{e_z  �          @��
?#�
@C33@�33BYB�ff?#�
?�  @�
=B�p�Be�                                    Bx{en   �          @���?W
=@U@�Q�BEz�B���?W
=?�z�@���B��
Bl{                                    Bx{e|�  I          @�?n{@G�@�  BKp�B��
?n{?�(�@�B�z�BRz�                                    Bx{e�l  �          @��?��@9��@�(�BUG�B�W
?��?u@��RB�
=B*ff                                    Bx{e�  "          @�ff?fff@B�\@��
BQ�
B�{?fff?���@�  B�W
BJ�H                                    Bx{e��  
�          @���?u@C33@�G�BN�B�33?u?�33@�B�L�BH��                                    Bx{e�^  _          @��\?�z�@I��@n{B<�B��)?�z�?�\)@�B�k�B1                                      Bx{e�  "          @�ff?��@p  @N{B��B�aH?��@�@��RBn33BXQ�                                    Bx{eԪ  T          @�  ?�{@<��@�33BN33B��\?�{?��@�{B��RB�                                    Bx{e�P  �          @�ff�W
=?��@���B�B�B�Q�W
=�5@�G�B�k�C[�q                                    Bx{e��  �          @�  ��
=?�  @�p�B�� C:ῗ
=�W
=@�  B�ǮCW��                                    Bx{f �  I          @�{���?�  @�33B���B�Ǯ��;�33@�z�B��\CT޸                                    Bx{fB  -          @�
=���
?��@�
=B��{B�녽��
�
=@�p�B���C�.                                    Bx{f�  �          @���aG�?�
=@�ffB�33B�{�aG���ff@��RB�  Cs�                                    Bx{f,�  �          @���\?���@�(�B��B�uþ\���@�
=B�ǮCVz�                                    Bx{f;4  �          @��þ�\)?�(�@�z�B���B�p���\)���@���B�  CO�                                    Bx{fI�  �          @���>��?�33@�{B��qB��q>���k�@�G�B���C���                                    Bx{fX�  �          @�G�>.{@G�@��
B�ǮB�\)>.{��Q�@���B�ǮC��                                    Bx{fg&  �          @�33=#�
@��@���B}�HB��=#�
>�\)@��HB��HB�                                    Bx{fu�  T          @��=�\)@�@��B��
B��=�\)<#�
@��B�p�@��                                    Bx{f�r  �          @�z�.{@G�@�(�B��B�k��.{>��@�(�B�G�C�                                    Bx{f�  T          @������@'
=@��BtffB�aH����?�@�(�B���B���                                    Bx{f��  T          @���u@{@���B{(�B��3�u>�Q�@�z�B�33B�(�                                    Bx{f�d  
�          @�ff    @33@�B���B�.    >#�
@�{B���B��{                                    Bx{f�
  T          @�{=�G�?�p�@��B�L�B�=�G��B�\@�B�z�C�\                                    Bx{fͰ  "          @�zὣ�
@��@��B}�RB��)���
>��R@��B�#�B��                                    Bx{f�V  �          @��\�L��@+�@�33Bn�B�p��L��?&ff@���B�B�Bը�                                    Bx{f��  T          @�=q�#�
@.�R@��Bl�B����#�
?8Q�@���B��B̸R                                    Bx{f��  �          @�G��#�
@1�@��Bh�HB�Ǯ�#�
?J=q@�\)B��B�33                                    Bx{gH  I          @�
=>�=q@�\@���B~\)B���>�=q>�z�@�p�B�W
B;\)                                    Bx{g�  �          @���>��H@��@�\)B�HB�W
>��H>aG�@��B�.AÙ�                                    Bx{g%�  �          @�{>Ǯ@��@��BQ�B��>Ǯ>��@�p�B��qB�R                                    Bx{g4:  �          @�\)=�G�@$z�@���Br�B��f=�G�?
=@�{B���B�                                      Bx{gB�  �          @�  =u@'�@�G�Bo�
B�z�=u?&ff@�ffB�B�Ǯ                                    Bx{gQ�  �          @�ff>W
=@)��@�
=Bm{B�>W
=?333@���B���B�
=                                    Bx{g`,  
�          @���>��R@p�@�  BtB��>��R?�\@��B���Bl33                                    Bx{gn�  
2          @�z�>���@#�
@�{Bo��B���>���?!G�@��HB�Q�B�
=                                    Bx{g}x  �          @��
>�=q@(Q�@�(�Bk��B�L�>�=q?:�H@��B��\B�W
                                    Bx{g�  �          @�?�@,(�@�(�Bg��B���?�?J=q@��\B��B`�                                    Bx{g��  T          @��?G�@6ff@�
=B\�B�ff?G�?�  @�  B��\BP
=                                    Bx{g�j  "          @��?�=q@%�@�  Bb�B��\?�=q?=p�@��B�.B	��                                    Bx{g�  
�          @�{>aG�@(��@��RBm=qB�G�>aG�?8Q�@�(�B�Q�B�G�                                    Bx{gƶ  �          @��H?   @B�\@�33BZ��B�(�?   ?���@�{B�\B�\                                    Bx{g�\  "          @�G�?.{@J=q@�{BQp�B��q?.{?��@��\B���Bzp�                                    Bx{g�  �          @�33?k�@XQ�@�G�BD33B�u�?k�?�=q@���B�\Bn��                                    Bx{g�  �          @�(�?�\)@i��@q�B2�HB��R?�\)?�@�z�B���Bn�
                                    Bx{hN  J          @��\?�z�@r�\@c�
B'��B���?�z�@��@��B{{Bu�H                                    Bx{h�  ,          @�Q�?��@z=q@N�RB��B��R?��@Q�@�
=Bi\)Bn��                                    Bx{h�  �          @�(�?O\)@g�@aG�B.
=B��3?O\)@ ��@�(�B��fB��                                    Bx{h-@  �          @�  ?�@S�
@l(�B?
=B�p�?�?�33@�p�B��B�                                      Bx{h;�  �          @��>�(�@c33@UB,33B�{>�(�@G�@�{B��B�                                    Bx{hJ�  �          @�Q�?��@��?�=qA�B��
?��@O\)@S33B0z�B��{                                    Bx{hY2  �          @�
=?���@��?n{A6ffB��?���@n�R@!�B  B�.                                    Bx{hg�  �          @�
=?�=q@�  >�33@�\)B���?�=q@���@G�A�G�B�p�                                    Bx{hv~  �          @��?�z�@c33���H���B��q?�z�@��þ����
B�u�                                    Bx{h�$  �          @�{?��@C33� ���Q�B��?��@qG���
=���RB���                                    Bx{h��  �          @�33?��@j=q���H��Q�B�u�?��@�(���G���33B�(�                                    Bx{h�p  �          @�ff?s33@�Q�>W
=@,��B��q?s33@w�?��A��B�L�                                    Bx{h�  �          @�ff?��@��?ǮA��RB�8R?��@Vff@B�\B'\)B�W
                                    Bx{h��  �          @�33?.{@�Q�?�=qA�\)B��f?.{@e@I��B#=qB���                                    Bx{h�b  T          @��\>�  @k�@(Q�B��B�8R>�  @�@s�
Be{B�B�                                    Bx{h�  �          @�(�>L��@g�@P  B'z�B��\>L��@	��@��B~��B�.                                    Bx{h�  
�          @�  =���@g
=@[�B.�B��=���@�@���B���B���                                    Bx{h�T  T          @��H>W
=@^{@l��B;{B�>W
=?���@�
=B��B���                                    Bx{i�  �          @�p�>��R@O\)@\)BK�B�  >��R?��
@��B���B�                                      Bx{i�  T          @�p�?.{@e�@i��B4(�B�aH?.{?�p�@�
=B�(�B��
                                    Bx{i&F  �          @��>�@g�@h��B3��B�  >�@G�@��RB�z�B�u�                                    Bx{i4�  �          @�?�@U@z�HBD�
B�\)?�?�z�@��
B��B�aH                                    Bx{iC�  �          @��>��@`��@p  B:�RB���>��?��@���B���B��q                                    Bx{iR8  �          @���>�@hQ�@g
=B2�B��>�@�
@�{B�� B��                                    Bx{i`�  "          @��
?B�\@i��@_\)B,z�B���?B�\@�@��HB�\B��3                                    Bx{io�  �          @��
?&ff@Z�H@mp�B;ffB�u�?&ff?���@��RB�G�B�                                    Bx{i~*  "          @��
?\)@U�@u�BBG�B�\?\)?ٙ�@���B��HB��)                                    Bx{i��  "          @�z�>��R@c�
@k�B7Q�B���>��R?�p�@�\)B��B�                                      Bx{i�v  �          @�(�?��
@���@3�
Bp�B�{?��
@5@�(�BWG�B��                                    Bx{i�  �          @�33?�(�@`��@^{B,(�B���?�(�@ ��@�Q�By��Bj�                                    Bx{i��  �          @��H?���@e@[�B)z�B�?���@ff@�  BxG�Bv��                                    Bx{i�h  �          @��\?�ff@b�\@J�HBG�B|(�?�ff@	��@�\)Ba�
BHp�                                    Bx{i�  �          @�=q?���@aG�@W
=B&p�B�W
?���@z�@��Br33Ba��                                    Bx{i�  �          @�=q?(��@fff@^�RB.(�B�(�?(��@
=@���B�8RB�
=                                    Bx{i�Z  �          @��\>\@g�@a�B0��B���>\@ff@�33B�33B�u�                                    Bx{j   �          @���>�G�@l��@W
=B(=qB�L�>�G�@\)@�
=B{�B���                                    Bx{j�  �          @�?�@c�
@UB+�\B�L�?�@�@���B~33B�\)                                    Bx{jL  �          @�?�p�@^�R@L(�B#\)B��?�p�@
=@�
=Bo��Bnz�                                    Bx{j-�  �          @��?��
@`  @P��B'33B�B�?��
@ff@���Bu�B��                                    Bx{j<�  �          @�z�?��@i��@.{B�BG�?��@(�@uBO{BV�                                    Bx{jK>  �          @���?�ff@Y��@U�B,G�B���?�ff@   @�=qBy�
Bx�
                                    Bx{jY�  T          @�z�?z�H@X��@W
=B.ffB�Ǯ?z�H?�p�@�33B|p�B~z�                                    Bx{jh�  �          @���?�(�@]p�@Mp�B$B�\?�(�@ff@�\)Bp��Bn�                                    Bx{jw0  �          @��?�=q@dz�@EB(�B�
=?�=q@  @��BgffBm(�                                    Bx{j��  �          @�{?�p�@\��@S�
B(�\B��3?�p�@33@��Bs�BlG�                                    Bx{j�|  �          @�{?��@R�\@h��B>  B��?��?��@��B�Q�B�\                                    Bx{j�"  �          @��R>\)@Dz�@x��BN�RB��>\)?�G�@�\)B��B���                                    Bx{j��  T          @�Q�=�Q�@K�@w�BJ\)B��)=�Q�?�\)@��B�L�B��
                                    Bx{j�n  T          @�G����
@B�\@�  BR\)B�����
?�Q�@��B�ǮB�(�                                    Bx{j�  �          @�녿z�@)��@�Q�Be�
B��)�z�?}p�@�p�B���B��                                    Bx{jݺ  �          @�녿!G�@4z�@���B\B�{�!G�?�Q�@��
B�(�B뙚                                    Bx{j�`  "          @��þ�ff@5@�(�B\�B�
=��ff?�(�@��B�\B܏\                                    Bx{j�  �          @��þ\@:=q@��HBY(�B��f�\?��@�33B���B�u�                                    Bx{k	�  T          @��׿�\@8Q�@�=qBY33B���\?��@�=qB���B��                                    Bx{kR  T          @��׿   @8��@��BX��B��   ?�ff@��B���B�G�                                    Bx{k&�  T          @�Q�(��@7
=@���BXQ�B�{�(��?��
@�G�B��B�3                                    Bx{k5�  �          @�Q�L��@C33@x��BL  B�p��L��?\@�ffB���B�                                    Bx{kDD  �          @�  �&ff@P  @n�RBA�Bʽq�&ff?�\@�(�B��RBܨ�                                    Bx{kR�  "          @�G����@P��@r�\BC��B�����?�G�@�B�{Bֳ3                                    Bx{ka�  "          @�G��(�@=p�@���BTz�B�33�(�?�33@���B���B�                                    Bx{kp6  
�          @�=q��  @&ff@�\)Bb\)B��
��  ?�  @��B��C��                                    Bx{k~�  
�          @�G��k�@(�@��Bk�B݊=�k�?O\)@�z�B�C
�                                    Bx{k��  �          @��׿\(�@!�@��Bg(�Bٞ��\(�?n{@�33B��qCٚ                                    Bx{k�(  �          @�  �8Q�@"�\@��Bh=qBӞ��8Q�?s33@�33B�u�B��\                                    Bx{k��  �          @�
=�
=@'�@�p�Be�B�z�
=?��@�=qB��B�G�                                    Bx{k�t  �          @�ff�8Q�@&ff@�z�Bc�B���8Q�?��@���B�Q�B��R                                    Bx{k�  �          @�ff�(�@7�@~�RBV=qB��H�(�?�{@�
=B�B�G�                                    Bx{k��  "          @�ff�!G�@1�@���B[z�B�aH�!G�?�  @�  B�G�B�W
                                    Bx{k�f  �          @��R�5@333@���BY�B�z�5?��\@�\)B�B�=q                                    Bx{k�  
�          @�{�O\)@&ff@��
BbffB֮�O\)?��@�Q�B�B�\                                    Bx{l�  �          @��8Q�@(Q�@�33BaQ�BҮ�8Q�?���@�  B���B��                                     Bx{lX  T          @����@(��@��BcQ�B�ff��?�{@�Q�B�� B�W
                                    Bx{l�  T          @��Ϳ
=q@&ff@��Bd��Bˏ\�
=q?���@��B��B�                                    Bx{l.�  "          @�z��@ ��@��BiffBˏ\��?xQ�@�  B���B�p�                                    Bx{l=J  "          @��;��H@�R@�{Bk�\B�.���H?p��@���B�  B��                                    Bx{lK�  �          @��;��@!�@�{Bj�
B������?}p�@�G�B���B�                                    Bx{lZ�  �          @����=q@,��@��HBa�RB�W
��=q?�Q�@�  B�B�B�ff                                    Bx{li<  �          @��ͽ�Q�@@��@w�BPp�B�W
��Q�?Ǯ@���B��B�p�                                    Bx{lw�  �          @�(�����@,(�@���B`�HB��þ���?���@��RB�u�BҸR                                    Bx{l��  �          @�(����@��@�{Bm�B��H���?n{@���B�B�u�                                    Bx{l�.  �          @�33���@7�@y��BV(�B�����?�
=@��
B�u�B�=q                                    Bx{l��  T          @��H��@Dz�@n�RBJ(�B�B���?�@���B�u�B�L�                                    Bx{l�z  �          @��\���
@8��@w
=BT��B�B����
?��H@��HB���B�p�                                    Bx{l�   �          @������@HQ�@l(�BF\)B��þ���?�  @���B��Bǀ                                     Bx{l��  
P          @��
��\)@C�
@p  BJ��B��{��\)?�@�G�B�.B�G�                                    Bx{l�l  ,          @�(���\)@Dz�@qG�BJ�
B�Q쾏\)?�@��B�8RB���                                    Bx{l�  T          @�(��Ǯ@,(�@�G�B`p�B�W
�Ǯ?�(�@�{B�ffB���                                    Bx{l��  �          @�(���
=@4z�@z�HBW�HB�  ��
=?��@��
B�
=Bգ�                                    Bx{m
^  T          @�(���=q@U@b�\B:
=B�G���=q@G�@�B���B�Q�                                    Bx{m  T          @��
��\)@^{@Z=qB1�HB�uý�\)@�@��HB}\)B��f                                    Bx{m'�  �          @��
=#�
@^�R@Z=qB1�B���=#�
@(�@�33B|�HB�{                                    Bx{m6P  �          @��<��
@_\)@XQ�B0(�B�u�<��
@{@�=qB{\)B�(�                                    Bx{mD�  �          @��=�G�@a�@U�B-(�B�z�=�G�@G�@�G�Bx�B��=                                    Bx{mS�  �          @���u@Z�H@\(�B4=qB���u@��@�33B~��B���                                    Bx{mbB  
�          @�(��Ǯ@O\)@g�B?�HB����Ǯ?�z�@�
=B��
B��H                                    Bx{mp�  �          @����@J=q@i��BC=qB�aH��?�=q@�
=B�B�B��                                    Bx{m�  J          @��H���H@B�\@o\)BJQ�B�B����H?�
=@�Q�B��=B�L�                                    Bx{m�4  ^          @�녾��@Mp�@c�
B>�RB®���?�z�@�z�B���B�ff                                    Bx{m��  "          @����\)@K�@c33B>�HB�\�\)?��@��
B��=B�.                                    Bx{m��  �          @��\�0��@>�R@n�RBJ�\B���0��?�33@�\)B��{B�\                                    Bx{m�&  �          @�33�=p�@<��@qG�BL�RB�
=�=p�?���@�Q�B�Q�B�Q�                                    Bx{m��  
Z          @��\�^�R@P  @]p�B733B��^�R?��R@��B|�B��                                    Bx{m�r  
�          @��\�.{@^{@QG�B+G�B�(��.{@��@�ffBr��B�p�                                    Bx{m�  
�          @�33�!G�@mp�@E�BG�B�(��!G�@#33@��HBe{Bϊ=                                    Bx{m��  
�          @����@qG�@@��B(�B����@(��@�G�Ba�B�B�                                    Bx{nd  �          @��\�   @r�\@<(�B{B�
=�   @+�@~�RB^�B�#�                                    Bx{n
  �          @�녾\@vff@7
=B�
B�ff�\@0��@z�HBZ�B���                                    Bx{n �  �          @�G����@w�@4z�B  B�����@333@xQ�BXffB���                                    Bx{n/V  �          @����\)@u�@8Q�B�
B�
=�\)@/\)@{�B\=qB���                                    Bx{n=�  T          @�G��k�@xQ�@2�\B�HB��;k�@4z�@w
=BW  B�W
                                    Bx{nL�  �          @�G�����@xQ�@1G�BB��þ���@5�@uBU��B�=q                                    Bx{n[H  �          @��׾�@�Q�@��A�G�B�z��@X��@VffB2��B�aH                                    Bx{ni�  �          @�  =���@�z�@�\A��
B��=���@N�R@]p�B;�RB�L�                                    Bx{nx�  T          @�
=>�33@���?�z�A�p�B���>�33@j�H@<(�B{B�                                    Bx{n�:  "          @�ff?   @�G�?���A_33B�ff?   @~{@{A���B���                                    Bx{n��  
�          @�{?!G�@�Q�?���AZ{B�\?!G�@}p�@�A�(�B��)                                    Bx{n��  
�          @�(�>�@�\)?���AVffB���>�@|(�@��A���B�(�                                    Bx{n�,  �          @�=q=�Q�@�=q?�Q�A�p�B���=�Q�@k�@,(�B�RB�.                                    Bx{n��  ^          @��H��  @W�@:�HB�
B����  @z�@s�
Ba��B�3                                    Bx{n�x  �          @�녿�=q@b�\@*�HB33B��)��=q@#�
@g�BQ�
Bី                                    Bx{n�  �          @�����  @u�@{A�ffB�\)��  @>�R@Q�B8�
B�=q                                    Bx{n��  �          @�G��L��@y��@(�A�z�B�8R�L��@C33@P��B7�
B�aH                                    Bx{n�j  �          @�G��!G�@��?���A��HBř��!G�@Q�@E�B*�Bɨ�                                    Bx{o  �          @��׿�@�\)?���A�Q�B�녿�@fff@*=qB��B�W
                                    Bx{o�  �          @�\)��@��\?�=qA^�\B�
=��@s33@�A�=qB��                                    Bx{o(\  
�          @��;��@�\)>8Q�@�B��;��@~�R?�Q�A�  B���                                    Bx{o7  T          @�z��R@�Q�Q��,(�BĊ=��R@��>��H@θRB�W
                                    Bx{oE�  �          @�z��@��׿#�
��HB�33��@���?+�AG�B�8R                                    Bx{oTN  �          @��R��p�@��?�ffA��
B�{��p�@j=q@ ��B�HB���                                    Bx{ob�  �          @����@���?�Aq��B��
��@p��@��B  B®                                    Bx{oq�  �          @����G�@�p�?�R@�33B���G�@���?�33A�z�B���                                    Bx{o�@  �          @��׿G�@�{�����B���G�@���?�Q�At��BȊ=                                    Bx{o��  
�          @��ÿ333@��R�\)�޸RB��
�333@���?�
=As\)B�z�                                    Bx{o��  �          @�33�Tz�@�33?���Ad(�Bɮ�Tz�@u�@
=A�Q�B̏\                                    Bx{o�2  �          @���W
=@��H?�ffA��HB���W
=@p��@ ��BffB�
=                                    Bx{o��  "          @��H�&ff@�(�?�Al��B��ÿ&ff@vff@��A��B�G�                                    Bx{o�~  T          @�(��!G�@�p�?���ApQ�B�B��!G�@xQ�@(�A���Bƅ                                    Bx{o�$  �          @�(���G�@�(�?���A\��B��)��G�@w�@�A�=qB�#�                                    Bx{o��  �          @��Ϳ�{@���?��A��RB��)��{@n{@"�\B��B��                                    Bx{o�p  �          @�(��s33@��
?�{A�(�B��
�s33@Y��@>�RB ��B�\                                    Bx{p  
�          @�(���=q@x��@  A�G�B�
=��=q@Dz�@Q�B5  B�                                    Bx{p�  �          @�33�k�@w�@33A�=qBΣ׿k�@B�\@Tz�B9�BՊ=                                    Bx{p!b  �          @�=q�Tz�@w
=@��A��B�aH�Tz�@C33@QG�B8{BҞ�                                    Bx{p0  �          @�=q�z�@hQ�@,��B�HB�(��z�@,��@hQ�BR�B�=q                                    Bx{p>�  �          @��\�Ǯ@\(�@>�RB#  B��H�Ǯ@(�@vffBe33B��                                    Bx{pMT  �          @�녿(�@c�
@1G�B
=B�W
�(�@'�@k�BWffB��                                    Bx{p[�  �          @�녿u@p  @=qA��B��
�u@9��@XQ�B?��Bؽq                                    Bx{pj�  T          @�G����\@z�H@G�A�=qB�#׿��\@K�@C�
B)�HBׅ                                    Bx{pyF  �          @��ÿ��@���?��
A�(�B�  ���@\(�@'�B�B���                                    Bx{p��  
�          @�G����
@�Q�?�Q�A�33B�zΰ�
@W�@0��B��Bݽq                                    Bx{p��  T          @�G����@�z�?��
A��B�녿��@g
=@��A��B���                                    Bx{p�8  T          @������@���?uAEB������@mp�@A�33B�\                                    Bx{p��  T          @�z῏\)@HQ��8Q��#\)B�G���\)@tz������BԨ�                                    Bx{p  �          @�
=��G�@^�R�\)��
B��쿡G�@�G���z�����B֨�                                    Bx{p�*  T          @���ff@S�
�0  ��\B�33��ff@|�Ϳ��H��B�Ǯ                                    Bx{p��  T          @�{���
@O\)�.�R��B�8R���
@w���(����\Bأ�                                    Bx{p�v  �          @������@aG��
=q���B��)����@\)����g
=B�{                                    Bx{p�  �          @��Ϳ�{@c33�ff��B��
��{@�  ���
�[
=Bم                                    Bx{q�  �          @�(���ff@X�������B��H��ff@{���{���Bؙ�                                    Bx{qh  �          @��
��{@dz��
=���B���{@�Q쿃�
�[\)B�z�                                    Bx{q)  �          @��Ϳ��@g��z���
=B�{���@�녿}p��P  B��                                    Bx{q7�  "          @����
@^�R����p�B�\)���
@~{��p����\B�                                    Bx{qFZ  T          @�p���{@p�׿�����=qB���{@��
�8Q��{B؏\                                    Bx{qU   �          @�{���@�33�n{�BffB�#׿��@�ff>u@C�
B�\)                                    Bx{qc�  �          @���@�
=�
=q��=qB�녿�@��R?#�
A��B�                                      Bx{qrL  �          @�{���@�{�#�
�p�B�(����@��R?�@ۅB�
=                                    Bx{q��  T          @�ff��=q@��\��
=��p�B�8R��=q@�G�?8Q�A{Bޞ�                                    Bx{q��  �          @��R��  @��
�333�z�B�\��  @���>�ff@�33B�                                    Bx{q�>  
�          @�p���z�@\)��p�����B�aH��z�@�  ��{����BҊ=                                    Bx{q��  �          @�p�����@\)��(���ffB�(�����@���������B�u�                                    Bx{q��  �          @�{���@�G���  ��(�B��H���@�G���33����B�8R                                    Bx{q�0  �          @��z�H@|(���G���
=B���z�H@��׿!G���B�                                    Bx{q��  "          @��Ϳ}p�@vff���
����Bнq�}p�@�{�+��=qB΅                                    Bx{q�|  �          @������@��׿�G����RBҳ3����@��R����\B�\)                                    Bx{q�"  �          @�33���\@�Q쿇��aB�=q���\@���=u?@  B�#�                                    Bx{r�  �          @��Ϳ��R@�녿���g
=B��H���R@�ff<�>�G�B�Ǯ                                    Bx{rn  �          @�
=����@�G���Q��w�B�\)����@��R��\)�Y��B��H                                    Bx{r"  �          @�  ��=q@���}p��L��B�k���=q@���>��?���B֊=                                    Bx{r0�  T          @�\)��Q�@�����\�Q�Bڏ\��Q�@�\)=�G�?��HBم                                    Bx{r?`  �          @��ٙ�@r�\���H����B�LͿٙ�@�G��������RBី                                    Bx{rN  T          @�ff��\)@p  ��p���\)B���\)@�=q�+���B�W
                                    Bx{r\�  T          @�{����@��׿��R��{B��)����@��R���޸RB�Q�                                    Bx{rkR  T          @�
=��\)@�=q��(��~{B����\)@�  ��G���{Bר�                                    Bx{ry�  �          @�  ���
@�z�=p����B�uÿ��
@�{>�p�@�Q�B�                                    Bx{r��  "          @�
=��G�@y�����
���\B�=q��G�@����ff���\B�                                    Bx{r�D  �          @����  @w
=��z��ɅB��
��  @�
=�Tz��+
=B��                                    Bx{r��  �          @��׿�@~{��=q���RB��Ϳ�@�녿:�H�z�B�W
                                    Bx{r��  �          @������\@�Q����ģ�BЏ\���\@���G���RB�Q�                                    Bx{r�6  
�          @�녿��\@�33�ٙ���B��
���\@��Ϳz���\B�                                      Bx{r��  "          @��\�n{@�(���p����B�LͿn{@�{������B˙�                                    Bx{r��  �          @��\���
@~�R�   ��G�B�{���
@���fff�5BΣ�                                    Bx{r�(  T          @�G���=q@o\)�
=q����B�=q��=q@�p������f�HB�u�                                    Bx{r��  �          @�=q�G�@k��ٙ����\B�u��G�@\)�333�B陚                                    Bx{st  �          @�33��@�=q���k\)B����@�\)��Q쿇�B�Q�                                    Bx{s  �          @�(���ff@�G�������=qB�#׿�ff@�  ��=q�Z=qB�                                      Bx{s)�  �          @�p���{@��׿�(����HB���{@�Q�Ǯ���HB�.                                    Bx{s8f  �          @��Ϳ�ff@z�H��z���Q�B�Q��ff@�
=�(����B�B�                                    Bx{sG  T          @�{��@s�
��
=��z�B�\��@��fff�0Q�B�u�                                    Bx{sU�  T          @���@dz���H����B��Ϳ�@�=q��
=���\B�                                    Bx{sdX  T          @�z��@dz��ff���\B�\��@�����\)����B�                                    Bx{sr�  
�          @�(��ٙ�@q�� ����G�B�aH�ٙ�@�p��}p��G
=B�\)                                    Bx{s��  �          @��
��{@\)��z���
=B�\��{@��ÿ�����B�Q�                                    Bx{s�J  "          @�(���33@�p���z��g�
B�=q��33@��\��\)�h��B���                                    Bx{s��  �          @���ٙ�@�33�����r{B��ÿٙ�@��׾���33B�W
                                    Bx{s��  �          @��
��ff@���Y���)��B���ff@��>aG�@/\)B���                                    Bx{s�<  �          @�33� ��@��׿n{�;�
B�{� ��@��
=���?��RB��f                                    Bx{s��  +          @��H���H@�=q��
=�o
=Bី���H@�����˅B�                                      Bx{sو  "          @�����z�@~�R����  B�G���z�@��R�Ǯ����B�\                                    Bx{s�.  �          @�G�����@��Ϳ�G�����B�Q쿨��@��\�B�\��B��                                    Bx{s��  T          @�녿�@�G��z�H�F�\BҀ ��@���=�G�?�
=B�Ǯ                                    Bx{tz  T          @�G��z�H@��
���H��G�B�=q�z�H@��?��@��B�Q�                                    Bx{t   T          @���?��@��?�=qA��B�u�?��@l��@�
A��
B��                                    Bx{t"�  �          @��R>�{@��?B�\A{B���>�{@�G�?�A�z�B�B�                                    Bx{t1l  T          @��>u@��?�G�AP  B��3>u@~�R@33A�G�B��                                    Bx{t@  �          @���W
=@�z�?aG�A6{B��\�W
=@�G�?�
=A�(�B�
=                                    Bx{tN�  �          @����  @��?��\AQ�B��=��  @~�R@33A�B�(�                                    Bx{t]^  �          @��R>aG�@�Q�?��A��\B�(�>aG�@tz�@�\A��B�z�                                    Bx{tl  �          @���=�Q�@��?���Ahz�B��3=�Q�@}p�@
=qA�RB�z�                                    Bx{tz�  �          @�Q�!G�@��?aG�A4��B�\)�!G�@���?�z�A�G�BŸR                                    Bx{t�P  T          @��R?�@��?���A]�B�z�?�@q�@�\A�Q�B�z�                                    Bx{t��  "          @���?c�
@��?��RA~�RB�G�?c�
@tz�@{A�ffB��q                                    Bx{t��  �          @�  ?B�\@���?��A`��B��f?B�\@x��@A�{B��                                    Bx{t�B  �          @��?�33@�(�?��\A�z�B���?�33@l��@�RA�33B��{                                    Bx{t��  �          @�33?�Q�@\)?��
A��B�p�?�Q�@\��@+�B�RB���                                    Bx{tҎ  �          @���?�33@��
?���A��\B��?�33@c�
@0  B=qB��                                    Bx{t�4  "          @�
=?xQ�@��\?˅A�p�B���?xQ�@u�@$z�BffB���                                    Bx{t��  �          @�
=?
=q@�\)?���A��B�\)?
=q@���@Q�A�ffB��
                                    Bx{t��  �          @�
=?\)@�ff?�Q�A��B��=?\)@\)@��A���B��H                                    Bx{u&  T          @�\)?��@�z�?��A��B���?��@w�@(��B�HB�                                    Bx{u�  T          @�\)>�Q�@��H?���A���B��{>�Q�@r�\@333B=qB�8R                                    Bx{u*r  �          @�\)?=p�@��H?�33A�ffB��=?=p�@u�@(Q�B�
B�                                      Bx{u9  T          @��?�@���?�A�\)B�G�?�@p��@(Q�B�B�Q�                                    Bx{uG�  �          @�Q�?Q�@���?��A�33B�B�?Q�@n{@5B��B�\                                    Bx{uVd  T          @�  ?L��@��\?�\A��HB�
=?L��@s33@.�RB��B�.                                    Bx{ue
  T          @���?:�H@��\?�33A�
=B���?:�H@qG�@7
=B�RB���                                    Bx{us�  "          @��H?5@�Q�?��
A��
B��?5@���@"�\A�=qB�
=                                    Bx{u�V  
�          @��?   @�ff?��A��RB�B�?   @x��@7�B��B�p�                                    Bx{u��  
�          @�z�>#�
@�33@\)AٮB��q>#�
@mp�@K�B"z�B�                                    Bx{u��  �          @�(�>k�@�ff@�RA�{B���>k�@`��@XQ�B/z�B�k�                                    Bx{u�H  �          @��
�#�
@��
@%B �HB�G��#�
@Z=q@^{B6Q�B�W
                                    Bx{u��  "          @���?��@o\)@;�B�B��f?��@>{@mp�BK(�B�G�                                    Bx{u˔  �          @��?B�\@���@z�A�\)B�?B�\@X��@K�B)��B��)                                    Bx{u�:  �          @�\)>��@��@\)A��
B�  >��@^�R@G�B&�\B��                                    Bx{u��  "          @�
=>�ff@x��@'�B
=B���>�ff@L(�@\(�B;�\B���                                    Bx{u��  �          @�?���@k�@*�HBp�B��3?���@>�R@\(�B=�RB��\                                    Bx{v,  T          @�p�>�@u@&ffBB�8R>�@H��@Z=qB<  B�8R                                    Bx{v�  "          @�p�>�  @��@�A�B�{>�  @\(�@H��B)\)B���                                    Bx{v#x  T          @�=�@�=q@�\A�
=B��R=�@\(�@J=qB)��B�#�                                    Bx{v2  
�          @���    @�{@ ��Ȁ\B��    @hQ�@9��B��B�#�                                    Bx{v@�  
�          @��
>\)@�{?���A�p�B�{>\)@h��@6ffB�B��                                     Bx{vOj  �          @�(��B�\@�z�@z�A�Q�B�\)�B�\@c�
@<��B33B�8R                                    Bx{v^  �          @��H>u@~�R@  A�\)B�  >u@W�@EB)�B�                                    Bx{vl�  
�          @��ü#�
@��
?�{A�{B�ff�#�
@fff@/\)B�B�u�                                    Bx{v{\  �          @��\�Ǯ@~�R@p�A�
=B�8R�Ǯ@X��@B�\B&��B�(�                                    Bx{v�  T          @�=q��@�=q@�
A֣�B��;�@`  @:=qB�B�ff                                    Bx{v��  T          @�Q�?u@���?��HA�z�B�\)?u@dz�@$z�B33B��                                    Bx{v�N  T          @��R?aG�@���?��A�(�B�z�?aG�@e@   B(�B�k�                                    Bx{v��  �          @�>L��@|��?�(�A�G�B�33>L��@Z�H@2�\B�
B�L�                                    Bx{vĚ  "          @���>�z�@��\?�=qA�
=B�>�z�@hQ�@(�B�B��q                                    Bx{v�@  �          @��=#�
@�ff?��A�33B��=#�
@s33@�RA�G�B���                                    Bx{v��  �          @��H���@�z�?��Al��B�B����@s33?�p�A�(�B���                                    Bx{v��  "          @�G��#�
@�Q�>aG�@AG�B�#׼#�
@��?�z�A{�B�(�                                    Bx{v�2  �          @�ff���H@^{@	��A�(�B�\���H@:=q@7
=B%=qB�(�                                    Bx{w�  �          @�G����@=p�@8��B�B�\���@  @]p�BF�HC�                                    Bx{w~  �          @��H�   @@��@5�BffB�#��   @z�@Z�HB@��C                                    Bx{w+$  �          @�z��@N�R@.{B�B�p���@#�
@W
=B9�B��R                                    Bx{w9�  �          @����@H��@2�\B��B��)��@p�@Y��B;z�C��                                    Bx{wHp  �          @����
=@2�\@@��B#
=B�B��
=@z�@c33BH�
C��                                    Bx{wW  �          @��H�fff@o\)@�A�\)B���fff@HQ�@G�B/�B��f                                    Bx{we�  �          @�Q�:�H@{�?޸RA��
B�(��:�H@]p�@"�\B�B���                                    Bx{wtb  �          @��\���
@~{?У�A��B�����
@a�@��BB��)                                    Bx{w�  T          @�z���@n�R?��HA�(�B��H��@U�@�RA�B��)                                    Bx{w��  �          @�{��Q�@�  ?��HA�Q�B��)��Q�@b�\@!�Bp�B��                                    Bx{w�T  
�          @��
�L��@���?��HA��HB�G��L��@s33@%B�HB��                                    Bx{w��  "          @��;�Q�@�p�@ ��A�\)B���Q�@h��@6ffB�HB�.                                    Bx{w��  K          @��R�!G�@�=q@�RA��Bř��!G�@`  @C33B"�\B�k�                                    Bx{w�F  
�          @�\)�^�R@��@�A���Bˏ\�^�R@g�@:�HBffB��                                    Bx{w��  "          @�ff��\)@���?���A���B��f��\)@u@{B�HB��                                    Bx{w�  "          @�ff�n{@xQ�@�A�\)B��f�n{@R�\@H��B*{BӀ                                     Bx{w�8  "          @��׿�\)@g
=@2�\B�RB݀ ��\)@;�@_\)B<�
B���                                    Bx{x�  �          @�33���H@k�@333B\)B�33���H@@  @aG�B:{B�q                                    Bx{x�  �          @�z��{@���@
�HA�(�Bߏ\��{@^�R@>{B�
B垸                                    Bx{x$*  T          @���ff@��R?��A��B����ff@tz�@\)A��
B�                                     Bx{x2�  �          @�{���H@�G�?���A�33B�{���H@x��@z�A��B�u�                                    Bx{xAv  "          @�Q��{@��\?�Q�A��
B���{@xQ�@#�
A�  B�B�                                    Bx{xP  �          @�����z�@�=q?��
A��B�Ǯ��z�@vff@)��A���B��
                                    Bx{x^�  �          @��H��G�@���?�G�A��\B�p���G�@��@�HA��
B�L�                                    Bx{xmh  �          @�����@��@�
A��
B�����@u@:�HB�B��
                                    Bx{x|  �          @�(����@��R@#33A���B�k����@e@W�B$��B�33                                    Bx{x��  �          @��Ϳ�\)@��@
=A�(�B��쿯\)@q�@Mp�Bz�B���                                    Bx{x�Z  �          @��Ϳ�=q@�p�@,(�A��HB�p���=q@aG�@`  B+��B�p�                                    Bx{x�   �          @�p��У�@z�H@<��B
=B�G��У�@N{@l(�B6ffB�q                                    Bx{x��  �          @���{@w
=@C33Bp�B�.��{@I��@q�B;�RB�
=                                    Bx{x�L  �          @�(���p�@e@Mp�B33B�p���p�@7
=@xQ�BE
=B�k�                                    Bx{x��  T          @�Q쿵@Z=q@W�B)�B�.��@)��@�  BTffB�k�                                    Bx{x�  �          @�\)��G�@AG�@u�BI
=B��H��G�@
�H@��Bu�B�                                    Bx{x�>  �          @�p���\)@J=q@S�
B2��B�#׿�\)@=q@x��B^ffB��f                                    Bx{x��  
�          @�ff�aG�@q�@C�
B(�B�\)�aG�@Dz�@qG�BFB��                                    Bx{y�  �          @��R�s33@\)@1G�B\)B��)�s33@U@a�B5�HB���                                    Bx{y0  �          @�p����@�(�?�A��B�\���@y��@1G�B	33B�z�                                    Bx{y+�  �          @�Q쿎{@�z�@Q�A�G�B�uÿ�{@w�@>{B�
B��                                    Bx{y:|  "          @��ÿ�
=@�\)@{A�\)B�.��
=@h��@Q�B"z�B��                                    Bx{yI"  "          @������\@��@
�HA�(�Bԙ����\@u�@@  B��B���                                    Bx{yW�  �          @��׿���@�33?�Q�A�G�BՀ ����@�
=@A�(�B�\)                                    Bx{yfn  �          @�\)���@��H?�33A��B��Ϳ��@�
=@33A�=qB�z�                                    Bx{yu  �          @�p�����@��H?�z�AZ=qB�(�����@���@�
A�  B�z�                                    Bx{y��  �          @��R���@��@�
A�33B�����@g
=@5B��B�33                                    Bx{y�`  �          @�녿��H@`  @Dz�Bz�B�aH���H@4z�@mp�B<B��\                                    Bx{y�  �          @��\��\)@b�\@G�B�\B뙚��\)@6ff@p��B?\)B��                                    Bx{y��  T          @�=q��{@g�@AG�B�B�\)��{@<��@k�B:33B�                                    Bx{y�R  �          @�Q��=q@j=q@AG�Bz�B��Ϳ�=q@>�R@l(�B=��B��H                                    Bx{y��  �          @��Ϳ�  @@  @h��BA(�B�8R��  @p�@��Bjz�B��
                                    Bx{y۞  T          @����@K�@j=qB<B�G���@��@��RBf��B�                                      Bx{y�D  �          @������H@,(�@qG�BG
=B��H���H?��@�
=Bk�\C�                                    Bx{y��  �          @��R����?�  @��HBt�
C�H����?�R@���B���C!��                                    Bx{z�  T          @��
��?�ff@���Bz��CLͿ�>�
=@�
=B��C&�=                                    Bx{z6  T          @��H��ff?�=q@�{Bp�HCE��ff?xQ�@�
=B�� C                                      Bx{z$�  T          @��\��\)?�
=@��B}G�C쿯\)?L��@��B��3C�
                                    Bx{z3�  �          @��H��33?�{@�p�B��3B�녿�33?5@���B��=C)                                    Bx{zB(  �          @�z῝p�@<��@mp�BE{B�33��p�@
�H@�
=Bm��B��                                    Bx{zP�  �          @�(�����@)��@z=qBUz�B䞸����?�=q@�33B}(�B�B�                                    Bx{z_t  �          @����=q@\)@��B_�RB��\��=q?�33@��B�C
p�                                    Bx{zn  �          @�33��(�?���@~{B\\)CLͿ�(�?��@���Bw�C�q                                    Bx{z|�  T          @�G���
?�z�@��HBjQ�C���
?
=@���B}��C#�R                                    Bx{z�f  
�          @��׿��@�\@u�BX\)C�Ϳ��?�  @���Bu��Cc�                                    Bx{z�  �          @��ÿ�
=@�@qG�BQ(�C����
=?�z�@��
Boz�C�)                                    Bx{z��  �          @�G�����@
=@u�BTQ�C�{����?�=q@��Bq�C�\                                    Bx{z�X  �          @����   @�\@tz�BT��Cff�   ?��\@�z�Bq{C��                                    Bx{z��  �          @�{�
=q?�@p  BTG�C���
=q?��@���BlC�                                    Bx{zԤ  T          @�(��p�?У�@dz�BI��Ck��p�?h��@s�
B]��C��                                    Bx{z�J  �          @�p����?�33@_\)BA��C#����?���@r�\BYG�C��                                    Bx{z��  T          @�
=�(Q�?��R@W�B6ffC�)�(Q�?��@l(�BM�HC��                                    Bx{{ �  �          @��R�{@ff@O\)B-��CaH�{?�Q�@hQ�BI�C��                                    Bx{{<  �          @�p����@#33@L(�B,��C�����?��@g�BK�C�                                    Bx{{�  �          @��
�%�?�@W�B;�C��%�?���@i��BQp�CG�                                    Bx{{,�  �          @�z��)��?�=q@U�B7��CT{�)��?�z�@g�BMffCJ=                                    Bx{{;.  T          @�33�
�H@G�@@��B��B����
�H@\)@c33B<\)C�                                    Bx{{I�  T          @�{�p�@P��@>{B  B�8R�p�@(��@b�\B7{C�3                                    Bx{{Xz  
�          @�ff�Q�@E@C33B  B�\�Q�@p�@e�B9=qC                                    Bx{{g   T          @��R�
=@AG�@H��B\)B��H�
=@�@j=qB>=qC��                                    Bx{{u�  
�          @�ff���@N{@EBB�33���@$z�@h��B=�
C��                                    Bx{{�l  "          @��(�@9��@I��B
=C
=�(�@  @h��B>��C	@                                     Bx{{�  T          @�\)��H@8Q�@P��B#�RC��H@p�@p  BC33C	��                                    Bx{{��  
�          @�����@>{@N�RB!=qC �����@�
@n�RBAffC��                                    Bx{{�^  �          @���!G�@QG�@3�
B�HB�#��!G�@+�@W�B*  C#�                                    Bx{{�  T          @��R���@[�@(Q�B ��B������@8Q�@N�RB#{C�                                    Bx{{ͪ  �          @��
=@dz�@(�A�B�  �
=@C33@C�
B��B��\                                    Bx{{�P  �          @�{���@Tz�@+�B��B����@0��@P  B%��C�=                                    Bx{{��  �          @�p�� ��@ ��@X��B.�\C�3� ��?�@s33BJC�\                                    Bx{{��  �          @��ff@�H@eB:�C8R�ff?��H@~�RBW=qC\                                    Bx{|B  �          @��R�(Q�@2�\@HQ�B(�CW
�(Q�@	��@fffB:�\C�                                    Bx{|�  "          @�p��#33@(��@QG�B&�HC  �#33?��R@mp�BC�C�                                    Bx{|%�  �          @�
=���@%�@`��B3�C����?��@{�BP��C�H                                    Bx{|44  �          @�\)���@z�@tz�BH=qC�=���?���@�Bdp�C�=                                    Bx{|B�  �          @�\)�{@��@w�BLQ�C0��{?���@�
=BgffC�H                                    Bx{|Q�  �          @�  ��@
=q@xQ�BL�C� ��?�z�@��RBfQ�CQ�                                    Bx{|`&  T          @�G����@�R@u�BE�RC�����?��R@�p�B_��C!H                                    Bx{|n�  �          @�(���@�@u�BA�RCY���?�
=@��RB]�C�R                                    Bx{|}r  
�          @�Q���
?��@�=qB]��C����
?��@�33BtQ�CaH                                    Bx{|�  
�          @�  ��@p�@z�HBB=qC�H��?ٙ�@��B]��C�                                    Bx{|��  �          @�  �(Q�@1�@eB,�
CaH�(Q�@z�@�G�BI�\C�q                                    Bx{|�d  �          @��R�(��@S�
@A�B(�C �{�(��@,��@e�B-��CaH                                    Bx{|�
  "          @�ff�Dz�@Tz�@=qA��C�\�Dz�@4z�@>�RB33C	xR                                    Bx{|ư  "          @����@��>��
@qG�B�=��@�?���AT(�B��                                    Bx{|�V  
�          @��H��@���:�H�
{B�G���@�
==�Q�?�{B�                                    Bx{|��  "          @���:=q@s33?���A�Q�B���:=q@\(�@G�A�  C:�                                    Bx{|�  T          @����+�@�Q�>.{?�Q�B�Q��+�@��?s33A-�B��{                                    Bx{}H  T          @��H�&ff@��>W
=@��B��&ff@�  ?�  A6ffB��                                    Bx{}�  �          @��
�p�@�
=>���@mp�B�\�p�@��H?��AN{B�\                                    Bx{}�  �          @�z��\)@�\)>�G�@��RB�#��\)@�=q?�  AaG�B��H                                    Bx{}-:  �          @�p��*=q@����\)�G�B�=q�*=q@�33?:�HA�RB��                                    Bx{};�  �          @��1�@��\���
�uB�\)�1�@�Q�?B�\A��B�33                                   Bx{}J�  T          @��� ��@�\)?E�A�
B� � ��@���?���A�ffB��f                                   Bx{}Y,  �          @�
=�(Q�@�  ����5�B�z��(Q�@�\)?\)@�B���                                    Bx{}g�  T          @�\)�+�@�\)�8Q���B�q�+�@�{?!G�@��B�33                                    Bx{}vx  �          @���9��@�=�\)?L��B�u��9��@�33?^�RA�B�z�                                    Bx{}�  �          @�33�8Q�@�����
�uB�ff�8Q�@�p�?J=qA  B�8R                                    Bx{}��  T          @��H�AG�@��;8Q���HB��f�AG�@��?(�@�33B�aH                                    Bx{}�j  �          @��
�L(�@�  �&ff�߮B����L(�@���=�?�=qB�.                                    Bx{}�  "          @��\�H��@�p�����3�
B��)�H��@�G���=q�:=qB�Q�                                    Bx{}��  �          @��\�L��@�z�u�%B�k��L��@���L����B�
=                                    Bx{}�\  T          @����@��@�{�����R=qB�aH�@��@��H��(���B��                                     Bx{}�  �          @���N�R@z=q���
�b=qC�H�N�R@�=q�\)�ÅC u�                                    Bx{}�  T          @�
=�C33@��R�   ��\)B��
�C33@�\)>��@8��B��\                                    Bx{}�N  "          @�G��Q�@qG�����p�B���Q�@�(���z�����B�                                    Bx{~�  �          @����
=q@z=q?�@�\)B��f�
=q@p  ?�G�A���B�                                    Bx{~�  
�          @�{��p�@{��Tz��.ffB�z��p�@�Q��G����HB♚                                    Bx{~&@  �          @���˅@qG��5�
��B�q�˅@��R���ɅB�W
                                    Bx{~4�  
�          @�  ����@vff�&ff���B�=����@�  ��\)��(�B�G�                                    Bx{~C�  �          @�{��  @|(��%�����B�Ǯ��  @��\��=q��ffB�B�                                    Bx{~R2  �          @�
=��
@c�
��R�z�B�8R��
@g
=<�>�p�B�z�                                    Bx{~`�  T          @�G����R?޸R@n�RBZ��C
�=���R?�ff@~�RBr  C#�                                    Bx{~o~  �          @�ff��33?��R@�p�B�8RC��33>Ǯ@�=qB�#�C(�                                     Bx{~~$  "          @�{��?޸R@��\BeQ�C����?fff@��By{C:�                                    Bx{~��  T          @�\)�\)?��@�G�B`
=C33�\)?u@�G�Bs�RC�                                    Bx{~�p  T          @����R?�33@��
Bd��CJ=��R?L��@��HBw  C!�q                                    Bx{~�  
�          @������@��@�{BU��C�����?�Q�@���Bn�
C�                                    Bx{~��  
�          @�33��@$z�@���BN��C(���?�  @���Bjz�C�                                    Bx{~�b  T          @�(���R@7
=@���BF�
B��H��R@�
@�
=Be  C	J=                                    Bx{~�  �          @��
��R@A�@���B7ffC\)��R@G�@�  BU{C	�)                                    Bx{~�  �          @��H�%@N{@o\)B(�C �\�%@ ��@�  BF�RC�                                    Bx{~�T  
�          @����N{@E�@U�B�HCO\�N{@(�@tz�B-Q�C�{                                    Bx{�  �          @����!�@X��@^{BffB�aH�!�@.�R@�Q�B<�RC��                                    Bx{�  T          @�G��3�
@<��@p  B*��C�)�3�
@\)@��RBF33Cn                                    Bx{F  "          @�G��"�\@X��@e�B �RB��"�\@-p�@��
B?�HC8R                                    Bx{-�  "          @��\��
@vff@]p�Bz�B�\)��
@K�@��\B<z�B��                                    Bx{<�  !          @��
�z�@�  @>�RB  B��z�@j=q@l(�B$�B�{                                    Bx{K8  "          @��\�(�@��@ffA�(�B��f�(�@z�H@EB{B�3                                    Bx{Y�  "          @�{�6ff@�Q�@{A�  B����6ff@c�
@8��BffC ��                                    Bx{h�  
�          @����,��@�(���z��L(�B�W
�,��@��>��H@�p�B�=                                    Bx{w*  
�          @����=q@�(����
�o33B��=�=q@�G������B��                                    Bx{��  �          @�Q쿆ff@�z�>��@@  B����ff@���?�\)AN{B̸R                                    Bx{�v  "          @�(��	��@����G��/33B��)�	��@��H�#�
��p�B��                                    Bx{�  
�          @�  ��=q@�z�޸R��  B�Ǯ��=q@��
�\(���B�B�                                    Bx{��  n          @��R���\@�
=�ff��\)B�����\@�녿�p��~ffB�(�                                    Bx{�h  �          @�z῵@���K��
=Bڮ��@�����H��p�Bֽq                                    Bx{�  �          @�p���=q@�Q��.{��\)B♚��=q@�p���
=���RB��H                                    Bx{ݴ  �          @��H����@���B�\�z�B�����@�������p�B��                                    Bx{�Z  �          @�G���\@p���C�
�G�B�{��\@�\)�ff�ԸRB�                                    Bx{�   �          @�z���@��H��33��{B�G���@�33��{�AB�z�                                    Bx{�	�  �          @��׿���@�Q��{��(�Bγ3����@��׿�  �/\)B�ff                                    Bx{�L  T          @���-p�@���>�ff@�(�B����-p�@��
?�p�AW�
B�                                     Bx{�&�  �          @����7
=@��?0��@�p�B�{�7
=@�
=?��RA{33B�=q                                    Bx{�5�  �          @�G��E@�G�?0��@�RB�k��E@�33?�p�Aw�B��q                                    Bx{�D>  �          @��H�N{@��R?xQ�A z�B����N{@�
=?޸RA��HB���                                    Bx{�R�  �          @����X��@�p�?�\)Aep�C!H�X��@w
=@�A�=qCE                                    Bx{�a�  �          @���N�R@�(�?���AZ{B����N�R@��\@�
A�
=C W
                                    Bx{�p0  T          @�(��[�@���?�{A��Cs3�[�@h��@!�A�
=CQ�                                    Bx{�~�  �          @��l(�@��\?��HAn�\C
�l(�@p��@	��A��Ch�                                    Bx{��|  �          @�G�����@i��?��A�C	�=����@Q�@��A�=qC��                                    Bx{��"  �          @����q�@�33�8Q��ffC��q�@�=q?
=q@��\C�f                                    Bx{���  �          @���|(�@w��\)����Cu��|(�@z=q=�?��C+�                                    Bx{��n  �          @�(��fff@�녾���(��C޸�fff@�G�?   @�=qC�                                    Bx{��  �          @�(��Y��@��#�
��  B���Y��@�
=>\)?�(�B��{                                    Bx{�ֺ  �          @�=q�?\)@��
�W
=�\)B��)�?\)@�{�����B�                                    Bx{��`  T          @���L��@���&ff��ffB���L��@�G�>\)?��RB�ff                                    Bx{��  �          @����>�R@��>��@.�RB����>�R@��?��A2{B�#�                                    Bx{��  �          @�G��W
=@�33?   @�\)B�p��W
=@�{?�  AQ�C ��                                    Bx{�R  �          @���p��@u������
Cu��p��@w
=>��?���C5�                                    Bx{��  �          @�  �5�@�(�?c�
A z�B��
�5�@z=q?˅A��B�                                    Bx{�.�  
�          @��\��\@���@!�A�p�B�Ǯ��\@�z�@S33B�B�k�                                    Bx{�=D  �          @��
��  @x��@\��B��B�Q��  @O\)@�=qB?�B�q                                    Bx{�K�  �          @��Ϳ�\)@Tz�@���BDp�B��f��\)@!�@���Bg�B�33                                    Bx{�Z�  �          @�=q�L��@�\)@33A�(�B�33�L��@�Q�@H��B	�B�{                                    Bx{�i6  �          @������@�  @8Q�A��HB�Q쿈��@�p�@j=qB!p�BШ�                                    Bx{�w�  �          @����z�@���@^�RBG�B����z�@g�@�B>��B�\)                                    Bx{���  �          @����p�@���@QG�B�B��´p�@j=q@~{B4�B���                                    Bx{��(  T          @�p����R@�{@2�\A�RB�33���R@�(�@dz�BffBۙ�                                    Bx{���  �          @��ÿ�Q�@���@W
=B�B�aH��Q�@w�@�33B5Q�B�\                                    Bx{��t  �          @��
����@�33@Z�HB��B�.����@|(�@�p�B5��B��)                                    Bx{��  
�          @����@�  @}p�B(Q�B��H���@`  @�z�BO
=Bՙ�                                    Bx{���  �          @�  �(�@g�@�\)BP�RB�
=�(�@/\)@�G�Bx(�B���                                    Bx{��f  
�          @�\)�=p�@e�@�\)BQ(�B�ff�=p�@,��@���Bx=qBҳ3                                    Bx{��  �          @��G�@?\)@�G�Bj
=B�LͿG�@�
@�\)B�.B�z�                                    Bx{���  �          @��H�^�R@=p�@�ffBh\)BԞ��^�R@33@�z�B�#�B���                                    Bx{�
X  �          @�(���@=p�@�G�Bl�B�#׿�@�\@�\)B�  B���                                    Bx{��  �          @��\�.{@c33?���A�33B��)�.{@P��?�p�A�{Cٚ                                    Bx{�'�  �          @�ff�Z�H@\)����eG�C�
�Z�H@�p��!G����
Cc�                                    Bx{�6J  T          @�\)�Tz�@Z�H�+���RC!H�Tz�@u��\��G�C�)                                    Bx{�D�  �          @�33�@��@_\)�*�H����C�{�@��@y���G���ffB�aH                                    Bx{�S�  T          @��R�HQ�@�녿޸R��B�\)�HQ�@�����  �*�\B�.                                    Bx{�b<  �          @����Q�@�p��/\)��B�u��Q�@��\��(����RB��H                                    Bx{�p�  �          @�(��
�H@�33�{��ffB�
=�
�H@�p����w�B��
                                    Bx{��  �          @�  �l(�@z�H>��R@S33CB��l(�@s�
?}p�A)��C�                                    Bx{��.  �          @��R�w
=@s�
���
�W
=C^��w
=@s�
>�33@n�RCff                                    Bx{���  �          @�ff�x��@qG�����=qC޸�x��@s33>B�\?�p�C�                                    Bx{��z  �          @�\)��z�@R�\�����e��C� ��z�@^�R�B�\��ffC�3                                    Bx{��   �          @����z�@`  �=p���Q�C�=��z�@e���G���Q�C0�                                    Bx{���  �          @�33�x��@y���z�H� ��C�R�x��@�Q쾏\)�6ffC+�                                    Bx{��l  �          @�����=q@p  ����9�C	c���=q@x�þ���ffCT{                                    Bx{��  �          @������@P�׿ٙ����C�����@`�׿����8��CxR                                    Bx{���  �          @�=q���
@@�׿�
=���HCh����
@P�׿����;33CB�                                    Bx{�^  �          @������@G���\)���C�3����@Y������T��C�{                                    Bx{�  �          @�33�r�\@l(�����  C��r�\@{��}p��$��C                                      Bx{� �  �          @������
@c33��G��x��C:����
@p�׿\(����C	�
                                    Bx{�/P  �          @����\)@\(���(���
=CG��\)@l(�����5�C	J=                                    Bx{�=�  T          @�Q���  @C33��Q���\)CY���  @S33�����=p�C.                                    Bx{�L�  �          @�����\)@5������CB���\)@I���Ǯ��{CT{                                    Bx{�[B  �          @������\@6ff����  C{���\@N�R�����p�C�)                                    Bx{�i�  �          @���~{@,(��*�H���C�
�~{@G��
=q��
=C��                                    Bx{�x�  �          @����k�@$z��K��G�C��k�@Fff�+����HC�                                    Bx{��4  �          @����Tz�@�
�dz��%�\C33�Tz�@9���Fff�ffC
�)                                    Bx{���  �          @�z��5�@��=q�D��C���5�@1G��hQ��*�C�{                                    Bx{���  �          @�33�.{@����H�HG�C�)�.{@1G��j=q�-33Cz�                                    Bx{��&  �          @����\@���(��Z�HCL���\@C�
�y���:ffB�ff                                    Bx{���  �          @�z��
=@Q����H�W(�C	�f�
=@7
=�x���9z�Cp�                                    Bx{��r  �          @���=q@�
�����V�C���=q@1��vff�933C޸                                    Bx{��  �          @��\�N{@��b�\�&�C)�N{@:�H�Dz��{C	�                                    Bx{���  �          @�=q�(Q�@(���=q�G��C+��(Q�@8Q��g
=�+=qCaH                                    Bx{��d  T          @���-p�?�ff��G��W��CQ��-p�@��|���@\)C��                                    Bx{�
  �          @���C33?����x���=��C�q�C33@!G��`  �&{C}q                                    Bx{��  �          @��H�<��?����(��K{C�3�<��@ff�qG��4(�CxR                                    Bx{�(V  �          @�G��#�
?�G����\�^Q�C��#�
@  ��  �F{C
��                                    Bx{�6�  �          @�(��O\)@1G��P�����Ck��O\)@S�
�-p����HC\)                                    Bx{�E�  �          @����C�
@2�\�Fff�ffC	���C�
@R�\�#33����C�f                                    Bx{�TH  �          @�z��!G�@�\)=u?&ffB�\�!G�@�z�?fffA�B��                                    Bx{�b�  �          @�z��Dz�@�  �����B�RB����Dz�@�(���33�p��B���                                    Bx{�q�  �          @�z��\(�@n{�����p�C���\(�@~�R����;�C�=                                    Bx{��:  �          @���X��@^{�  ���HCO\�X��@tz�˅��Q�C�
                                    Bx{���  �          @�(��4z�@vff�p����B�\)�4z�@�\)��(���  B�Q�                                    Bx{���  T          @���4z�@c33�=q���C ��4z�@z�H��p����
B��                                     Bx{��,  �          @�(��Q�@QG��p���  C��Q�@i����=q���
C�f                                    Bx{���  �          @����y��@K���ff��33C�=�y��@\�Ϳ�Q��N�RC
z�                                    Bx{��x  �          @�{�hQ�@u���H�MC^��hQ�@�  ���H��\)C@                                     Bx{��  �          @�\)�w�@j=q����]�C�
�w�@u�!G��ҏ\C@                                     Bx{���  �          @��R�~{@Y�������w�Ck��~{@g
=�O\)�	�C	�                                     Bx{��j  �          @�\)��33@TzῬ���f{C
=��33@`�׿:�H���RCxR                                    Bx{�  �          @�G����@Dz�u� ��C�����@L�;Ǯ��=qC��                                    Bx{��  �          @�=q��{@@�׿�(��u��Cٚ��{@N�R�fff�G�C��                                    Bx{�!\  �          @�(����\@5�У����\C0����\@E����2{C                                      Bx{�0  �          @������@   ��33���C\)����@0�׿�z��>ffC                                      Bx{�>�  �          @�
=���@&ff������C޸���@:�H���
�w�
C��                                    Bx{�MN  �          @�{��p�@������C�=��p�@%��33����C                                      Bx{�[�  �          @�p���?�\)�%��p�CE��@�
�p���33C�                                    Bx{�j�  �          @�����p�?��R�1G���{C"G���p�?�(��������C
                                    Bx{�y@  �          @���{@33�7���\)C8R��{@"�\����̣�C:�                                    Bx{���  �          @����\)?=p��8����
=C+���\)?�G��-p���C%{                                    Bx{���  �          @������?����AG��Q�C!����?�z��.{��{C��                                    Bx{��2  T          @�����?��\�B�\�
  C&�����?����333����C �                                    Bx{���  �          @�ff��  ?��\�I���Q�C#k���  ?����7
=���C��                                    Bx{��~  �          @�33��=q?}p��>�R���C'���=q?\�0  ����C �{                                    Bx{��$  �          @�=q�������
�>�R�
�C5
=����>���<(��C-�                                     Bx{���  �          @�{�\)?�  �]p�� \)C%��\)?�\)�N{�{C�f                                    Bx{��p  �          @���}p�?����_\)�!�C$�\�}p�?ٙ��N�R��\C�q                                    Bx{��  �          @����Tz��<(��*=q���
C]���Tz�����J=q���CW�\                                    Bx{��  �          @��\�Z=q�\(��Q���(�CaE�Z=q�?\)�.�R����C]8R                                    Bx{�b  �          @�����
���   �ݙ�CK���
���4z���=qCF�                                    Bx{�)  �          @�\)���ÿ�G��;��Q�CJQ����ÿ�
=�Mp��=qCCs3                                    Bx{�7�  �          @�\)���Ϳ�
=�<(����CF���Ϳ\(��I�����C?�                                    Bx{�FT  �          @�  ��  ��33�3�
��z�CEG���  �Y���AG���C>�R                                    Bx{�T�  �          @��
�������N�R�z�CCs3�����\�XQ���C;!H                                    Bx{�c�  �          @����z�H��z��W���CD� �z�H��\�a��'�C;k�                                    Bx{�rF  �          @��\���ÿ���mp��&��CBu����þ����u�.(�C8�H                                    Bx{���  �          @��H�g
=�:�H��ff�B�C?u��g
==�\)��Q��F�
C2�                                    Bx{���  �          @��
�fff�L����p��B(�C@z��fff    ��  �F�HC4                                      Bx{��8  �          @�ff�r�\��\)�X���&�RC8=q�r�\>�Q��XQ��&=qC.�{                                    Bx{���  �          @�G���{?n{�%��p�C((���{?�33�
=��\)C"}q                                    Bx{���  �          @��R����?�����\��ffC&n����?�G���\��ffC!��                                    Bx{��*  �          @������?����%���C$p�����?�z��33��=qC޸                                    Bx{���  
�          @��R���H?��\�+���=qC&�����H?�G��(����
C ޸                                    Bx{��v  �          @�
=���
?�
=�W��C$����
?���E�	�C}q                                    Bx{��  �          @�Q����H���R�?\)���C7�f���H>���?\)�33C0                                    Bx{��  �          @�Q����\>W
=�.{��\)C1h����\?8Q��(Q���z�C+�                                    Bx{�h  �          @�����Q�?����"�\��33C)��Q�@������\)C+�                                    Bx{�"  �          @��\�vff@
�H��R��z�C�H�vff@&ff�G���{C�R                                    Bx{�0�  �          @���vff@���$z����C���vff@&ff����C
=                                    Bx{�?Z  �          @��\�i��@*=q�G��ՙ�C�f�i��@C33��(����C#�                                    Bx{�N   �          @�p��c�
@���\(��C��c�
@7��<(����C#�                                    Bx{�\�  �          @���xQ�@<(��������C�\�xQ�@K��s33�+�C�)                                    Bx{�kL  �          @�33�|(�@L�Ϳ�33��{C�H�|(�@]p���  �-p�C
��                                    Bx{�y�  �          @����^�R@AG��(���p�C
���^�R@\(������HCO\                                    Bx{���  T          @�33�.�R@���aG��/ffC
��.�R@E�>�R��RCz�                                    Bx{��>  �          @��H�.�R@(��a��/�C
0��.�R@Dz��?\)�=qC�\                                    Bx{���  �          @�p�� ��@   �Z�H�0
=C{� ��@G
=�7���RC �{                                    Bx{���  �          @�z���@���\���4  C����@Dz��:=q�p�C ^�                                    Bx{��0  �          @��R�!G�@Q��b�\�6p�C���!G�@AG��@���C�f                                    Bx{���  �          @�
=�4z�@z��Vff�*=qC���4z�@;��5����C�H                                    Bx{��|  �          @����=q@z��y���K=qCh��=q@2�\�Z�H�+\)C�H                                    Bx{��"  �          @���Q�@(��s�
�@��C@ �Q�@H���P���ffB�Q�                                    Bx{���  �          @��
���@&ff�p���=��C�����@R�\�K��33B�                                    Bx{�n  �          @�z���@-p��tz��A  B�����@Y���N{���B�                                    Bx{�  �          @���p�@\)�vff�D�\C	���p�@=p��U��#�CǮ                                    Bx{�)�  �          @�=q��@����H�V��C����@8���e�4\)B��                                    Bx{�8`  �          @�=q���@	����Q��QC�����@9���`  �/�B�ff                                    Bx{�G  
�          @�����@	����G��W\)Ch���@9���b�\�3�HB��                                    Bx{�U�  �          @��R���@(������Y��C�׿��@<(��`���4��B��                                    Bx{�dR  �          @��R���@  �\)�Vp�C  ���@@  �]p��1\)B�p�                                    Bx{�r�  �          @�
=��33@�������Xz�C�q��33@=p��`  �3��B�aH                                    Bx{���  �          @����=q@���~{�S��B��H��=q@HQ��Z�H�-\)B��\                                    Bx{��D  �          @�p���Q�@���z=q�R�B�=q��Q�@K��U�+�B�
=                                    Bx{���  �          @���  @#�
�s�
�K�B��׿�  @QG��N�R�$(�B�8R                                    Bx{���  �          @�����R@
=q��G��d  B�\���R@;��a��<33B��H                                    Bx{��6  �          @������@���Q��0=qCǮ���@@���.�R��C aH                                    Bx{���  �          @�=q�	��@XQ���R���
B�\�	��@s�
��  ��=qB�                                      Bx{�ق  �          @�ff���@����
=q��33B��f���@�=q>\@�
=B�                                    Bx{��(  �          @�
=��\@��
��
=�[�
B�  ��\@��׾�=q�J=qB�\                                    Bx{���  �          @���G�@�G��������B�L��G�@�  �����33B�G�                                    Bx{�t  
�          @�=q��@vff�
�H���B�B���@��R����x��B�Ǯ                                    Bx{�  �          @���
�H@|(��  ��ffB��q�
�H@�=q��33��p�B�u�                                    Bx{�"�  �          @��
�ff@u��(���{B�#��ff@�  ��{���HB���                                    Bx{�1f  �          @�z����@j=q�+���{B�����@�(������  B�z�                                    Bx{�@  �          @����33@[��8Q���\B���33@|������
=B�W
                                    Bx{�N�  �          @�Q��P  @���;���C��P  @0  ��H����C�                                     Bx{�]X  �          @���[�?��
�Tz��(��C�{�[�?�
=�?\)�C�H                                    Bx{�k�  �          @�Q��b�\?����S33�$�\C���b�\?�(��=p���C޸                                    Bx{�z�  �          @�Q��]p�@ ���'
=���HC\�]p�@?\)������C(�                                    Bx{��J  �          @�Q��P��@!��6ff�
�\C+��P��@C�
�G���
=C�                                    Bx{���  �          @�G��.{@�Q쿨���tQ�B�W
�.{@�{�����B��                                    Bx{���  �          @����.�R@n{��(����HB��{�.�R@�����z��U�B�
=                                    Bx{��<  �          @�\)�/\)@HQ��,(����C(��/\)@hQ��p����RB�(�                                    Bx{���  �          @���� ��@L���)���G�C {� ��@k������HB��                                    Bx{�҈  �          @�\)�:=q@?\)�+���HC8R�:=q@^�R�   ��{C޸                                    Bx{��.  �          @����>{@%��B�\�
=C��>{@I���(�����C:�                                    Bx{���  �          @��H�L(�@Vff�����Q�C���L(�@n�R����|��C�
                                    Bx{��z  �          @�33�0  @mp��Q����B���0  @�=q��ff�mG�B�\                                    Bx{�   �          @�(��>�R@e��
����C���>�R@|�Ϳ�G��d��B�                                    Bx{��  �          @�
=�S33@  �:�H��RC��S33@4z��Q���{C�                                     Bx{�*l  �          @�{�W
=?�(��@  ��\C�{�W
=@#�
�!G���z�C��                                    Bx{�9  �          @��H�p  ?.{�Vff�%=qC)� �p  ?�\)�HQ��\)C��                                    Bx{�G�  �          @��\�o\)?���W
=�&Q�C*��o\)?�ff�J=q�G�C ��                                    Bx{�V^  �          @�=q�u�?^�R�L����
C'@ �u�?��
�<(��33C@                                     Bx{�e  �          @����Tz�@XQ쿾�R��(�CxR�Tz�@g��=p��	G�C�                                     Bx{�s�  T          @�G��Y��@G���(�����C	aH�Y��@^{��G��hQ�CY�                                    Bx{��P  �          @��H�9��@h������Q�C �\�9��@\)�����[
=B�\                                    Bx{���  �          @�����@X���3�
���B�G����@z=q�G���=qB�q                                    Bx{���  �          @�33�4z�@_\)�p���p�C ޸�4z�@x�ÿ�33��ffB��
                                    Bx{��B  �          @���A�@Q�����z�C�3�A�@mp��������C@                                     Bx{���  �          @�\)�L(�@Tz῎{�Y��C޸�L(�@^�R��p����C�                                     Bx{�ˎ  �          @��\�mp�@X��=��
?fffC	���mp�@S33?O\)Az�C
ff                                    Bx{��4  �          @�p��e�@hQ�333���HC�\�e�@l��=���?�33C�                                    Bx{���  �          @���L��@`  ���R��ffCk��L��@vff��Q��U�C�R                                    Bx{���  �          @�G��7
=@O\)�C33��Ck��7
=@u��G���p�B��                                    Bx{�&  �          @��R�'�@E�P  ��
CJ=�'�@n�R�\)�噚B�=q                                    Bx{��  �          @��R�+�@0  �aG��*{CE�+�@\���5����B���                                    Bx{�#r  �          @��R�A�@)���P�����C
�)�A�@S33�&ff���HC�{                                    Bx{�2  �          @��R�J=q@%��L(����C���J=q@N{�"�\���Cs3                                    Bx{�@�  �          @�p��333@   �\(��*(�C
0��333@L���3�
���C(�                                    Bx{�Od  �          @�
=��p�@Q��aG��*Q�B�Q��p�@~�R�-p���
=B�                                      Bx{�^
  �          @��G�@,���y���D�
B�z��G�@`  �Mp����B��f                                    Bx{�l�  �          @�ff�=q@U��G
=��B�Ǯ�=q@|(��33���
B��                                    Bx{�{V  �          @������@E��U�"  B��
���@p  �$z����
B�L�                                    Bx{���  �          @��\���@%�h���7�RC�=���@U��>{��B�aH                                    Bx{���  �          @��H�$z�@S33�9���
��B����$z�@w�����
B��                                    Bx{��H  �          @��\�,(�@0���>{���CG��,(�@Vff����\)C ��                                    Bx{���  �          @���=p�@I���#33��{C0��=p�@i�����
���C{                                    Bx{�Ĕ  �          @�(���33@�{��
���HB�
=��33@��������G
=B��                                    Bx{��:  T          @��H����@�{��
��  B�����@������o�
B�G�                                    Bx{���  �          @�33�#�
@�  ��=q����B�=q�#�
@�녿aG���B�k�                                    Bx{���  �          @�
=�,��>u�����c�C.�R�,��?��\)�V\)C��                                    Bx{��,  �          @�33�8��@!��z����C
���8��@?\)��Q�����C�                                    Bx{��  �          @�\)�8Q�@g��(�����C ���8Q�@��ÿ���m�B�33                                    Bx{�x  �          @�����p�@<���p  �Bz�B�=q��p�@n�R�>�R�33B�G�                                    Bx{�+  �          @��Ϳ�  @H���|���H�RB�LͿ�  @}p��HQ����B�Q�                                    Bx{�9�  �          @�
=��ff@dz��^{�&�B��H��ff@�Q��#�
���B��)                                    Bx{�Hj  �          @�  �%@�  �
�H��{B��f�%@�zῙ���S
=B��                                    Bx{�W  �          @�����@�{�Ǯ���B�Ǯ���@��   ��\)B�ff                                    Bx{�e�  �          @�{�ff@���=�Q�?��B��)�ff@�z�?��AN=qB�ff                                    Bx{�t\  �          @�p���@�G���G����\B�G���@�{?p��A(Q�B�Q�                                    Bx{��  �          @����
@�33=L��?�B�aH��
@��R?�{AG�
B�Ǯ                                    Bx{���  T          @�ff�Q�@�33>�p�@�G�B��Q�@�z�?�
=A
=B��f                                    Bx{��N  �          @����{@�Q�?\(�AG�B�Ǯ��{@�?���A��RB��                                    Bx{���  �          @������@�Q�?�=qA���B������@�Q�@)��A�  B�B�                                    Bx{���  �          @�{����@�z�?���Ah��B��Ϳ���@�@p�AمB��{                                    Bx{��@  �          @��R�0��@��;u� ��B�k��0��@�=q?\(�A�\B�8R                                    Bx{���  �          @��R�8Q�@�ff��(���p�B��H�8Q�@���.{���B�aH                                    Bx{��  �          @�  �:=q@����{��B�L��:=q@��R�����J�HB�.                                    Bx{��2  �          @�\)�@  @i���0����  Ch��@  @�ff�����HB�{                                    Bx{��  �          @���\��@z=q����Ch��\��@�Q�p����C�                                    Bx{�~  �          @�Q��L(�@qG�����G�CE�L(�@�����H�v=qB�                                      Bx{�$$  �          @��
�.�R@HQ��Y���=qC&f�.�R@vff�#33�㙚B�                                    Bx{�2�  �          @����'�@\)�{��C�C���'�@G
=�R�\���C
                                    Bx{�Ap  �          @�Q��333@
=q�tz��=  CL��333@@���L(��\)Cٚ                                    Bx{�P  
�          @�\)��R@���G��L��C33��R@QG��W
=�!=qB�u�                                    Bx{�^�  �          @�
=��33@!G����
�Rz�B�\��33@[��Y���#�\B��
                                    Bx{�mb  �          @���
@{��G��M��C޸��
@W��U�� \)B��f                                    Bx{�|  �          @�
=�/\)@3�
�Y���#�C@ �/\)@c33�'���{B�Q�                                    Bx{���  �          @�Q��AG�@C�
�AG��ffC�H�AG�@l���(�����C8R                                    Bx{��T  �          @�
=��Q�@N{�dz��.G�B�8R��Q�@\)�,(���p�B��f                                    Bx{���  �          @�
=���@/\)�l(��4z�CǮ���@c�
�:=q���B�{                                    Bx{���  �          @�Q��,(�@���tz��;  C
k��,(�@P  �HQ��{C�H                                    Bx{��F  �          @����?\)@���n�R�3ffC�3�?\)@G
=�C�
��\C�3                                    Bx{���  �          @����'�?�
=����O�RC���'�@8���c�
�)\)C:�                                    Bx{��  �          @�����?�
=���
�t  C���@!G���(��Mp�C{                                    Bx{��8  T          @����
�H?�����H�s�HCE�
�H@����
�N��C��                                    Bx{���  T          @���У�?�33������CE�У�@"�\�����[p�B�L�                                    Bx{��  T          @�{��
=?��H�w��Z\)C�=��
=@7
=�P���-��B�
=                                    Bx{�*  �          @����E�@xQ�������C s3�E�@��R�L���
�HB�ff                                    Bx{�+�  �          @��\�J=q@w
=�������HC\)�J=q@�ff�Tz���
B�                                    Bx{�:v  �          @����E�@c�
�ff��(�C���E�@��ÿ���tQ�B�                                    Bx{�I  �          @���@  @i���\)�ʸRCaH�@  @�33��  �]B�ff                                    Bx{�W�  �          @�33�=p�@n�R�(���33C aH�=p�@�
=��
=�w�B���                                    Bx{�fh  �          @���?\)@r�\�33��  C L��?\)@�  ���
�\��B�W
                                    Bx{�u  �          @���C33@p  ����ffC(��C33@�ff���\�[
=B���                                    Bx{���  �          @��b�\@z�H���H  C��b�\@��H�\)��  C�H                                    Bx{��Z  �          @��G
=@g��!G��ޏ\C���G
=@�z�\����B��H                                    Bx{��   �          @��Dz�@Z=q�9���Q�C���Dz�@��ÿ�Q�����B�z�                                    Bx{���  
�          @��R�HQ�@Tz��@  �=qCJ=�HQ�@~{�33���RC 33                                    Bx{��L  �          @�{�?\)@U��E��
�C���?\)@�Q��Q���Q�B�p�                                    Bx{���  �          @�\)�Fff@^�R�6ff��\)C���Fff@��H��{��p�B�G�                                    Bx{�ۘ  �          @��R�H��@fff�(���z�C&f�H��@�33��Q��xQ�B��                                    Bx{��>  �          @�ff�:�H@^{�@���\)C{�:�H@�(��G���{B��=                                    Bx{���  �          @�ff�R�\@j�H��R��=qC�f�R�\@��
�����Mp�C ��                                    Bx{��  �          @��
�@��@HQ��J=q�(�C���@��@u�\)�ř�C (�                                    Bx{�0  �          @���?\)@@���QG��ffC�)�?\)@p  �Q��ҸRC �{                                    Bx{�$�  �          @��H�1G�@>�R�\���!�C��1G�@qG��#�
��
=B���                                    Bx{�3|  �          @����>�R@G��B�\��RC� �>�R@s33�����
C !H                                    Bx{�B"  �          @����333@?\)�E�33C
=�333@l�������\)B�33                                    Bx{�P�  �          @�(��.{@C�
�E��C�)�.{@qG�����G�B��3                                    Bx{�_n  
�          @��R�"�\@P  �K��z�C {�"�\@~{�{��p�B�W
                                    Bx{�n  �          @�Q��(��@AG��Z=q�!Q�C!H�(��@s�
�\)��{B�ff                                    Bx{�|�  �          @����5@G
=�Mp��  Cs3�5@u�G���G�B�                                      Bx{��`  �          @��H�A�@E��HQ����C���A�@s33�p���33C �)                                    Bx{��  �          @��;�@C33�W���C�{�;�@u��(���z�B��q                                    Bx{���  �          @���Mp�@B�\�Fff�  C�{�Mp�@p������Q�C�\                                    Bx{��R  �          @��
�G
=@B�\�HQ���C���G
=@p���p�����C��                                    Bx{���  �          @�p��S�
@p  �z����Cff�S�
@�p��}p��(��C p�                                    Bx{�Ԟ  �          @��R�X��@e��
�ȏ\CT{�X��@��\��G��V�\C��                                    Bx{��D  �          @�Q��^{@j=q�{��  C��^{@��
��z��A�C#�                                    Bx{���  �          @��R�Vff@k��{��=qCL��Vff@�zΐ33�B�\C ��                                    Bx{� �  �          @��O\)@p  �p���(�C���O\)@��R��{�=B�(�                                    Bx{�6  �          @��P  @{��������
C�=�P  @�G��8Q�����B�B�                                    Bx{��  
�          @�  �Y��@~{��Q����C���Y��@��ÿ������C p�                                    Bx{�,�  �          @����W
=@�  �����8Q�C J=�W
=@�z�=�Q�?uB�Ǯ                                    Bx{�;(  �          @����[�@�{����5C@ �[�@��\=�Q�?z�HC W
                                    Bx{�I�  �          @�\)�dz�@|�Ϳ�ff�\��C)�dz�@��;8Q��\)C��                                    Bx{�Xt  �          @�  �i��@x�ÿ����c�C+��i��@���u�p�C�
                                    Bx{�g  �          @�Q��b�\@��\�����8(�C�R�b�\@�
==u?!G�C�R                                    Bx{�u�  �          @�  �]p�@���p�����CǮ�]p�@�Q�>u@#33C!H                                    Bx{��f  �          @����`  @�z�z�H�#�C5��`  @�  >L��@Cz�                                    Bx{��  �          @�  �r�\@q녿�(��M�C��r�\@~{����\)C��                                    Bx{���  �          @�  �xQ�@l�Ϳ�z��C�CQ��xQ�@xQ콸Q�s33C��                                    Bx{��X  �          @���mp�@�  �&ff�أ�C�
�mp�@���>��H@�33C�                                    Bx{���  �          @�Q��p  @}p��E�� Q�Cs3�p  @�Q�>�Q�@r�\C
=                                    Bx{�ͤ  �          @����i��@���z����C���i��@��?
=@��HC��                                    Bx{��J  T          @����XQ�@��
��Q�z�HB�u��XQ�@�
=?���A<��C ��                                    Bx{���  �          @�\)�W�@�녾#�
��C ��W�@�{?��A1�C �H                                    Bx{���  �          @����Q�@�\)��Q�aG�B�ff�Q�@�=q?�
=AC�
B�W
                                    Bx{�<  �          @���L��@�������{B�B��L��@��
?�  AO�B�aH                                    Bx{��  �          @����U@�=q�^�R���B�u��U@�z�>\@|(�B��=                                    Bx{�%�  �          @�ff�U�@�z῕�F=qC �)�U�@���=L��?��B���                                    Bx{�4.  �          @�G��W
=@{������?�C�\�W
=@�=q=u?0��C��                                    Bx{�B�  �          @�G��\��@~�R������C�H�\��@~{?�R@�  C�R                                    Bx{�Qz  �          @�=q�W�@�=q>L��@
�HC�)�W�@u?���Am�C:�                                    Bx{�`   �          @�33�QG�@��?B�\AffC ��QG�@qG�?�A�{C��                                    Bx{�n�  �          @����Q�@���?8Q�@��
C �q�Q�@k�?���A��RC�                                     Bx{�}l  �          @����P  @~{>�ff@��CG��P  @k�?�=qA��\Cz�                                    Bx{��  �          @����c33@j�H=#�
>��C
=�c33@`  ?���AFffC^�                                    Bx{���  T          @�����G�@�
�
=q��=qC\��G�@�>��?��Cz�                                    Bx{��^  �          @�ff�hQ�@l(���{�o\)C�{�hQ�@hQ�?=p�A�
C�                                    Bx{��  �          @���S33@Z�H��=q����C�S33@s33�B�\�	C��                                    Bx{�ƪ  �          @�z��N�R@`  ��33��Q�C���N�R@y���L���Q�C�)                                    Bx{��P  �          @�Q��]p�@r�\��  �1G�CaH�]p�@z�H>\)?���Cn                                    Bx{���  �          @����fff@u���p���G�CB��fff@qG�?B�\A�C�3                                    Bx{��  �          @����p  @Z=q��z��xz�C	�R�p  @k���33�s33C��                                    Bx{�B  �          @�G��hQ�@hQ쿢�\�]�C  �hQ�@u����˅C^�                                    Bx{��  �          @��\�qG�@`�׿����k�C	��qG�@p  ����333C�                                    Bx{��  �          @����p��@XQ�˅��\)C
)�p��@l(�����(�C�=                                    Bx{�-4  �          @����mp�@X�ÿ�����\C	�)�mp�@n{����z�C�                                    Bx{�;�  T          @�  �p  @H�ÿ�{����C
�p  @c33�Y����C��                                    Bx{�J�  �          @���xQ�@%�����ř�CaH�xQ�@Fff���
�g33CaH                                    Bx{�Y&  �          @��R�l(�@Fff��
=��z�C���l(�@b�\�k��"�HC:�                                    Bx{�g�  �          @�  �o\)@J=q�������
C�{�o\)@dz�Tz��G�C\)                                    Bx{�vr  �          @����vff@>�R�����RC33�vff@^{����=C	�R                                    Bx{��  �          @��
�|��@G������RC���|��@a녿Q���C
@                                     Bx{���  �          @��
��  @1G�����CJ=��  @QG���z��J{C�q                                    Bx{��d  �          @�
=�U@��ÿ.{��(�B���U@�G�?#�
@�\)B��
                                    Bx{��
  �          @�ff�k�@w
=��ff�1�C�H�k�@�  >.{?�  C��                                    Bx{���  �          @�(��vff@`�׿�=q�d(�C	�H�vff@p  �L���Q�C�                                    Bx{��V  �          @�z�����@W������MC�����@dz����G�C
\)                                    Bx{���  �          @�{���@>�R��������Ck����@X�ÿTz����C�{                                    Bx{��  �          @�p���p�@;�����33C���p�@Vff�\(���HC:�                                    Bx{��H  T          @�z�����@+���=q���HC�����@G
=�k��{C#�                                    Bx{��  �          @�(���@1녿���p�CT{��@O\)�z�H�'\)C:�                                    Bx{��  �          @�p�����@
=���ˮC#�����@=p����R��CW
                                    Bx{�&:  �          @��H��?�(��(�����HC$����?�(������=qC\                                    Bx{�4�  �          @�Q����?J=q�8Q��Q�C)����?���   ��C:�                                    Bx{�C�  �          @�����G�?B�\�>{�	
=C*���G�?���%��G�C!H                                    Bx{�R,  �          @��\���H?fff�>{�33C(B����H?��
�"�\��C�q                                    Bx{�`�  �          @����=q?����>{�C%����=q?�(���R���HCu�                                    Bx{�ox  �          @�(���
=?��
�,(���{C#����
=@�
�
=q���CO\                                    Bx{�~  �          @�33����?aG��7��p�C(� ����?�p��p��ۅC�{                                    Bx{���  T          @������H?L���<(���C)�����H?�
=�"�\����C��                                    Bx{��j  
�          @�������?   �@  �
�C-T{����?��,(����C!�                                     Bx{��  
�          @����{?����*=q���C%����{?�z��
�H��p�C�3                                    Bx{���  T          @��\��{?�33��H��z�C����{@����
��=qC\)                                    Bx{��\  �          @��
��Q�?������ȸRC{��Q�@(��˅����C�
                                    Bx{��  �          @�=q���?�33�{��C&f���@%��  ��33C                                      Bx{��  �          @�����\)?�
=���ڸRC"W
��\)@���{��  C��                                    Bx{��N  �          @�\)����?}p��=q��(�C'�3����?ٙ���(���Cc�                                    Bx{��  T          @�����{>#�
�ff����C1����{?fff����33C(��                                    Bx{��  �          @�����ff>����\)�߅C.����ff?�����R��\)C%k�                                    Bx{�@  �          @�Q����?&ff��\��RC+����?��Ϳ��H��G�C!�{                                    Bx{�-�  �          @�ff�\)?Y�������C'���\)?�ff��
=��Q�C�q                                    Bx{�<�  �          @�(��}p�?s33�������C&}q�}p�?�{���
���\C�                                    Bx{�K2  �          @�p�����?���33�Џ\C$0�����?޸R�������C��                                    Bx{�Y�  T          @�33����?�������p�C$����?�p���(����HC��                                    Bx{�h~  �          @�ff���H?��R��=q��(�C �q���H@ �׿�  �j�\C.                                    Bx{�w$  �          @�p���33?�G���z����HC#ٚ��33?���33��=qCh�                                    Bx{���  �          @�\)���\?\��\)���C �
���\@33���\�mC��                                    Bx{��p  �          @�ff�hQ�@0�׿������CǮ�hQ�@N{�aG��%�C
k�                                    Bx{��  �          @��R�vff@(Q��\)���C���vff@@�׿.{���
C�                                    Bx{���  �          @�Q��^{@U��z��XQ�C��^{@a�    ��Cz�                                    Bx{��b  �          @�33�l(�@Q녿����IG�C
^��l(�@]p�<�>\Cٚ                                    Bx{��  �          @�Q��L��@s�
�u�#�
C�q�L��@g�?���A_33Cu�                                    Bx{�ݮ  �          @�p��0��@s�
=�G�?��B����0��@c�
?���A�B���                                    Bx{��T  �          @�{�E�@aG�������ffC.�E�@u���
�n�RC �R                                    Bx{���  �          @����$z�@u��n{�5B����$z�@z�H>Ǯ@�Q�B��\                                    Bx{�	�  �          @��R��=q@���W
=�!G�B�z��=q@�\)?��\Az�\B�                                    Bx{�F  �          @�{�33@�p���
=��p�B�u��33@�=q?}p�AC\)B�                                    Bx{�&�  
�          @�ff��@��ÿ�������B���@��ü��
��z�B�aH                                    Bx{�5�  T          @�{� ��@\�Ϳ�33��ffB��� ��@x�ÿ#�
��p�B��{                                    Bx{�D8  �          @��R�@aG�����B�33�@�Q�J=q���B�u�                                    Bx{�R�  �          @�  �7
=@<(��Q����C.�7
=@dzῡG��vffC ��                                    Bx{�a�  �          @��R�E�@<(��   ��G�CQ��E�@g
=��\)����Cz�                                    Bx{�p*  �          @��
�HQ�@O\)�����HC��HQ�@w������Y�C ��                                    Bx{�~�  �          @�33�U@Z=q�(���Q�Cp��U@��ÿ�33�E��C��                                    Bx{��v  �          @��H�Mp�@aG������
=C^��Mp�@��
����7�B��)                                    Bx{��  �          @�=q�Fff@`��� �����
Cs3�Fff@��Ϳ�
=�L��B�u�                                    Bx{���  �          @����H��@W��#�
��RC
=�H��@�G����\�_
=B���                                    Bx{��h  �          @�G��Z=q@HQ��#�
��C	u��Z=q@s�
�����l  C��                                    Bx{��  �          @��\�_\)@<���0  ���\CǮ�_\)@mp���=q��=qC:�                                    Bx{�ִ  �          @��H�\(�@Fff�*=q��G�C	��\(�@u������{33C�3                                    Bx{��Z  �          @����e@Mp������C
5��e@vff��
=�J�HC
=                                    Bx{��   �          @��R�j=q@G
=�%��\C���j=q@s�
��\)�h(�C�H                                    Bx{��  �          @�ff�g�@@���0  ��\CE�g�@qG���ff��Q�C�{                                    Bx{�L  �          @�
=�o\)@8Q��0����Q�C^��o\)@i���������C��                                    Bx{��  �          @�Q��r�\@5��5����C@ �r�\@hQ��
=��  C8R                                    Bx{�.�  �          @��\�u�@1G��=p�� �C��u�@g�������=qC��                                    Bx{�=>  �          @����mp�@:=q�9������C�
�mp�@o\)���H��Q�C�                                    Bx{�K�  �          @����]p�@N{�6ff���\C	
�]p�@�Q�Ǯ����C�{                                    Bx{�Z�  T          @����^�R@o\)�Q�����C  �^�R@���333��\)Cc�                                    Bx{�i0  �          @����b�\@Z=q�#33�ۅC�b�\@��H�����H��C�)                                    Bx{�w�  �          @�Q��h��@I���*�H��{C(��h��@xQ쿳33�lQ�C+�                                    Bx{��|  
�          @���qG�@333�5�����CY��qG�@g�����(�C+�                                    Bx{��"  �          @���r�\@7��.�R���Cٚ�r�\@i����ff��33C�                                    Bx{���  �          @�  �|��@'
=�333��z�Cz��|��@[��ٙ���{C                                    Bx{��n  �          @�ff�s33@333�.{��\)C�f�s33@e��Ǯ����C                                    Bx{��  �          @�{�qG�@8Q��*=q��(�C���qG�@hQ쿼(��{\)C
=                                    Bx{�Ϻ  �          @�p��mp�@5��0  ��C���mp�@g���������C�                                    Bx{��`  �          @�\)�n�R@4z��6ff���RC��n�R@i�������C�                                    Bx{��  �          @�G��s�
@4z��6ff��{Cz��s�
@i������{C=q                                    Bx{���  �          @�=q�xQ�@2�\�7
=��
=C=q�xQ�@hQ��
=��ffC�                                    Bx{�
R  �          @��H�~�R@0���2�\���CE�~�R@dz��\)����C
�                                    Bx{��  T          @�33�vff@5��:�H���
C���vff@k���(���G�C8R                                    Bx{�'�  �          @��
�j�H@333�K��
33C���j�H@p�׿�(���ffCY�                                    Bx{�6D  �          @��H�j=q@2�\�J=q�
  C�R�j=q@o\)���H���Cn                                    Bx{�D�  S          @��H�Z�H@,(��`  ��C� �Z�H@qG��33��Q�C33                                    Bx{�S�  �          @�z��aG�@-p��^{��HCaH�aG�@q������33C�R                                    Bx{�b6  �          @�z��e@(Q��]p��p�C�R�e@l�������RC�                                    Bx{�p�  �          @����k�@#�
�\(���C)�k�@hQ�����ffCY�                                    Bx{��  �          @�  �mp�@(Q��`�����C���mp�@n{�z�����C�                                    Bx{��(  �          @��R�\(�@ ���r�\�&�C��\(�@mp��(Q���G�Cٚ                                    Bx{���  �          @�ff�\(�@��tz��(�
C� �\(�@j=q�+���RCG�                                    Bx{��t  �          @�z��^�R@!G��h���!�C��^�R@j�H��R�Џ\Cs3                                    Bx{��  �          @��\�\��@!��dz��\)C���\��@j=q�����Q�CL�                                    Bx{���  �          @�  �g�@(��U��C�q�g�@_\)�{��ffC                                    Bx{��f  �          @�ff�fff@��Y���G�C�
�fff@W�����=qC�H                                    Bx{��  �          @�(��K�@
�H�o\)�0��C��K�@X���,(���(�C:�                                    Bx{���  �          @����@��@���|���;��C�
�@��@\(��8���
=C0�                                    Bx{�X  
�          @�\)�5�@ ������J��C��5�@[��Mp��(�C�{                                    Bx{��  �          @�G��G
=@   ����@=qC0��G
=@XQ��E��C��                                    Bx{� �  �          @�\)�333?��H��(��T�\C���333@L���\(��C8R                                    Bx{�/J  �          @�  �0��?�ff�����T��C޸�0��@S33�Z�H���C�f                                    Bx{�=�  �          @��R�33?�(���z��i  C.�33@Tz��j�H�)(�B�p�                                    Bx{�L�  �          @����{?����(��l��C�\�{@O\)�l(��,�
B���                                    Bx{�[<  �          @��H�(�?�Q�����l{C�(�@<���`���.��B��                                    Bx{�i�  �          @��ÿ�p�?�����33C���p�@'��l���Kp�B��H                                    Bx{�x�  �          @���z�?�=q���H�w=qC33��z�@6ff�a��7=qB���                                    Bx{��.  �          @�33�\)?�(���(��i{C���\)@*�H�W��0(�C                                    Bx{���  �          @�����?�z���ff�o�\Cz����@)���]p��5�RC �f                                    Bx{��z  �          @�
=��\?�p���=q�n��C�{��\@*�H�S�
�2Q�B��3                                    Bx{��   T          @�����?��\�u�_  C����@'��E��&
=C�                                    Bx{���  T          @���
�H?�=q�vff�bG�Cn�
�H@+��Dz��&�\C ��                                    Bx{��l  �          @��R��z�?�=q��Q�\)C����z�@0  �Mp��8p�B�Q�                                    Bx{��  �          @���
=?�\)����zG�Ck���
=@Dz��O\)�.�HB��H                                    Bx{���  �          @����Q�?\��G��xG�C	�R��Q�@B�\�Y���1\)B�8R                                    Bx{��^  �          @�����?�33����uCG����@<(��]p��3{B�W
                                    Bx{�  �          @�33��p�?�\)���R�p�CLͿ�p�@7��XQ��0G�B�=q                                    Bx{��  �          @�G��?�p����
�m�
C}q�@-p��U�1�B�ff                                    Bx{�(P  �          @��R� ��?����H�q�RC޸� ��@(���U�4B��\                                    Bx{�6�  �          @��
�{@(���333�G�C+��{@`�׿˅��=qB�\)                                    Bx{�E�  �          @�ff�*=q@�!G����C
���*=q@G���
=��(�CaH                                    Bx{�TB  �          @�  �G�@,(��(����\C(��G�@_\)��z���\)B���                                    Bx{�b�  T          @�  ��Q�?�Q��|���yp�B�33��Q�@E��?\)�(��B�=q                                    Bx{�q�  T          @��þ\?˅��ff��B��;\@E��QG��9��B���                                    Bx{��4  �          @�Q�W
=?���p�¥�{B�  �W
=@��xQ��r(�B���                                    Bx{���  �          @�=�\)�E����\ �
C��=�\)?��������B�.                                    Bx{���  �          @��\?\)������R\)C�g�?\)=�����¥��A@��                                    Bx{��&  �          @��;8Q�L�����H­L�Cb�H�8Q�?������\B�.                                    Bx{���  �          @�Q�?333��33���fC�N?333>���������Aə�                                    Bx{��r  �          @��
?xQ�����XQ��R��C��?xQ�c�
��G�{C���                                    Bx{��  �          @�z�����R�aG��\Cx����>k��n�R£��C�\                                    Bx{��  �          @�  ?�ff�G��L���E=qC�(�?�ff�W
=�u��C��
                                    Bx{��d  �          @�?�Q��8���(��� ��C�0�?�Q�У��c�
�q�
C�3                                    Bx{�
  �          @�p�@z��z��@  �:�HC���@z�
=�`  �i��C��                                    Bx{��  
�          @\(�?�p���  �!G��6�RC�P�?�p���R�@  �j33C��                                    Bx{�!V  �          @[�?��H��z�����%�C��)?��H�^�R�5��_G�C���                                    Bx{�/�  �          @E�?�z��z���H��\C�p�?�z�B�\�p��U�
C��3                                    Bx{�>�  �          @9��?��Ϳ�  ����� �HC�b�?��Ϳ&ff��\�TQ�C��                                    Bx{�MH  �          @(��?�p������G��'33C�s3?�p����H���V�RC��f                                    Bx{�[�  �          @/\)?�  ��
=���
�#
=C�9�?�  �(�����V��C��                                    Bx{�j�  �          @�R?�=q��{��=q���C�C�?�=q��R���R�U��C���                                    Bx{�y:  �          @Q�?&ff��G���\)�S(�C���?&ff��
=�{�=C���                                    Bx{���  T          @ ��?8Q쿮{��
=�M�HC���?8Q����
  C�
                                    Bx{���  �          @�>�ff�z�H����q\)C�N>�ff����Q���C��3                                    Bx{��,  �          @�>�  �s33��\��C�t{>�  �u�  ¦�)C���                                    Bx{���  �          @3�
?����\)���R�6=qC��{?���0���{�y�RC��=                                    Bx{��x  �          @n{@   ������C�\)@   �����2�\�E=qC�H�                                    Bx{��  �          @^�R?�p��G���p���
=C���?�p�����\)�7�C��H                                    Bx{���  �          @w
=@��'
=������Q�C�N@���{�5��<ffC�*=                                    Bx{��j  �          @�\)@��:�H�{��C�G�@���Q��[��H�C�
=                                    Bx{��  �          @�z�@!��H���33��\)C�l�@!녿����W
=�9��C�+�                                    Bx{��  �          @��@��E��\)��HC��f@����aG��G
=C�C�                                    Bx{�\  �          @��\@{�E��33���
C�^�@{����Vff�<\)C�J=                                    Bx{�)  �          @��@���:�H��R��=qC���@����\�N{�<�C���                                    Bx{�7�  �          @���@0���>{�������
C�|)@0�����(���z�C��=                                    Bx{�FN  �          @�ff@>�R�(Q�?���AqC�K�@>�R�5���
��z�C�4{                                    Bx{�T�  �          @��
@0  �0  �.{��
C��f@0  ��R�޸R��33C�w
                                    Bx{�c�  �          @�@Mp��1G�������  C���@Mp���(��Q��=qC�AH                                    Bx{�r@  �          @�z�@{���
�.{�
=C�Ǯ@{���\����]�C�Ff                                    Bx{���  �          @�  @^�R�L(�����p�C�� @^�R�,(����
��z�C�q                                    Bx{���  �          @�=q@N{�HQ�G��p�C��@N{�"�\���R����C��H                                    Bx{��2  �          @�
=@Fff�'
=����q��C���@Fff��
=��
��C�                                    Bx{���  �          @�@L(��Q쿡G����HC�� @L(���33�Q���
=C�G�                                    Bx{��~  �          @�
=@J�H�1G�>\)?�C�p�@J�H�%���G��^=qC�n                                    Bx{��$  �          @�\)@QG��(��>��@�Q�C��=@QG��$z�5��C��                                    Bx{���  �          @�  @Vff�%�u�J=qC�"�@Vff���\)�t��C��=                                    Bx{��p  �          @���@\(���R?��@�C�{@\(��\)���޸RC�
=                                    Bx{��  �          @���@N�R�/\)?
=q@�
=C��{@N�R�.{�#�
�
=qC���                                    Bx{��  �          @���@Q��Q�?���Am�C��)@Q��'
=�#�
����C��
                                    Bx{�b  �          @��
@C�
����z���z�C�)@C�
�����G���C�aH                                    Bx{�"  �          @�33@>{��Ϳ�\)��z�C�5�@>{������ffC�U�                                    Bx{�0�  �          @��\@0  �z���
��  C���@0  ��\)�%��C��)                                    Bx{�?T  �          @��@G�����p��33C��q@G���33�N�R�OQ�C���                                    Bx{�M�  �          @�Q�@C�
�$zΌ�����
C��@C�
��p������RC�B�                                    Bx{�\�  �          @��@Q��)�������ffC��=@Q녿�\)�����HC�'�                                    Bx{�kF  �          @���@Z�H�)�����
��\)C�!H@Z�H����G�����C���                                    Bx{�y�  �          @���@]p��*�H��
=�qG�C�#�@]p���Q��(���
=C�]q                                    Bx{���  �          @�Q�@Vff�7
=�p���@��C��H@Vff����33�أ�C�Q�                                    Bx{��8  �          @�\)@O\)�=p��k��=C��3@O\)�33����ffC�U�                                    Bx{���  �          @��@W��:�H��z��iC���@W��
=q�����C��f                                    Bx{���  �          @�  @`  �:=q�������
C�%@`  �z��(���C���                                    Bx{��*  �          @���@`���AG���z��`  C��H@`������z���ffC��                                     Bx{���  T          @�(�@W��%���G����
C�N@W������+����C�Q�                                    Bx{��v  T          @�=q@J�H�?\)���H�xQ�C�U�@J�H����
=��\)C��)                                    Bx{��  T          @�z�@9���"�\�������C�h�@9�������
=��Q�C���                                    Bx{���  T          @[�@+��   �#�
�,z�C�� @+�������ř�C�                                    Bx{�h  �          @W
=@&ff�������
=C��@&ff��G���z����C���                                    Bx{�  �          @L(�@����R����
ffC��@����Ϳ��\��G�C��)                                    Bx{�)�  �          @QG�@�����<�?
=qC�l�@�Ϳ����aG��{33C���                                    Bx{�8Z  �          @Z�H@,��� �׾�{���\C���@,�Ϳ�
=����33C�
                                    Bx{�G   �          @]p�@/\)���
������C�xR@/\)���׿������C�˅                                    Bx{�U�  �          @e@'
=����  �ƣ�C���@'
=�������{C���                                    Bx{�dL  �          @mp�@!녿�  ����C���@!녿8Q��%��1C��                                    Bx{�r�  �          @Mp�?�=q�������AffC��)?�=q=#�
�(Q��\�R?�33                                    Bx{���  �          @fff@���
=�{���C���@���{�'��;z�C��)                                    Bx{��>  �          @tz�@5��˅��������C�c�@5�����(�� \)C�f                                    Bx{���  �          @w�@J�H���
��=q��p�C�)@J�H�5��C���                                    Bx{���  �          @�G�@B�\��ff����Q�C���@B�\�L��� ���\)C���                                    Bx{��0  �          @�G�@.{��p��+��  C��\@.{�#�
�QG��F  C�Z�                                    Bx{���  �          @�  @B�\����33����C�y�@B�\��ff�1G��"C��H                                    Bx{��|  T          @�p�@:�H�z�������C�` @:�H�s33�333�)\)C���                                    Bx{��"  �          @��H@?\)�������C�(�@?\)�E��*�H�#��C���                                    Bx{���  �          @�(�@?\)�������C���@?\)�W
=�-p��$G�C�0�                                    Bx{�n  �          @�(�@C33��
��{��{C��q@C33����$z���
C�w
                                    Bx{�  �          @��\@A녿���Q���\)C���@A녿c�
�%��
=C��                                    Bx{�"�  �          @�  @N�R����ff�ȣ�C��)@N�R�����!G���C��H                                    Bx{�1`  �          @�
=@U���׿�p���{C���@U���G����(�C�ٚ                                    Bx{�@  �          @�(�@Q��{�����~�HC��)@Q녿�G���p����C���                                    Bx{�N�  �          @���@W�������lz�C���@W���  ��33���
C��                                    Bx{�]R  �          @�\)@S�
�������s
=C�ff@S�
��\)�   ��ffC��\                                    Bx{�k�  T          @��@I���Vff=#�
>�C�� @I���A녿�
=��z�C�                                    Bx{�z�  �          @���@@  �X��>�\)@a�C��
@@  �K���p��{�C���                                    Bx{��D  �          @���@Mp��tz�?(�@�{C��@Mp��mp���\)�NffC�t{                                    Bx{���  �          @�Q�@c33�vff�#�
����C�Y�@c33�]p���Q�����C��                                    Bx{���  �          @���@w
=�]p��!G�����C��@w
=�5�����C�ٚ                                    Bx{��6  �          @�ff@^{�w�>W
=@��C��\@^{�dz��G����C�
                                    Bx{���  �          @�p�@Vff�x��?�@ʏ\C�W
@Vff�p  �����V{C��                                     Bx{�҂  �          @��R@Y���x��?z�@��C��3@Y���p  ��Q��S33C��                                    Bx{��(  �          @�p�@`  �r�\>�
=@�z�C�Y�@`  �e����f�HC�"�                                    Bx{���  �          @�{@.{��33?@  Ap�C��@.{������R�]�C�Y�                                    Bx{��t  �          @�{@<(����R?B�\A�HC�z�@<(�������P��C���                                    Bx{�  �          @�{@G
=��=q?8Q�A ��C���@G
=�~{��33�L��C��                                    Bx{��  �          @�@fff�l��>�G�@�(�C�!H@fff�`�׿��R�\��C��q                                    Bx{�*f  �          @��@^{�z�H?:�H@�C�@^{�u��=q�<Q�C�\                                    Bx{�9  �          @��H@qG��qG�>u@$z�C�w
@qG��_\)���H�}G�C���                                    Bx{�G�  �          @���@s33�hQ�>k�@\)C�'�@s33�W
=��33�w�C�Ff                                    Bx{�VX  �          @�\)@u�x��>�(�@��C�U�@u�j�H����b{C�#�                                    Bx{�d�  �          @�33@fff�x��?+�@�C�b�@fff�r�\��\)�B=qC�Ǯ                                    Bx{�s�  �          @���@\(��p��?:�HA33C�9�@\(��l(���G��5p�C�y�                                    Bx{��J  �          @�z�@I���x��?�{AH��C�~�@I���~{�:�H��C�,�                                    Bx{���  �          @�z�@Z=q����?�33AE�C�'�@Z=q����B�\�G�C���                                    Bx{���  �          @�  @c33���?���A8  C��{@c33���
�Tz��
ffC�\)                                    Bx{��<  �          @�G�@Mp����?���A�p�C��@Mp�������H����C�                                    Bx{���  �          @�G�@L(�����?��
A�ffC�^�@L(���  ����G�C���                                    Bx{�ˈ  �          @���@N�R��ff?���A��
C��@N�R���R����{C��{                                    Bx{��.  T          @��@S�
���R?�  Ay�C�q@S�
���
=q��33C�e                                    Bx{���  �          @��@P  ��Q�?��HAt  C��@P  ��
=����ƸRC�
=                                    Bx{��z  �          @�=q@N{����?���Aq�C�ff@N{����!G���  C���                                    Bx{�   �          @��\@Mp���{?��A���C�� @Mp���G���\)�9��C���                                    Bx{��  �          @��H@R�\���?�(�A��C�*=@R�\��������Z�HC�q                                    Bx{�#l  �          @��\@\�����?��A��C�*=@\�����
��Q��k�C�%                                    Bx{�2  �          @�33@j=q�}p�?�  Ax��C�aH@j=q���R��G���G�C��H                                    Bx{�@�  �          @�\)@\)���
@�\A�Q�C�.@\)���ͽ#�
�ǮC���                                    Bx{�O^  �          @��@"�\��ff@�A��RC�1�@"�\��
=��Q�Y��C��R                                    Bx{�^  �          @���@*=q��ff@�\A���C��H@*=q���
�����>�RC��q                                    Bx{�l�  �          @�  @AG���
=?�(�A���C�
=@AG���(�����#33C���                                    Bx{�{P  �          @�\)@P�����?��A��\C���@P����ff����%C�^�                                    Bx{���  �          @�33@G
=��=q?�(�A�33C��q@G
=��(���
=���\C��\                                    Bx{���  
�          @�
=@7
=����?�ffA�\)C��{@7
=���;�33�hQ�C���                                    Bx{��B  �          @�  @>{��Q�?�A��C�t{@>{��(������E�C�Z�                                    Bx{���  �          @�p�@Dz���ff?ٙ�A���C�L�@Dz������\���C�xR                                    Bx{�Ď  �          @�(�@[���(�?�33A���C��@[����׾����7�C�޸                                    Bx{��4  �          @��@/\)����?���AmC���@/\)��G��E����C��\                                    Bx{���  �          @�G�@  ��G�?�G�A��HC�33@  ��\)�8Q���=qC��q                                    Bx{���  �          @�z�@������?���A�=qC�@ @����  �!G���Q�C���                                    Bx{��&  �          @�\)@ff��(�?�  A�33C�s3@ff��p��������C�                                    Bx{��  �          @���@�R���?�G�A�
=C��@�R�������G�C�c�                                    Bx{�r  �          @��H@   ���?���A���C�  @   ��\)��\���C�Z�                                    Bx{�+  T          @�G�@z���=q@A���C�t{@z����þk��
=C�ff                                    Bx{�9�  �          @��H@���33@33A�z�C���@���z�u�(�C�|)                                    Bx{�Hd  �          @���@33��33@�A\C�J=@33����#�
�ǮC��                                    Bx{�W
  �          @�
=@
=��@33A�G�C�\)@
=���R���Ϳ��
C�(�                                    Bx{�e�  �          @��\@ff��  @Q�A�\)C�+�@ff��=q�u��RC��                                    Bx{�tV  �          @�@p���@z�A�\)C��@p���
=���
�W
=C�xR                                    Bx{���  �          @�{?�\)��G�@7
=A�  C�.?�\)��33>�@��
C��                                     Bx{���  �          @�  ?�  ��z�@9��A���C�XR?�  ���R>�ff@��RC��                                    Bx{��H  �          @��R?\��
=@5A�Q�C���?\��  >�33@`  C��
                                    Bx{���  T          @�\)?�p���p�@E�B�
C�e?�p���=q?z�@���C�@                                     Bx{���  �          @���@ff��p�@1�A��C��@ff��p�>��R@FffC��3                                    Bx{��:  �          @���@��
��  ?�Q�A5C�f@��
��=q�h���
{C�˅                                    Bx{���  T          @���@����r�\?
=q@���C�e@����g
=���\�B�RC��                                    Bx{��  �          @Å@�Q��s33>\@c�
C���@�Q��b�\���V�HC���                                    Bx{��,  �          @Å@�(��~{>��@s�
C��{@�(��mp���(��_33C��=                                    Bx{��  �          @���@��H���\>�
=@y��C�/\@��H�s�
��G��d(�C�&f                                    Bx{�x  �          @�@������?(�@��
C��)@����~{����O�C�h�                                    Bx{�$  �          @\@�p����?(��@�\)C���@�p��|�Ϳ����H��C�)                                    Bx{�2�  �          @�  @�����?G�@�C��=@��������p��>=qC�.                                    Bx{�Aj  �          @��H@������R?p��A=qC��{@��������\)�)�C���                                    Bx{�P  �          @��
@����ff?Tz�@�G�C��@�������(��7�
C�XR                                    Bx{�^�  �          @ƸR@�����=L��>�
=C��R@���l�Ϳ�����C��)                                    Bx{�m\  T          @�ff@�����\����
=C�e@���c33� ����p�C�U�                                    Bx{�|  �          @ƸR@�(���(��#�
��Q�C�#�@�(��e���
����C�"�                                    Bx{���  �          @Ǯ@�z���p�>L��?���C�H@�z��r�\��\����C�b�                                    Bx{��N  �          @ȣ�@��
��?@  @�33C���@��
�������
�=�C�b�                                    Bx{���  �          @���@����\)>�ff@��C��)@���|�Ϳ����hQ�C��3                                    Bx{���  T          @�{@�����ff�#�
�uC���@����n{��(���z�C�L�                                    Bx{��@  �          @Å@�����ͽ��
�=p�C���@���hQ�� ����G�C��                                     Bx{���  �          @���@u���?���AZ=qC�8R@u����H�=p���Q�C���                                    Bx{��  T          @�\)@j=q��{@A�G�C���@j=q����    �#�
C��                                    Bx{��2  �          @�{@|����Q�@�A��
C�c�@|����������z�C��                                     Bx{���  �          @�ff@Z�H����@�A���C���@Z�H���H�L�Ϳ�\C��\                                    Bx{�~  �          @�{@XQ�����@\)A�\)C�L�@XQ���{=u?��C���                                    Bx{�$  �          @�p�@g���G�?�(�A��C�H�@g����Ǯ�g�C�"�                                    Bx{�+�  �          @�p�@�  ��
=?���AJ=qC��@�  ���\�h����C��3                                    Bx{�:p  T          @ƸR@�  ���?\Ab�\C��
@�  ���E���33C�C�                                    Bx{�I  �          @�{@p����33?�(�A���C��H@p�����
�#�
��
=C��
                                    Bx{�W�  �          @�@l(����R?��Ag33C��@l(���(��Y����p�C��=                                    Bx{�fb  �          @�\)@~{��ff?�\A��\C��
@~{���׿���33C��                                    Bx{�u  T          @�p�@l����33@  A�(�C�5�@l�����ͽu��C���                                    Bx{���  �          @���@Z=q����@��A��RC�)@Z=q��������4z�C��R                                    Bx{��T  �          @���@S33��ff@(�A��RC��f@S33����\)�)��C�@                                     Bx{���  �          @Å@`  ��33?�
=A���C��@`  ��
=�����(�C���                                    Bx{���  �          @�
=@p  ���\>��@{�C��@p  ��\)��ff��\)C���                                    Bx{��F  �          @�\)@����ff    ��C���@���l�Ϳ�p���G�C�q�                                    Bx{���  �          @��@����\(�?=p�@�C���@����XQ�xQ����C��                                    Bx{�ے  �          @���@�����33�L�Ϳ�=qC�T{@����o\)��R����C�z�                                    Bx{��8  �          @��@����{?\)@��\C��f@���}p����R�d  C��
                                    Bx{���  �          @�(�@�ff��{>�
=@|(�C�c�@�ff�x�ÿ�{�s\)C�p�                                    Bx{��  �          @�z�@�z���\)?+�@���C�
=@�z���G���z��Tz�C��                                    Bx{�*  �          @�z�@�=q���H>���@5�C�p�@�=q�~{������C���                                    Bx{�$�  �          @��H@��
��\)�#�
��
=C���@��
�l����\���HC��)                                    Bx{�3v  T          @\@�ff��(�>��?�C���@�ff�mp�������C��                                    Bx{�B  �          @\@�����{>k�@
�HC�33@����r�\���
��ffC���                                    Bx{�P�  
�          @��
@��H����?�@�p�C��{@��H���׿����mp�C��)                                    Bx{�_h  �          @���@�(���{?fffA33C�  @�(����
�����4  C�aH                                    Bx{�n  T          @Å@��
����>k�@(�C��
@��
�w��������\C�>�                                    Bx{�|�  �          @�=q@�(���ff>#�
?ǮC��@�(��qG��������
C��                                     Bx{��Z  T          @��@�(���p���\)�&ffC�5�@�(��hQ���\��C�(�                                    Bx{��   �          @��@�33��녿#�
��33C�|)@�33�QG��\)�ģ�C���                                    Bx{���  �          @�Q�@�z��q녾���33C�g�@�z��Fff�{��C�!H                                    Bx{��L  �          @���@�(��p�׿E���z�C�xR@�(��<(��{���
C��                                    Bx{���  �          @���@������\>�G�@��C��
@����s33�Ǯ�p(�C��q                                    Bx{�Ԙ  �          @�  @�z���Q�?h��A
=C�˅@�z��}p���{�*=qC���                                    Bx{��>  �          @�ff@�33��ff?n{AQ�C�&f@�33��zῗ
=�7�C�]q                                    Bx{���  �          @��@y����33?��\AC�C��{@y����p��u��C��{                                    Bx{� �  T          @���@z�H��p�?�
=A5�C�˅@z�H��{�����#33C��3                                    Bx{�0  �          @���@x������?�
=AZ�\C���@x�������Y��� ��C�B�                                    Bx{��  T          @���@�����G�?�  A?�
C�L�@������ͿTz���{C���                                    Bx{�,|  
�          @���@|����\)?�\)Ay��C��@|����������
=C���                                    Bx{�;"  �          @�Q�@j�H��  @�A��C�e@j�H���R��  �
=C��R                                    Bx{�I�  �          @�G�@c�
��\)?�\)A���C�:�@c�
���\�   ���C�1�                                    Bx{�Xn  �          @�G�@fff���?�\)A,��C�ٚ@fff��(����R�>�HC��                                    Bx{�g  T          @�=q@|(����\?��Ak�
C�*=@|(���G��8Q��ٙ�C�}q                                    Bx{�u�  T          @�=q@s33���?���A"�\C��@s33��녿�G��@(�C��                                    Bx{��`  �          @��H@p����p�?��
A  C�n@p�����H��=q�I�C���                                    Bx{��  T          @\@�=q���?p��A�HC�^�@�=q��=q���
�C
=C��f                                    Bx{���  �          @���@u����H?\(�A�C��@u���{��Q��\z�C�ff                                    Bx{��R  �          @\@h����G�?Q�@�\)C��H@h�����H��=q�p��C�4{                                    Bx{���  �          @���@fff��녾�=q�#�
C�g�@fff���H�"�\��  C��=                                    Bx{�͞  "          @�Q�@qG�����W
=�   C�|)@qG��\)��H��=qC���                                    Bx{��D  �          @���@��
���>�?�(�C���@��
�{��   ����C�+�                                    Bx{���  �          @���@�{���H�8Q���HC�@�{�n{�\)��(�C�7
                                    Bx{���  O          @���@�Q����׾\)��{C�˅@�Q��y����\���\C��=                                    Bx{�6  
�          @���@�����ͿJ=q��C�f@���QG��,(���G�C�k�                                    Bx{��  �          @���@�����  �z�H�ffC��@����A��1����C��R                                    Bx{�%�  T          @���@��
���\�333���C�}q@��
�O\)�$z���ffC��
                                    Bx{�4(  T          @�G�@������׿xQ��  C���@����C�
�1��ۙ�C���                                    Bx{�B�  �          @���@��
���ÿ����'�C��f@��
�@���9����p�C���                                    Bx{�Qt  �          @�=q@���s33���� Q�C�>�@���5��0�����
C�@                                     Bx{�`  �          @�=q@�
=�l�Ϳh���
ffC��3@�
=�3�
�%���\)C���                                    Bx{�n�  �          @�=q@���fff�Tz����
C���@���0���p���p�C��                                    Bx{�}f  T          @���@����l�;aG��
=C��@����H�ÿ�p����RC�U�                                    Bx{��  �          @�Q�@�33�k���\)�-�C���@�33�+��0����
=C��                                     Bx{���  T          @�\)@���hQ�aG��
=C��@���0���!G���G�C���                                    Bx{��X  �          @�  @�
=�x�ÿW
=� ��C��H@�
=�@  �'
=�Σ�C��                                    Bx{���  "          @�G�@���|(�=�Q�?W
=C���@���`  ��������C�B�                                    Bx{�Ƥ  T          @\@�ff���R?�G�A?�C�w
@�ff�����k���
C�,�                                    Bx{��J  
�          @��@�ff���?uA�C�` @�ff��p������7�
C��
                                    Bx{���  T          @�33@|�����@ ��A�z�C���@|�����
��  ��
C�Ff                                    Bx{��  T          @Å@q���{@�A�33C��@q���G�<��
>.{C�"�                                    Bx{�<  �          @�33@�G���(�?�A�\)C�,�@�G���G����R�8��C��                                    Bx{��  �          @�=q@x�����?�33A�=qC�AH@x����(��\�g
=C�H                                    Bx{��  �          @��@����z�?p��A�
C��@�����\���2�\C�C�                                    Bx{�-.  �          @�Q�@�����ff>��?�33C��q@����p  ��33���
C�\)                                    Bx{�;�  �          @���@��R��Q�?z�@���C�f@��R�r�\���Z�HC�Ф                                    Bx{�Jz  �          @���@�p��qG��aG���C��@�p��L��� ������C���                                    Bx{�Y   
�          @���@����]p��\�fffC�c�@����6ff�   ����C��                                    Bx{�g�  
�          @�Q�@�33�U��Q��7
=C��R@�33�ff�*=q���C�3                                    Bx{�vl  "          @���@�ff�=p���=q�}C���@�ff��=q�3�
��33C�Z�                                    Bx{��  "          @�=q@�z��I������up�C��@�z��G��7���\C�B�                                    Bx{���  �          @���@�
=���G���{C�xR@�
=�aG��2�\�癚C�                                      Bx{��^  �          @�
=@���R�\�����HC��@���   �L���	(�C�h�                                    Bx{��  �          @�
=@�G�����9������C���@�G���  �]p��  C�l�                                    Bx{���  T          @�@�G��k��fff��HC�y�@�G�?���P���ffA�{                                    Bx{��P  "          @�@�
=�Tz������
C�<)@�
=>�G��!���
=@��                                    Bx{���  T          @�33@��R�33�"�\���C��\@��R�\)�Mp��=qC�k�                                    Bx{��  �          @��
@��R���
�  ����C���@��R����5�����C�+�                                    Bx{��B  �          @�33@����&ff�����0��C�˅@�����G��(����C���                                    Bx{��  
�          @��H@���!G��У�����C�˅@�����)�����C���                                    Bx{��  �          @���@l���q녿���[�C�5�@l���+��=p����C��                                    Bx{�&4  
�          @��H@Tz���p���p��vffC�y�@Tz��l(��p���33C�                                    Bx{�4�  �          @�
=@~�R���ÿE����RC�P�@~�R�I���'���ffC��=                                    Bx{�C�  �          @�
=@�z��w��:�H��C�t{@�z��B�\�!G����HC��H                                    Bx{�R&  �          @�p�@���w
=�Q��Q�C�8R@���?\)�%���
=C��R                                    Bx{�`�  �          @�@��H�g
=�������C�"�@��H�7��G���ffC�Ff                                    Bx{�or  
�          @��@�ff�^{�\�w�C�  @�ff�7
=�G����C��f                                    Bx{�~  �          @���@����b�\��\)�6ffC���@����=p����H��Q�C�f                                    Bx{���  T          @�@�  �o\)�#�
����C�O\@�  �L�Ϳ�����33C���                                    Bx{��d  �          @��\@}p��}p��u�z�C�� @}p��\(���(����\C��                                    Bx{��
  
�          @�
=@����QG��������HC��=@������?\)���C�aH                                    Bx{���  �          @��@x���XQ��33��  C���@x���	���E��
��C�~�                                    Bx{��V  
�          @�p�@|(��Dz���
��p�C�@|(���Q��R�\��C�c�                                    Bx{���  
�          @�ff@u�n{?B�\A�C��
@u�g�����B�RC�XR                                    Bx{��  T          @�@��
�_\)�\)���RC��)@��
�?\)������C��                                    Bx{��H  �          @�
=@���C�
��{�hQ�C��)@��� �׿��
��
=C�e                                    Bx{��  �          @�Q�@��
=�����\)C�9�@���\��z�����C�)                                    Bx{��  �          @�ff@�G��AG��\)���HC�*=@�G��%��˅��G�C�4{                                    Bx{�:  �          @�@e�����>��@.{C��\@e��i����p���p�C�9�                                    Bx{�-�  	�          @�ff@j�H�vff?
=@�(�C���@j�H�j=q����h��C��=                                    Bx{�<�  
_          @�  @q��|(���\)�:�HC��@q��Z=q��(���G�C��3                                    Bx{�K,  �          @�ff@j�H�~�R>���@�C�Q�@j�H�j�H��=q���C�w
                                    Bx{�Y�  �          @�\)@l���x��?G�Az�C��\@l���q녿����LQ�C�5�                                    Bx{�hx  �          @�
=@�33�_\)?uA!��C���@�33�aG��Y����C���                                    Bx{�w  �          @�\)@xQ��mp�?�  A)��C�"�@xQ��n�R�k��
=C��                                    Bx{���  �          @�
=@�G��h��?   @�\)C��q@�G��[���=q�aG�C�ٚ                                    Bx{��j  �          @�@�33�b�\>�?�z�C��
@�33�J=q������G�C�,�                                    Bx{��  �          @�@c33���H>\)?�
=C�|)@c33�h�ÿ�{���HC�"�                                    Bx{���  �          @�ff@A���  �#�
��C��
@A��z=q�{���C���                                    Bx{��\  �          @�ff@q��u�������\C�S3@q��I���p���p�C��                                    Bx{��  �          @�ff@@����  �#�
��Q�C��H@@���{��(����RC��{                                    Bx{�ݨ  �          @�{@-p���{��
=��C�
=@-p��y���(����ffC�n                                    Bx{��N  �          @�{@333���H�@  ��\)C���@333�i���8Q�� G�C��                                     Bx{���  �          @�@(���  �����xz�C�` @(��\���dz��$��C�33                                    Bx{�	�  �          @�(�@�����
��{���C�@���P���h���+33C��                                    Bx{�@  �          @��H@ff��{��p����C��=@ff�;��x���<�RC�ٚ                                    Bx{�&�  �          @�33?��
��z�����G�C�  ?��
�.{��p��P=qC��{                                    Bx{�5�  �          @��H@n{�q녾�{�hQ�C�@ @n{�I���Q�����C��H                                    Bx{�D2  �          @�33@b�\�\)�L���C��@b�\�X���ff��p�C�)                                    Bx{�R�  �          @�33@n{�j=q��
=���C���@n{�@  �Q����RC��=                                    Bx{�a~  �          @�z�@�33�^�R>��?���C��)@�33�G
=��ff��Q�C�g�                                    Bx{�p$  �          @���@���Vff>�G�@�C��
@���H�ÿ��R�UC�y�                                    Bx{�~�  �          @��
@�p��S�
�aG���C��@�p��2�\�������C�                                      Bx{��p  �          @��@��\�9��������(�C�E@��\��G��3�
�=qC�K�                                    Bx{��  �          @�Q�@�  �(���{��{C��@�  ���R�3�
�z�C���                                    Bx{���  T          @���@�33�녿�
=����C�*=@�33����2�\���C�"�                                    Bx{��b  �          @�G�@q��HQ쿾�R��G�C�0�@q��G��4z��\)C��{                                    Bx{��  T          @���@e��@���{�޸RC���@e���Q��g��,��C�                                    Bx{�֮  �          @�  @y���z��#�
���
C���@y���J=q�W
=� ��C�L�                                    Bx{��T  �          @�G�@�Q�����{��=qC�@ @�Q��(��C�
�=qC�&f                                    Bx{���  T          @�=q@�p���Q��*�H��  C�|)@�p�>�  �:=q�G�@P                                      Bx{��  T          @�G�@�����\�{����C�!H@��=��
�1G���\)?�=q                                    Bx{�F  T          @�G�@�\)��\)� ������C��f@�\)�+��*=q��C��q                                    Bx{��  
�          @�G�@�  ���
�Q���{C�0�@�  ���.{��Q�C��H                                    Bx{�.�  �          @��@�G������\)C�\@�G��O\)�5�  C���                                    Bx{�=8  �          @���@�=q��H�����=qC��H@�=q���H�5����C��                                     Bx{�K�  �          @��H@�z��33���H����C�33@�z��R��\�Џ\C�,�                                    Bx{�Z�  �          @���@����R�����C��
@��u�p���C��                                    Bx{�i*  �          @���@�33��p�������HC���@�33<#�
�$z��陚>��                                    Bx{�w�  �          @�=q@�G���{�C�
���C��
@�G�?�
=�6ff�ffAu�                                    Bx{��v  "          @��
@{����\�^�R�"G�C��
@{�?G��b�\�%A4Q�                                    Bx{��  �          @�z�@\)�#�
�\(��!��C�q�@\)?����Tz��  Av=q                                    Bx{���  "          @�=q@P�׾\���H�M
=C��H@P��?���qG��7��A�33                                    Bx{��h  
�          @��\@���녿0����C�'�@����z���H��  C�S3                                    Bx{��  T          @��@��\�.{���R�Tz�C���@��\��R��=q���
C��                                    Bx{�ϴ  
�          @���@�ff����ff���C���@�ff��{����o\)C��                                    Bx{��Z  T          @��@�p���Q�=�G�?�33C�7
@�p����������C�޸                                    Bx{��   �          @��@�(���zὣ�
�Y��C��@�(���z�^�R�Q�C�L�                                    Bx{���  �          @�p�@���   ?��HAQG�C���@���=���?��
C��                                    Bx{�
L  �          @�@�녿&ff@�RA��C��@�녿�\?�{A���C�h�                                    Bx{��  �          @�ff@�p��W
=@(�AٮC��@�p���z�?޸RA��C��                                    Bx{�'�  �          @�p�@��R��Q�?�ffA��
C��@��R�$z�?&ff@ۅC��3                                    Bx{�6>  �          @��@��-p�?�=qA<z�C�>�@��9����Q��vffC�b�                                    Bx{�D�  �          @��@�ff�@��?   @���C��
@�ff�8Q쿁G��,��C��                                    Bx{�S�  Y          @�33@|���]p��&ff��33C�h�@|���.{�\)���C���                                    Bx{�b0  "          @���@tz�?L��@uB2(�A>=q@tzῑ�@p  B-(�C��                                    Bx{�p�  �          @�Q�@y��?^�R@qG�B-�AH��@y����ff@n{B*�\C�~�                                    Bx{�|  �          @��@�p�=�\)@a�B �H?��\@�p��У�@HQ�B��C�T{                                    Bx{��"  �          @�{@��
�#�
@J=qBp�C���@��
��z�@,��A�  C��H                                    Bx{���  �          @�\)@|��?�
=@b�\B"�A��@|�Ϳ(��@j�HB*�C�9�                                    Bx{��n  �          @�  @h��?��\@z�HB6{A��@h�ÿE�@���B=C�                                    Bx{��  T          @�
=@���<��
@>{B>��R@�����z�@'�A���C�h�                                    Bx{�Ⱥ            @�p�@��R�
=@>{B�C�5�@��R����@z�A�  C�8R                                    Bx{��`  �          @���@�(��(�<��
>�  C�U�@�(���
=���
�9C��=                                    Bx{��  
�          @��\@�Q��{?�33A���C���@�Q��E�>�@��C�3                                    Bx{���  
�          @�G�@��\�=q?��HA�{C���@��\�C�
?�@��
C��3                                    Bx{�R  "          @�\)@�\)�:=q?Y��A{C���@�\)�>{�&ff��{C�z�                                    Bx{��  �          @�
=@w
=�[�?(��@陚C�1�@w
=�U�����<(�C���                                    Bx{� �  T          @�@����0��?
=q@���C��\@����+��Y����C���                                    Bx{�/D  �          @�p�@��\�&ff�\)��G�C��@��\� �׿�p���p�C��                                    Bx{�=�  �          @��
@�\)����(������C���@�\)>����5����@�Q�                                    Bx{�L�  �          @��@!���p�<#�
=���C��
@!��j�H���R��Q�C�C�                                    Bx{�[6  �          @��
@Vff�j=q����O
=C�:�@Vff�*�H�0  ���C��
                                    Bx{�i�  Y          @��H@S33�i���k��)��C�R@S33�0���#33����C��                                    Bx{�x�  T          @�(�@aG��k��aG��   C��H@aG��HQ������{C�4{                                    Bx{��(  �          @�@<(������=q�k�C�Ǯ@<(��>�R�HQ��(�C�N                                    Bx{���  T          @�p�@
�H�Mp��L(��
=C��@
�H������H�o
=C��
                                    Bx{��t  
�          @��\?�ff�}p��/\)�p�C��\?�ff�
�H�����h��C��H                                    Bx{��  T          @�33@���e�2�\��HC�ff@�ÿ�=q��p��_��C�                                    